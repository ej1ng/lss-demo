"""Lift-Splat-Shoot, written small enough to read in one sitting.

    images (B, N, 3, H, W)  + camera calibration K, R, t
      |  CamEncoder           per-pixel feature c (C) and depth distribution alpha (D)
      |  LIFT                 outer product alpha (x) c  ->  a D x C "ray of guesses" per pixel
      |  geometry             where every (pixel, depth bin) lands in the ego frame
      |  SPLAT                sum all points that fall into the same top-down cell
      |  BEVEncoder           2D CNN on the top-down feature map
    logits (B, 2, NX, NY)     vehicle + drivable-area maps
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import world

DOWN = 4                                         # image -> feature map stride
DEPTH_BINS = torch.arange(3.0, 36.0, 1.0)        # D = 33 candidate depths (metres)


def conv_bn(cin, cout, stride=1, dil=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, padding=dil, dilation=dil, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class CamEncoder(nn.Module):
    """Image -> (depth distribution alpha, context feature c) for every feature pixel.

    Besides RGB it gets one extra channel: the vertical slope of each pixel's
    ray, (v - cy) / fy. This is a cheap stand-in for the large receptive
    field of a real backbone, and it is intrinsics-aware, so it still means
    the same thing on a camera with a different lens.
    """

    def __init__(self, D, C):
        super().__init__()
        self.D, self.C = D, C
        self.net = nn.Sequential(
            conv_bn(4, 16), conv_bn(16, 32, stride=2), conv_bn(32, 32),
            conv_bn(32, 64, stride=2), conv_bn(64, 64, dil=2),
            conv_bn(64, 64, dil=4), conv_bn(64, 64),
            nn.Conv2d(64, D + C, 1),
        )

    def forward(self, x, K):
        BN, _, H, W = x.shape
        v = torch.arange(H, device=x.device, dtype=x.dtype) + 0.5
        slope = (v[None, :] - K[:, 1, 2, None]) / K[:, 1, 1, None]        # (BN, H)
        slope = slope[:, None, :, None].expand(BN, 1, H, W)
        out = self.net(torch.cat([x, slope], 1))
        alpha = out[:, :self.D].softmax(1)                                  # (BN, D, h, w)
        c = out[:, self.D:]                                                 # (BN, C, h, w)
        return alpha, c


def frustum_points(K, R, t, h, w, depths=DEPTH_BINS):
    """Ego-frame xyz of every (camera, depth bin, feature pixel):  R (d K^-1 [u v 1]) + t.
    K, R: (B, N, 3, 3), t: (B, N, 3)  ->  (B, N, D, h, w, 3)."""
    dev = K.device
    u = (torch.arange(w, device=dev) + 0.5) * DOWN      # centre of each feature cell, in image pixels
    v = (torch.arange(h, device=dev) + 0.5) * DOWN
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    pix = torch.stack([uu, vv, torch.ones_like(uu)], -1)                    # (h, w, 3)
    rays = torch.einsum("bnij,hwj->bnhwi", torch.linalg.inv(K), pix)        # camera frame, z = 1
    pts = depths.to(dev)[None, None, :, None, None, None] * rays[:, :, None]  # (B,N,D,h,w,3)
    return torch.einsum("bnij,bndhwj->bndhwi", R, pts) + t[:, :, None, None, None, :]


def _cell_index(geom, cam_mask):
    """Flat top-down cell index of every point, and which points are kept."""
    B = geom.shape[0]
    ix = ((geom[..., 0] - world.XMIN) / world.RES).floor().long()
    iy = ((geom[..., 1] - world.YMIN) / world.RES).floor().long()
    keep = (ix >= 0) & (ix < world.NX) & (iy >= 0) & (iy < world.NY) & (geom[..., 2].abs() < 10)
    if cam_mask is not None:                       # drop whole cameras (robustness test)
        keep &= cam_mask[:, :, None, None, None].bool()
    b = torch.arange(B, device=geom.device).view(B, 1, 1, 1, 1).expand_as(ix)
    return (b * world.NX + ix) * world.NY + iy, keep


def splat(geom, feats, cam_mask=None):
    """Sum-pool every lifted point into the top-down cell beneath it (height ignored).
    geom (B,N,D,h,w,3), feats (B,N,D,h,w,C)  ->  (B, C, NX, NY)."""
    B, C = geom.shape[0], feats.shape[-1]
    idx, keep = _cell_index(geom, cam_mask)
    out = feats.new_zeros(B * world.NX * world.NY, C)
    out.index_add_(0, idx[keep], feats[keep])
    return out.view(B, world.NX, world.NY, C).permute(0, 3, 1, 2)


def splat_cumsum(geom, feats, cam_mask=None):
    """The paper's "cumsum trick": sort points by cell, take one running sum,
    and read off each cell's total as the difference at its last point."""
    B, C = geom.shape[0], feats.shape[-1]
    idx, keep = _cell_index(geom, cam_mask)
    idx, x = idx[keep], feats[keep]
    order = idx.argsort()
    idx, x = idx[order], x[order]
    cs = x.cumsum(0)
    last = torch.ones_like(idx, dtype=torch.bool)
    last[:-1] = idx[1:] != idx[:-1]                 # last point of each cell
    cs, idx = cs[last], idx[last]
    cs = torch.cat([cs[:1], cs[1:] - cs[:-1]])      # per-cell sums
    out = feats.new_zeros(B * world.NX * world.NY, C)
    out[idx] = cs
    return out.view(B, world.NX, world.NY, C).permute(0, 3, 1, 2)


class BEVEncoder(nn.Module):
    """Small U-Net on the 100 x 100 top-down feature map."""

    def __init__(self, C, n_out=2):
        super().__init__()
        self.e1 = nn.Sequential(conv_bn(C, 32), conv_bn(32, 32))
        self.e2 = nn.Sequential(conv_bn(32, 64, stride=2), conv_bn(64, 64))
        self.e3 = nn.Sequential(conv_bn(64, 128, stride=2), conv_bn(128, 128, dil=2))
        self.d2 = nn.Sequential(conv_bn(128 + 64, 64), conv_bn(64, 64))
        self.d1 = nn.Sequential(conv_bn(64 + 32, 32), conv_bn(32, 32))
        self.head = nn.Conv2d(32, n_out, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        d2 = self.d2(torch.cat([F.interpolate(e3, size=e2.shape[-2:], mode="bilinear"), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear"), e1], 1))
        return self.head(d1)


class LSS(nn.Module):
    """uniform_depth=True is the ablation: every pixel is smeared evenly along
    its ray (alpha = 1/D) instead of using the predicted depth distribution."""

    def __init__(self, C=32, uniform_depth=False):
        super().__init__()
        self.D, self.C = len(DEPTH_BINS), C
        self.uniform_depth = uniform_depth
        self.cam = CamEncoder(self.D, C)
        self.bev = BEVEncoder(C)

    def lift(self, imgs, K):
        """(B,N,3,H,W) -> alpha (B,N,D,h,w), lifted features (B,N,D,h,w,C)."""
        B, N = imgs.shape[:2]
        alpha, c = self.cam(imgs.flatten(0, 1), K.flatten(0, 1))
        h, w = alpha.shape[-2:]
        if self.uniform_depth:
            alpha = torch.full_like(alpha, 1.0 / self.D)
        lifted = alpha[:, None] * c[:, :, None]          # outer product: (BN, C, D, h, w)
        lifted = lifted.permute(0, 2, 3, 4, 1)           # (BN, D, h, w, C)
        return alpha.view(B, N, self.D, h, w), lifted.reshape(B, N, self.D, h, w, self.C)

    def forward(self, imgs, K, R, t, cam_mask=None, return_aux=False):
        alpha, lifted = self.lift(imgs, K)
        geom = frustum_points(K, R, t, *alpha.shape[-2:])
        bev = splat(geom, lifted, cam_mask)
        logits = self.bev(bev)
        if return_aux:
            return logits, {"alpha": alpha, "bev_features": bev, "geom": geom}
        return logits


def iou(logits, target, thresh=0.5):
    """Per-channel IoU over a whole batch: (2,) tensor [vehicle, drivable]."""
    pred = logits.sigmoid() > thresh
    tgt = target > 0.5
    inter = (pred & tgt).flatten(2).sum((0, 2)).float()
    union = (pred | tgt).flatten(2).sum((0, 2)).float()
    return inter / union.clamp(min=1)
