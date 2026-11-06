"""A tiny synthetic driving world: a curved road, some box-shaped cars, and a
ring of pinhole cameras on the ego car.

Coordinate frames
-----------------
ego frame : x forward, y left, z up (metres). The ego car sits at the origin.
camera    : x right, y down, z forward (standard pinhole convention).

A camera is described by
    K : 3x3 intrinsics (the lens)
    R : 3x3 rotation, camera -> ego (which way it points)
    t : 3   translation, camera centre in the ego frame (where it sits on the car)
so a camera-frame point p maps to the ego frame as  R @ p + t, and a pixel
(u, v) at depth d back-projects to  R @ (d * K^-1 [u, v, 1]) + t.
"""
from dataclasses import dataclass

import numpy as np

# ---- top-down grid ---------------------------------------------------------
XMIN, XMAX = -25.0, 25.0
YMIN, YMAX = -25.0, 25.0
RES = 0.5
NX = int(round((XMAX - XMIN) / RES))  # 100 cells along x (forward)
NY = int(round((YMAX - YMIN) / RES))  # 100 cells along y (left)

# ---- cameras ---------------------------------------------------------------
IMG_H, IMG_W = 48, 96
CAM_HEIGHT = 1.6

SKY_TOP = np.array([0.35, 0.55, 0.85])
SKY_HORIZON = np.array([0.75, 0.85, 0.95])
ROAD = np.array([0.33, 0.33, 0.35])
GRASS = np.array([0.30, 0.48, 0.22])
LANE = np.array([0.92, 0.92, 0.85])
HAZE = np.array([0.78, 0.82, 0.88])


def intrinsics(fov_deg, W=IMG_W, H=IMG_H):
    f = (W / 2) / np.tan(np.radians(fov_deg) / 2)
    return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])


def cam_rotation(yaw, pitch=0.0):
    """Camera -> ego rotation for a camera looking along `yaw` (radians,
    counter-clockwise from +x), tilted down by `pitch`."""
    fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    right = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
    down = np.array([0.0, 0.0, -1.0])
    f2 = np.cos(pitch) * fwd + np.sin(pitch) * down
    d2 = -np.sin(pitch) * fwd + np.cos(pitch) * down
    return np.stack([right, d2, f2], axis=1)  # columns = camera axes in ego frame


@dataclass
class Rig:
    K: np.ndarray  # (N, 3, 3)
    R: np.ndarray  # (N, 3, 3)
    t: np.ndarray  # (N, 3)
    names: list

    @property
    def n(self):
        return len(self.K)


def make_rig(yaws_deg, fov_deg=70.0, height=CAM_HEIGHT, names=None):
    Ks, Rs, ts = [], [], []
    for yd in yaws_deg:
        y = np.radians(yd)
        Ks.append(intrinsics(fov_deg))
        Rs.append(cam_rotation(y))
        # cameras sit on an ellipse around the car body
        ts.append(np.array([1.0 * np.cos(y), 0.5 * np.sin(y), height]))
    names = names or [f"cam{int(y)}" for y in yaws_deg]
    return Rig(np.stack(Ks), np.stack(Rs), np.stack(ts), list(names))


# the "training car": 6 cameras, 70 deg FOV each, 60 deg apart
DEFAULT_RIG = make_rig(
    [0, 60, 120, 180, -120, -60],
    names=["front", "front-left", "back-left", "back", "back-right", "front-right"],
)

# a "different car": 5 wider cameras at completely different angles
OTHER_RIG = make_rig(
    [36, 108, 180, -108, -36], fov_deg=80.0,
    names=["A", "B", "C", "D", "E"],
)


def jitter_rig(rig, rng, yaw_deg=3.0):
    """Randomly nudge each camera's yaw (train-time augmentation). The model is
    always told the true (jittered) extrinsics."""
    Rs = []
    for R in rig.R:
        a = np.radians(rng.uniform(-yaw_deg, yaw_deg))
        Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        Rs.append(Rz @ R)
    return Rig(rig.K.copy(), np.stack(Rs), rig.t.copy(), rig.names)


# ---- scenes ----------------------------------------------------------------
@dataclass
class Scene:
    road: tuple   # (c0, c1, c2, half_width): centre line y = c0 + c1 x + c2 x^2
    boxes: list   # [(cx, cy, yaw, length, width, height, rgb)]

    def road_centre(self, x):
        c0, c1, c2, _ = self.road
        return c0 + c1 * x + c2 * x * x


def sample_scene(rng):
    road = (rng.uniform(-2, 2), rng.uniform(-0.15, 0.15), rng.uniform(-0.012, 0.012),
            rng.uniform(4.0, 7.0))
    scene = Scene(road, [])
    half = road[3]
    n_target = rng.integers(3, 10)
    for _ in range(200):
        if len(scene.boxes) >= n_target:
            break
        x = rng.uniform(-24, 24)
        y = scene.road_centre(x) + rng.uniform(-half + 1.1, half - 1.1)
        if abs(x) < 6 and abs(y) < 3:          # don't spawn on top of the ego car
            continue
        if any(np.hypot(x - b[0], y - b[1]) < 6.0 for b in scene.boxes):
            continue
        heading = np.arctan(road[1] + 2 * road[2] * x) + rng.normal(0, 0.08)
        if rng.random() < 0.5:
            heading += np.pi
        L, W, H = rng.uniform(3.8, 5.2), rng.uniform(1.7, 2.1), rng.uniform(1.4, 1.9)
        rgb = rng.uniform(0.05, 0.95, 3)
        scene.boxes.append((x, y, heading, L, W, H, rgb))
    return scene


# ---- rendering (a tiny ray caster) -----------------------------------------
def _pixel_grid(H, W):
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    return np.stack([u, v, np.ones_like(u)], -1).reshape(-1, 3)


def render(scene, rig, rng=None, noise=0.02):
    """Return images (N, H, W, 3) in [0, 1] and true depth (N, H, W).
    Depth is distance along the camera's optical axis (the `d` LSS predicts);
    sky pixels get +inf."""
    H, W = IMG_H, IMG_W
    pix = _pixel_grid(H, W)
    imgs = np.zeros((rig.n, H * W, 3))
    depths = np.full((rig.n, H * W), np.inf)
    c0, c1, c2, half = scene.road
    for i in range(rig.n):
        rays = pix @ np.linalg.inv(rig.K[i]).T @ rig.R[i].T   # (P,3), camera-z = 1
        o = rig.t[i]
        best = np.full(len(pix), np.inf)
        # sky gradient
        elev = np.clip(rays[:, 2] / np.linalg.norm(rays[:, :2], axis=1), 0, 1)
        col = SKY_HORIZON + (SKY_TOP - SKY_HORIZON) * elev[:, None] ** 0.5

        # ground plane z = 0
        dz = rays[:, 2]
        with np.errstate(divide="ignore"):
            s = np.where(dz < -1e-6, -o[2] / dz, np.inf)
        g = np.isfinite(s)
        p = o + s[g, None] * rays[g]
        off = p[:, 1] - (c0 + c1 * p[:, 0] + c2 * p[:, 0] ** 2)
        gcol = np.where((np.abs(off) < half)[:, None], ROAD, GRASS)
        dashed = (np.abs(off) < 0.12) & ((p[:, 0] % 6.0) < 3.0)
        edge = np.abs(np.abs(off) - (half - 0.25)) < 0.1
        gcol[dashed | edge] = LANE
        col[g] = gcol
        best[g] = s[g]

        # boxes (oriented, via slab test in each box's local frame)
        for (bx, by, yaw, L, Wd, Hb, rgb) in scene.boxes:
            cy_, sy_ = np.cos(-yaw), np.sin(-yaw)
            Rl = np.array([[cy_, -sy_, 0], [sy_, cy_, 0], [0, 0, 1]])
            ol = Rl @ (o - np.array([bx, by, 0.0]))
            dl = rays @ Rl.T
            lo = np.array([-L / 2, -Wd / 2, 0.0])
            hi = np.array([L / 2, Wd / 2, Hb])
            dl_safe = np.where(np.abs(dl) < 1e-9, 1e-9, dl)
            t0, t1 = (lo - ol) / dl_safe, (hi - ol) / dl_safe
            tmin, tmax = np.minimum(t0, t1), np.maximum(t0, t1)
            enter, exit_ = tmin.max(1), tmax.min(1)
            hit = (enter < exit_) & (enter > 0) & (enter < best)
            if not hit.any():
                continue
            axis = tmin[hit].argmax(1)
            shade = np.array([0.85, 0.65, 1.0])[axis]
            pz = (ol + enter[hit, None] * dl[hit])[:, 2]
            shade = shade * np.where(pz < 0.45, 0.35, 1.0)    # dark "wheel" band
            shade = shade * np.where((pz > 0.9 * Hb) & (axis != 2), 0.6, 1.0)  # windows
            col[hit] = rgb * shade[:, None]
            best[hit] = enter[hit]

        # atmospheric haze: far things fade toward grey
        fin = np.isfinite(best)
        a = 1 - np.exp(-best[fin] / 90.0)
        col[fin] = col[fin] * (1 - a[:, None]) + HAZE * a[:, None]
        imgs[i] = col
        depths[i] = best
    if rng is not None and noise > 0:
        imgs = imgs + rng.normal(0, noise, imgs.shape)
    return (np.clip(imgs, 0, 1).reshape(rig.n, H, W, 3).astype(np.float32),
            depths.reshape(rig.n, H, W).astype(np.float32))


# ---- ground truth top-down maps --------------------------------------------
def cell_centres():
    xs = XMIN + (np.arange(NX) + 0.5) * RES
    ys = YMIN + (np.arange(NY) + 0.5) * RES
    return np.meshgrid(xs, ys, indexing="ij")


def bev_targets(scene):
    """(2, NX, NY): channel 0 = vehicle occupancy, channel 1 = drivable road."""
    X, Y = cell_centres()
    veh = np.zeros_like(X, dtype=bool)
    for (bx, by, yaw, L, Wd, _, _) in scene.boxes:
        dx, dy = X - bx, Y - by
        lx = np.cos(yaw) * dx + np.sin(yaw) * dy
        ly = -np.sin(yaw) * dx + np.cos(yaw) * dy
        veh |= (np.abs(lx) < L / 2) & (np.abs(ly) < Wd / 2)
    drv = np.abs(Y - scene.road_centre(X)) < scene.road[3]
    return np.stack([veh, drv]).astype(np.float32)


# ---- batches ---------------------------------------------------------------
def make_batch(n, rng, rig=DEFAULT_RIG, jitter=False):
    """Render `n` random scenes. Returns a dict of numpy arrays:
    imgs (n,N,3,H,W), K/R/t, target (n,2,NX,NY), depth (n,N,H,W)."""
    out = {k: [] for k in ("imgs", "K", "R", "t", "target", "depth")}
    out["scenes"] = []
    for _ in range(n):
        scene = sample_scene(rng)
        r = jitter_rig(rig, rng) if jitter else rig
        imgs, depth = render(scene, r, rng)
        out["imgs"].append(imgs.transpose(0, 3, 1, 2))
        out["depth"].append(depth)
        out["K"].append(r.K)
        out["R"].append(r.R)
        out["t"].append(r.t)
        out["target"].append(bev_targets(scene))
        out["scenes"].append(scene)
    for k in ("imgs", "K", "R", "t", "target", "depth"):
        out[k] = np.stack(out[k]).astype(np.float32)
    return out
