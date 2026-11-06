"""Plotting helpers. Top-down maps are always drawn with forward = up and
the car's left = left of the picture, the ego car in the middle."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from . import world
from .model import DEPTH_BINS, DOWN

EXTENT = (world.YMAX, world.YMIN, world.XMIN, world.XMAX)   # (left, right, bottom, top) in (y, x)
CAM_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#FF9DA6"]
ORANGE = "#C0531A"


def to_display(m):
    return np.asarray(m)[::-1, ::-1]


def _ego(ax):
    ax.add_patch(Rectangle((1.0, -2.3), -2.0, 4.6, color=ORANGE, zorder=5))  # (y, x) coords
    ax.set_xlabel("y (m, left +)")
    ax.set_ylabel("x (m, forward +)")


def bev_rgb(veh, drv):
    """Compose vehicle / drivable maps (values in [0,1]) into an RGB image."""
    bg = np.array([0.93, 0.95, 0.90])
    road = np.array([0.62, 0.64, 0.68])
    car = np.array([0.08, 0.13, 0.20])
    img = bg + (road - bg) * drv[..., None]
    img = img + (car - img) * veh[..., None]
    return to_display(img)


def plot_cameras(imgs, names, dropped=(), axes=None):
    """imgs: (N, 3, H, W)."""
    n = len(imgs)
    if axes is None:
        _, axes = plt.subplots(2, (n + 1) // 2, figsize=(3.2 * ((n + 1) // 2), 3.6))
    for i, ax in enumerate(np.ravel(axes)):
        ax.axis("off")
        if i >= n:
            continue
        img = imgs[i].transpose(1, 2, 0)
        if i in dropped:
            img = img * 0.15
            ax.text(0.5, 0.5, "DROPPED", color="w", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, weight="bold")
        ax.imshow(img)
        ax.set_title(names[i], fontsize=10)
    return axes


def plot_bev_pair(probs, target, title_pred="LSS prediction", axes=None):
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].imshow(bev_rgb(target[0], target[1]), extent=EXTENT)
    axes[0].set_title("Ground truth")
    axes[1].imshow(bev_rgb(probs[0], probs[1]), extent=EXTENT)
    axes[1].set_title(title_pred)
    for ax in axes:
        _ego(ax)
    return axes


def plot_lifted_points(geom, alpha, names, thresh=0.04, ax=None):
    """Top-down scatter of every lifted (pixel, depth) point with alpha above
    `thresh`, coloured by camera and sized by alpha: the 'rays of guesses'
    from all cameras landing on one grid, just before they are summed."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    for n in range(geom.shape[0]):
        g = geom[n].reshape(-1, 3)
        a = alpha[n].reshape(-1)
        k = a > thresh
        ax.scatter(g[k, 1], g[k, 0], s=12 * a[k], color=CAM_COLORS[n % len(CAM_COLORS)],
                   alpha=0.5, lw=0, label=names[n])
    ax.set_xlim(world.YMAX, world.YMIN)
    ax.set_ylim(world.XMIN, world.XMAX)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=7, markerscale=3)
    ax.set_title("Lifted points, before splatting")
    _ego(ax)
    return ax


def plot_depth_ray(img, alpha_cam, depth_cam, cell, axes=None):
    """One feature pixel: where it is in the image, and its predicted depth
    distribution alpha(d) against the true depth (never used in training)."""
    r, c = cell
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw={"width_ratios": [1, 1.6]})
    axes[0].imshow(img.transpose(1, 2, 0))
    axes[0].add_patch(Rectangle((c * DOWN - 0.5, r * DOWN - 0.5), DOWN, DOWN,
                                fill=False, ec=ORANGE, lw=2))
    axes[0].axis("off")
    axes[0].set_title("pixel")
    a = alpha_cam[:, r, c]
    d = DEPTH_BINS.numpy()
    axes[1].bar(d, a, width=0.8, color=np.where(a > 0.5 * a.max(), ORANGE, "#5B8DB8"))
    true_d = cell_true_depth(depth_cam[None])[0, r, c]
    if np.isfinite(true_d):
        axes[1].axvline(true_d, color="k", ls="--", lw=1.5, label=f"true depth {true_d:.1f} m")
        axes[1].legend()
    axes[1].set_xlabel("depth d (m)")
    axes[1].set_ylabel("alpha(d)")
    axes[1].set_title("predicted depth distribution along the ray")
    return axes


def cell_true_depth(depth):
    """(N, H, W) pixel depth -> (N, h, w) true depth along the ray through each
    feature cell's centre (the ray the model lifts along). The centre sits
    between the middle 2x2 pixels; we take their harmonic mean (exact on flat
    ground, where 1/depth is linear in the image row). NaN on sky or where the
    four pixels disagree (object boundaries)."""
    m = DOWN // 2
    c = np.stack([depth[:, m - 1::DOWN, m - 1::DOWN], depth[:, m - 1::DOWN, m::DOWN],
                  depth[:, m::DOWN, m - 1::DOWN], depth[:, m::DOWN, m::DOWN]], -1)
    with np.errstate(invalid="ignore", divide="ignore"):
        hm = 4.0 / (1.0 / c).sum(-1)
        ok = np.isfinite(c).all(-1) & ((c.max(-1) - c.min(-1)) < 0.2 * hm)
    return np.where(ok, hm, np.nan)


def plot_depth_scatter(alpha, depth, ax=None):
    """alpha (S, N, D, h, w), depth (S, N, H, W): expected predicted depth vs truth."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    d = DEPTH_BINS.numpy()
    exp_d = (alpha * d[None, None, :, None, None]).sum(2)
    true_d = np.stack([cell_true_depth(x) for x in depth])
    k = np.isfinite(true_d) & (true_d >= d[0]) & (true_d <= d[-1])
    ax.hexbin(true_d[k], exp_d[k], gridsize=35, cmap="Blues", mincnt=1, bins="log")
    ax.plot([d[0], d[-1]], [d[0], d[-1]], color=ORANGE, lw=1.5)
    ax.set_xlabel("true depth (m)")
    ax.set_ylabel("predicted depth  E[d] (m)")
    mae = np.abs(exp_d[k] - true_d[k]).mean()
    const = np.abs(true_d[k] - np.median(true_d[k])).mean()
    ax.set_title(f"Depth was never labelled\nmean abs error {mae:.2f} m "
                 f"(always guessing one depth: {const:.2f} m)", fontsize=10)
    ax.set_aspect("equal")
    return ax, mae


def plot_plan(probs, paths, scores, best, cost, target=None, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(to_display(cost), extent=EXTENT, cmap="Greys", vmin=0, vmax=cost.max())
    if target is not None:   # outline the true cars
        ax.contour(to_display(target[0]), levels=[0.5], extent=EXTENT, origin="upper",
                   colors="#E45756", linewidths=1)
    for i, p in enumerate(paths):
        if i != best:
            ax.plot(p[:, 1], p[:, 0], color="#5B8DB8", lw=1, alpha=0.7)
    ax.plot(paths[best][:, 1], paths[best][:, 0], color=ORANGE, lw=3)
    ax.set_title("Shoot: cheapest path on the predicted cost map")
    _ego(ax)
    return ax


def plot_bars(labels, values, title, ylabel="vehicle IoU", ref=None, ax=None, highlight=()):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))
    colors = [ORANGE if i in highlight else "#5B8DB8" for i in range(len(labels))]
    bars = ax.bar(labels, values, color=colors)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}", ha="center", fontsize=9)
    if ref is not None:
        ax.axhline(ref, color="k", ls="--", lw=1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(values) * 1.18)
    ax.tick_params(axis="x", labelsize=8)
    return ax


def lifted_points_3d(geom, alpha, names, scene=None, thresh=0.04, max_points=40000):
    """Interactive 3D (plotly) view of the real lifted points: every (pixel,
    depth bin) from every camera, sized by alpha(d), with the true cars drawn
    as wireframe boxes. geom (N, D, h, w, 3), alpha (N, D, h, w)."""
    import plotly.graph_objects as go

    fig = go.Figure()
    kept = [(alpha[n] > thresh).sum() for n in range(geom.shape[0])]
    stride = max(1, int(np.ceil(sum(kept) / max_points)))
    for n in range(geom.shape[0]):
        g = geom[n].reshape(-1, 3)
        a = alpha[n].reshape(-1)
        k = np.flatnonzero(a > thresh)[::stride]
        inb = ((g[k, 0] > world.XMIN) & (g[k, 0] < world.XMAX)
               & (g[k, 1] > world.YMIN) & (g[k, 1] < world.YMAX))
        k = k[inb]
        fig.add_trace(go.Scatter3d(
            x=g[k, 0], y=g[k, 1], z=g[k, 2], mode="markers", name=names[n],
            marker=dict(size=2 + 25 * a[k], color=CAM_COLORS[n % len(CAM_COLORS)], opacity=0.55),
            hovertext=[f"{names[n]}  alpha={v:.2f}" for v in a[k]], hoverinfo="text"))

    def box_lines(cx, cy, yaw, L, W, H):
        c = np.array([[sx * L / 2, sy * W / 2, z] for z in (0, H)
                      for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        c = c @ Rz.T + [cx, cy, 0]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        xs, ys, zs = [], [], []
        for i, j in edges:
            xs += [c[i, 0], c[j, 0], None]
            ys += [c[i, 1], c[j, 1], None]
            zs += [c[i, 2], c[j, 2], None]
        return xs, ys, zs

    def add_boxes(boxes, name, color, width):
        xs, ys, zs = [], [], []
        for b in boxes:
            bx, by, bz = box_lines(*b)
            xs += bx; ys += by; zs += bz
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=name,
                                   line=dict(color=color, width=width)))

    if scene is not None:
        add_boxes([b[:6] for b in scene.boxes], "true cars", "black", 4)
        xr = np.linspace(world.XMIN, world.XMAX, 100)
        for side in (-1, 1):
            fig.add_trace(go.Scatter3d(
                x=xr, y=scene.road_centre(xr) + side * scene.road[3], z=np.zeros_like(xr),
                mode="lines", line=dict(color="gray", width=3), name="road edge",
                showlegend=side == 1))
    add_boxes([(0, 0, 0, 4.6, 2.0, 1.5)], "ego car", ORANGE, 6)

    fig.update_layout(
        title="Lift: every pixel's ray of depth guesses, from all cameras, in the car's frame"
              "<br><sup>dot size = alpha(d); drag to rotate. Splat = flatten these onto the ground grid and sum.</sup>",
        scene=dict(xaxis=dict(title="x forward (m)", range=[world.XMIN, world.XMAX]),
                   yaxis=dict(title="y left (m)", range=[world.YMIN, world.YMAX]),
                   zaxis=dict(title="z up (m)", range=[-6, 8]),
                   aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.3),
                   camera=dict(eye=dict(x=-1.1, y=0.9, z=0.8))),
        margin=dict(l=0, r=0, b=0, t=60), legend=dict(x=0.01, y=0.95))
    return fig
