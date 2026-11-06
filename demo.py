"""Run every experiment and save the figures + numbers for the presentation.

    python demo.py                      # uses checkpoints/lss.pt
    python demo.py --baseline checkpoints/lss_uniform.pt

Outputs go to figures/ (PNGs + metrics.json).
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lss import model as M
from lss import plan as P
from lss import viz as V
from lss import world

TEST_SEED = 2024


# ---- helpers usable from the notebook too ----------------------------------
def load_model(path):
    ck = torch.load(path, map_location="cpu")
    net = M.LSS(uniform_depth=ck.get("uniform_depth", False))
    net.load_state_dict(ck["model"])
    return net.eval()


def tensors(b):
    return {k: torch.from_numpy(b[k]) for k in ("imgs", "K", "R", "t", "target")}


@torch.no_grad()
def predict(net, b, cam_mask=None, R=None, bs=16, aux=False):
    """Run the model over a numpy batch. Optionally drop cameras (cam_mask) or
    feed it a different (e.g. mis-calibrated) R than the one used to render."""
    T = tensors(b)
    R = T["R"] if R is None else torch.as_tensor(R, dtype=torch.float32)
    logits, alphas, geoms = [], [], []
    for i in range(0, len(T["imgs"]), bs):
        sl = slice(i, i + bs)
        m = None if cam_mask is None else cam_mask[sl]
        out = net(T["imgs"][sl], T["K"][sl], R[sl], T["t"][sl], cam_mask=m, return_aux=aux)
        if aux:
            out, a = out
            alphas.append(a["alpha"])
            geoms.append(a["geom"])
        logits.append(out)
    logits = torch.cat(logits)
    if aux:
        return logits, torch.cat(alphas).numpy(), torch.cat(geoms).numpy()
    return logits


def ious(logits, b):
    v, d = M.iou(logits, torch.from_numpy(b["target"])).tolist()
    return {"vehicle": round(v, 4), "drivable": round(d, 4)}


def rotate_yaw(R, sigma_deg, rng):
    """Mis-calibrate: rotate each camera's R by a random yaw error."""
    R = R.copy()
    for s in range(R.shape[0]):
        for n in range(R.shape[1]):
            a = np.radians(rng.normal(0, sigma_deg))
            Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
            R[s, n] = Rz @ R[s, n]
    return R


def save(fig, name):
    os.makedirs("figures", exist_ok=True)
    fig.savefig(f"figures/{name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  wrote figures/%s.png" % name)


# ---- the experiments --------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/lss.pt")
    ap.add_argument("--baseline", default="checkpoints/lss_uniform.pt")
    ap.add_argument("--n-test", type=int, default=128)
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count())

    net = load_model(args.ckpt)
    rig = world.DEFAULT_RIG
    print("rendering test scenes ...")
    test = world.make_batch(args.n_test, np.random.default_rng(TEST_SEED))
    metrics = {}

    # 1. main result ---------------------------------------------------------
    logits, alpha, geom = predict(net, test, aux=True)
    probs = logits.sigmoid().numpy()
    metrics["lss"] = ious(logits, test)
    print("LSS IoU:", metrics["lss"])
    if os.path.exists(args.baseline):
        base = load_model(args.baseline)
        b_logits = predict(base, test)
        metrics["no_depth_baseline"] = ious(b_logits, test)
        print("no-depth baseline IoU:", metrics["no_depth_baseline"])
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        V.plot_bev_pair(probs[0], test["target"][0], axes=axes[[0, 2]])
        axes[1].imshow(V.bev_rgb(*b_logits.sigmoid().numpy()[0]), extent=V.EXTENT)
        axes[1].set_title("No-depth baseline (alpha = 1/D)")
        V._ego(axes[1])
        save(fig, "00_lss_vs_baseline_bev")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        V.plot_bars(["no-depth\nbaseline", "LSS\n(learned depth)"],
                    [metrics["no_depth_baseline"]["vehicle"], metrics["lss"]["vehicle"]],
                    "So, did it actually work?", ax=ax, highlight=(1,))
        save(fig, "00_lss_vs_baseline")

    # 2. figures for one example scene ---------------------------------------
    s = 0
    fig, axes = plt.subplots(2, 3, figsize=(10, 3.8))
    V.plot_cameras(test["imgs"][s], rig.names, axes=axes)
    fig.suptitle("Input: six camera images")
    save(fig, "01_cameras")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    V.plot_bev_pair(probs[s], test["target"][s], axes=axes)
    save(fig, "02_bev_prediction")

    fig, ax = plt.subplots(figsize=(6, 6))
    V.plot_lifted_points(geom[s], alpha[s], rig.names, ax=ax)
    save(fig, "03_lifted_points")
    V.lifted_points_3d(geom[s], alpha[s], rig.names, scene=test["scenes"][s]).write_html(
        "figures/03b_lifted_points_3d.html", include_plotlyjs="cdn")
    print("  wrote figures/03b_lifted_points_3d.html")

    # depth along one ray: pick a feature pixel on a car in the front camera
    tdepth = V.cell_true_depth(test["depth"][s])
    cam = 0
    img = test["imgs"][s, cam]
    h, w = tdepth.shape[1:]
    best_cell, best_score = (h // 2 + 1, w // 2), -1
    for r in range(h // 2 - 1, h):
        for c in range(w):
            d = tdepth[cam, r, c]
            if not np.isfinite(d):
                continue
            # prefer pixels on cars: colour differs from road/grass, mid-range depth
            patch = img[:, r * 4:(r + 1) * 4, c * 4:(c + 1) * 4].reshape(3, -1).mean(1)
            nonground = min(np.abs(patch - world.ROAD).sum(), np.abs(patch - world.GRASS).sum())
            score = nonground * (8 < d < 30)
            if score > best_score:
                best_cell, best_score = (r, c), score
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw={"width_ratios": [1, 1.6]})
    V.plot_depth_ray(img, alpha[s, cam], test["depth"][s, cam], best_cell, axes=axes)
    save(fig, "04_depth_ray")

    fig, ax = plt.subplots(figsize=(5, 5))
    _, mae = V.plot_depth_scatter(alpha, test["depth"], ax=ax)
    metrics["depth_mae_m"] = round(float(mae), 3)
    save(fig, "05_depth_learned")

    paths, scores, best, cost = P.plan(probs[s])
    fig, ax = plt.subplots(figsize=(6, 6))
    V.plot_plan(probs[s], paths, scores, best, cost, target=test["target"][s], ax=ax)
    save(fig, "06_shoot_plan")

    # 3. robustness: drop each camera ----------------------------------------
    drop = {}
    for i, name in enumerate(rig.names):
        mask = torch.ones(args.n_test, rig.n)
        mask[:, i] = 0
        drop[name] = ious(predict(net, test, cam_mask=mask), test)["vehicle"]
    metrics["drop_one_camera"] = drop
    fig, ax = plt.subplots(figsize=(8, 3.5))
    V.plot_bars(["all 6"] + [f"no {n}" for n in rig.names],
                [metrics["lss"]["vehicle"]] + list(drop.values()),
                "A camera goes down", ref=metrics["lss"]["vehicle"], ax=ax, highlight=(0,))
    save(fig, "07_drop_camera")

    mask = torch.ones(args.n_test, rig.n)
    mask[:, 0] = 0
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    V.plot_bev_pair(predict(net, test, cam_mask=mask).sigmoid().numpy()[s], test["target"][s],
                    title_pred="prediction with FRONT camera dropped", axes=axes)
    save(fig, "07b_drop_front_bev")

    # 4. robustness: calibration error ---------------------------------------
    calib = {}
    for sigma in [0, 1, 2, 4, 8]:
        Rn = rotate_yaw(test["R"], sigma, np.random.default_rng(sigma))
        calib[f"{sigma} deg"] = ious(predict(net, test, R=Rn), test)["vehicle"]
    metrics["calibration_yaw_noise"] = calib
    fig, ax = plt.subplots(figsize=(6, 3.5))
    V.plot_bars(list(calib), list(calib.values()), "Calibration off (random yaw error per camera)",
                ax=ax, highlight=(0,))
    save(fig, "08_calibration")

    # 5. a different car: new camera count, angles and lens, no retraining ----
    other = world.make_batch(args.n_test, np.random.default_rng(TEST_SEED), rig=world.OTHER_RIG)
    o_logits = predict(net, other)
    metrics["different_car"] = ious(o_logits, other)
    fig, axes = plt.subplots(2, 3, figsize=(10, 3.8))
    V.plot_cameras(other["imgs"][s], world.OTHER_RIG.names, axes=axes)
    fig.suptitle("A different car: 5 cameras, 80 deg lenses, new angles")
    save(fig, "09_other_car_cameras")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    V.plot_bev_pair(o_logits.sigmoid().numpy()[s], other["target"][s],
                    title_pred="prediction (never trained on this rig)", axes=axes)
    save(fig, "09b_other_car_bev")

    # 6. splat speed: index_add vs the paper's cumsum trick -------------------
    with torch.no_grad():
        T = tensors({k: v[:8] for k, v in test.items() if k != "scenes"})
        a, lifted = net.lift(T["imgs"], T["K"])
        g = M.frustum_points(T["K"], T["R"], T["t"], *a.shape[-2:])
        timing = {}
        for f in (M.splat, M.splat_cumsum):
            f(g, lifted)
            t0 = time.perf_counter()
            for _ in range(10):
                out = f(g, lifted)
            timing[f.__name__] = round((time.perf_counter() - t0) / 10 * 1000, 2)
        timing["max_abs_diff"] = float((M.splat(g, lifted) - M.splat_cumsum(g, lifted)).abs().max())
        timing["points_per_batch"] = int(np.prod(g.shape[:-1]))
    metrics["splat_ms_batch8"] = timing

    with open("figures/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
