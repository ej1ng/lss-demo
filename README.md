# lss-demo: Lift, Splat, Shoot from scratch

A small, readable re-implementation of **Lift, Splat, Shoot** (Philion & Fidler, ECCV 2020).
It turns six camera images into one top-down map of vehicles and road, then picks a path on that map.

It runs on a laptop CPU. Instead of nuScenes, it uses a synthetic world (a curved road and box-shaped cars) rendered by a tiny ray caster.
Because the world is synthetic, the true depth of every pixel is known. That lets the demo check something real data can't show directly: the model learns depth even though it never sees a depth label.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy matplotlib jupyter

python train.py                                                   # ~25 min on CPU -> checkpoints/lss.pt
python train.py --uniform-depth --out checkpoints/lss_uniform.pt  # optional no-depth baseline
python demo.py                                                    # figures/ + figures/metrics.json
jupyter notebook demo.ipynb                                       # live walkthrough
```

## Where each idea lives

| Paper idea | Code |
|---|---|
| Camera model `x = R (d K^-1 [u,v,1]) + t` | [lss/world.py](lss/world.py) (rendering), `frustum_points` in [lss/model.py](lss/model.py) |
| **Lift**: per-pixel depth distribution alpha(d), outer product alpha (x) c | `CamEncoder`, `LSS.lift` in [lss/model.py](lss/model.py) |
| **Splat**: sum-pool every lifted point into its top-down cell | `splat` (index_add) and `splat_cumsum` (the paper's cumsum trick) |
| Top-down segmentation (vehicle and drivable area) | `BEVEncoder`, trained with BCE in [train.py](train.py) |
| **Shoot**: score candidate trajectories on a cost map | [lss/plan.py](lss/plan.py) |
| Robustness: camera dropout, calibration error, a different car | [demo.py](demo.py) |

## Figures (after `python demo.py`)

1. `01_cameras`: the six input views
2. `02_bev_prediction`: predicted vs true top-down map
3. `03_lifted_points`: every camera's "rays of guesses" on one grid, before they are summed
4. `04_depth_ray`: alpha(d) for one pixel, against the true depth
5. `05_depth_learned`: predicted vs true depth for every pixel (depth was never supervised)
6. `06_shoot_plan`: candidate paths, with the cheapest one in orange
7. `07_drop_camera`: IoU when each camera is removed
8. `08_calibration`: IoU under yaw calibration error
9. `09_other_car_*`: a 5-camera rig with 80° lenses that the model never trained on

## Simplifications vs the paper (and why)

- **Data**: synthetic scenes instead of nuScenes or Lyft, so it trains in minutes on a CPU and the true depth is available for checking.
- **Backbone**: a small CNN instead of EfficientNet. To make up for its small receptive field, the network gets one extra input channel: each pixel's vertical ray slope `(v - cy) / fy`. The channel depends on the camera intrinsics, so it still means the same thing for a different lens.
- **Shoot**: the paper learns the cost map end to end from expert trajectories. Here the cost map is built by hand from the predicted vehicle and drivable-area maps. The planning step itself is the same: fire template paths into the cost map, sum the cost along each one, and take the cheapest.
- **Splat speed**: on CPU, PyTorch's `index_add_` is already a fused scatter-add, so the cumsum trick isn't faster here. The trick matters on GPU, and it was a big improvement over the earlier per-cell pooling approaches. Both versions are included, and they're checked to give the same result.
