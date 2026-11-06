"""SHOOT: fire a fan of candidate trajectories into the top-down map and keep
the cheapest one.

Note: in the paper the cost map is *learned* end to end from expert driving
(the planner is trained to pick the template the human drove). Here we build
the cost map by hand from the predicted vehicle / drivable maps, which keeps
the demo self-contained while showing the same "score templates on a BEV
cost map" idea.
"""
import numpy as np
import torch
import torch.nn.functional as F

from . import world


def candidate_paths(n=25, length=20.0, max_curv=0.09, step=0.25):
    """Constant-curvature arcs leaving the ego car heading +x. Returns (n, S, 2) xy."""
    s = np.arange(0, length, step)
    paths = []
    for k in np.linspace(-max_curv, max_curv, n):
        if abs(k) < 1e-6:
            x, y = s, np.zeros_like(s)
        else:
            x, y = np.sin(k * s) / k, (1 - np.cos(k * s)) / k
        paths.append(np.stack([x, y], -1))
    return np.stack(paths)


def cost_map(probs, w_vehicle=10.0, w_offroad=3.0, margin_cells=3):
    """probs: (2, NX, NY) [vehicle, drivable] probabilities -> (NX, NY) cost."""
    probs = torch.as_tensor(probs, dtype=torch.float32)
    veh = F.max_pool2d(probs[None, :1], 2 * margin_cells + 1, 1, margin_cells)[0, 0]  # safety margin
    return (w_vehicle * veh + w_offroad * (1 - probs[1])).numpy()


def score_paths(paths, cost):
    ix = np.clip(((paths[..., 0] - world.XMIN) / world.RES).astype(int), 0, world.NX - 1)
    iy = np.clip(((paths[..., 1] - world.YMIN) / world.RES).astype(int), 0, world.NY - 1)
    return cost[ix, iy].sum(1)


def plan(probs, **kw):
    paths = candidate_paths(**kw)
    cost = cost_map(probs)
    scores = score_paths(paths, cost)
    return paths, scores, int(scores.argmin()), cost
