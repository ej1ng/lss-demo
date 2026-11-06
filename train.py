"""Train the mini LSS model on freshly rendered synthetic scenes.

    python train.py --steps 3000

No depth labels are ever used: the only supervision is the top-down
vehicle / drivable-area map.
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from lss import model as M
from lss import world


class SceneStream(IterableDataset):
    """Infinite stream of freshly rendered batches (so the model never sees
    the same scene twice and can't overfit)."""

    def __init__(self, batch, seed):
        self.batch, self.seed = batch, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, info.id if info else 0])
        while True:
            b = world.make_batch(self.batch, rng, jitter=True)
            yield {k: torch.from_numpy(b[k]) for k in ("imgs", "K", "R", "t", "target")}


def to_torch(b):
    return {k: torch.from_numpy(b[k]) for k in ("imgs", "K", "R", "t", "target")}


@torch.no_grad()
def evaluate(net, val, bs=16):
    net.eval()
    ious = []
    for i in range(0, len(val["imgs"]), bs):
        sl = {k: v[i:i + bs] for k, v in val.items()}
        ious.append(M.iou(net(sl["imgs"], sl["K"], sl["R"], sl["t"]), sl["target"]) * len(sl["imgs"]))
    net.train()
    return (sum(ious) / len(val["imgs"])).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="checkpoints/lss.pt")
    ap.add_argument("--uniform-depth", action="store_true",
                    help="ablation: no depth prediction, spread features evenly along each ray")
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(max(1, os.cpu_count() - args.workers))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    net = M.LSS(uniform_depth=args.uniform_depth)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.1)
    pos_weight = torch.tensor([2.0, 1.0]).view(1, 2, 1, 1)   # vehicles are rare: up-weight them

    val = to_torch(world.make_batch(64, np.random.default_rng(1234)))
    loader = DataLoader(SceneStream(args.batch, seed=0), batch_size=None,
                        num_workers=args.workers, persistent_workers=args.workers > 0)

    print(f"params: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M", flush=True)
    t0, best = time.time(), -1.0
    for step, b in enumerate(loader, 1):
        logits = net(b["imgs"], b["K"], b["R"], b["t"])
        loss = F.binary_cross_entropy_with_logits(logits, b["target"], pos_weight=pos_weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if step % 50 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}  {time.time() - t0:6.0f}s", flush=True)
        if step % 500 == 0 or step == args.steps:
            v_iou, d_iou = evaluate(net, val)
            print(f"  val IoU  vehicle {v_iou:.3f}  drivable {d_iou:.3f}", flush=True)
            if v_iou > best:
                best = v_iou
                torch.save({"model": net.state_dict(), "step": step,
                            "uniform_depth": args.uniform_depth,
                            "val_iou": {"vehicle": v_iou, "drivable": d_iou}}, args.out)
        if step >= args.steps:
            break
    print(f"done. best vehicle IoU {best:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
