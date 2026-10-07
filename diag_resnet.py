"""
diag_resnet.py (round 2) - which ResNet variant actually leaks the image?

For CIFAR-100 index 30 and each variant: gradient size, starting loss, and the
image MSE after 20 / 50 / 100 iterations (lower = better; ~0.195 = image never
moved from the random start; < 0.01 = clear reconstruction).
Usage:  python diag_resnet.py
        python diag_resnet.py --models resnet20 --iters 200
"""
import argparse
import time

import torch
import torch.nn.functional as F

from dlg_lib import (load_dataset, get_image_tensor, build_model, run_dlg, label_to_onehot,
                     cross_entropy_for_onehot, mse)

NOTOL = dict(tolerance_grad=0.0, tolerance_change=0.0)
VARIANTS = [
    # name,                              bn,    head,  skip,       init
    ("A post-skip (v1), no tol",         False, "gap", "post",     "official"),
    ("B identity skip",                  False, "gap", "identity", "official"),
    ("C identity skip, PyTorch init",    False, "gap", "identity", "default"),
    ("D identity skip + BN",             True,  "gap", "identity", "official"),
    ("E identity skip + BN, PyT init",   True,  "gap", "identity", "default"),
]


def mse_at(snapshots, gt, it):
    best = None
    for i, img in snapshots:
        if i <= it:
            best = img
    return mse(gt, best) if best is not None and torch.isfinite(best).all() else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["resnet20", "resnet56"])
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--index", type=int, default=30)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dst, info = load_dataset("cifar100")
    gt, label = get_image_tensor(dst, args.index, info)
    marks = [m for m in (20, 50, 100, 200, 300) if m < args.iters] + [args.iters - 1]
    print(f"CIFAR-100 index {args.index}, up to {args.iters} iterations, L-BFGS tolerances off, device {device}\n")
    head = " ".join(f"{'@' + str(m + 1 if m == args.iters - 1 else m):>7}" for m in marks)
    print(f"{'model':<9} {'variant':<33} {'grad norm':>9} {'start loss':>10} {head} {'time':>6}")
    for arch in args.models:
        for name, bn, hd, skip, init in VARIANTS:
            net = build_model(info, 1234, device, arch, init, resnet_bn=bn, resnet_head=hd, resnet_skip=skip)
            x = gt.unsqueeze(0).to(device)
            y = label_to_onehot(torch.tensor([label], device=device), info["classes"])
            g = torch.autograd.grad(cross_entropy_for_onehot(net(x), y), net.parameters())
            gnorm = torch.sqrt(sum((t ** 2).sum() for t in g)).item()
            torch.manual_seed(0)
            dx = torch.randn_like(x)
            dy = torch.randn_like(y)
            dg = torch.autograd.grad(cross_entropy_for_onehot(net(dx), F.softmax(dy, -1)), net.parameters())
            start = sum(((a - b) ** 2).sum() for a, b in zip(dg, g)).item()
            t0 = time.time()
            res = run_dlg(net, gt, label, info["classes"], args.iters, 0, device=device, lbfgs_kwargs=NOTOL)
            vals = " ".join(f"{mse_at(res['snapshots'], gt, m):>7.4f}" for m in marks)
            print(f"{arch:<9} {name:<33} {gnorm:>9.2e} {start:>10.2e} {vals} {time.time() - t0:>5.0f}s",
                  flush=True)
        print()


if __name__ == "__main__":
    main()
