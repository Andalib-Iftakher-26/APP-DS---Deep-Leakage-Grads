"""
run_batch.py - Batched DLG (paper Table 1): leak a whole batch of B private
images from ONE averaged gradient, and record how many iterations the attack
needs to converge for each batch size. The paper used ResNet-20 for this.

The attacker optimises B dummy images + B dummy labels at once. Reconstructions
come back in arbitrary order, so each one is matched to an original with the
Hungarian algorithm (minimum total MSE) before scoring.

Examples
--------
python run_batch.py --dataset cifar100 --batch-sizes 1 2 4 8 --n-batches 2
python run_batch.py --dataset cifar10  --batch-sizes 1 2 4 8 --n-batches 2
python run_batch.py --dataset fmnist   --batch-sizes 1 2 4 8 --n-batches 2

Outputs go to results/<dataset>_<model>_batch/ :
  results.csv     one row per (batch size, batch, seed)
  summary.json    per-batch-size aggregates (also printed as a table)
  batch_bs<B>.png originals vs matched reconstructions, first batch of each size
"""

import argparse
import csv
import json
import math
import os
import statistics
import time

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dlg_lib import (ARCHS, load_dataset, stratified_indices, get_image_tensor, build_model,
                     label_to_onehot, cross_entropy_for_onehot, mse, psnr, ssim)

to_pil = transforms.ToPILImage()


def batch_attack(net, gt, labels, num_classes, max_iters, seed, threshold, stop_early, device):
    criterion = cross_entropy_for_onehot
    gt = gt.to(device)
    onehot = label_to_onehot(torch.tensor(labels, device=device).long(), num_classes=num_classes)

    # Victim: ONE gradient, averaged over the batch (as in federated SGD)
    loss = criterion(net(gt), onehot)
    orig = [g.detach().clone() for g in torch.autograd.grad(loss, net.parameters())]

    torch.manual_seed(seed)
    dummy_x = torch.randn(gt.size(), device=device).requires_grad_(True)
    dummy_y = torch.randn(onehot.size(), device=device).requires_grad_(True)
    opt = torch.optim.LBFGS([dummy_x, dummy_y])

    def closure():
        opt.zero_grad()
        dl = criterion(net(dummy_x), F.softmax(dummy_y, dim=-1))
        dg = torch.autograd.grad(dl, net.parameters(), create_graph=True)
        diff = sum(((a - b) ** 2).sum() for a, b in zip(dg, orig))
        diff.backward()
        return diff

    history, converged_at, diverged = [], None, False
    start = time.time()
    for it in range(max_iters):
        opt.step(closure)
        if it % 10 == 0 or it == max_iters - 1:
            cur = closure().item()
            history.append((it, cur))
            if not math.isfinite(cur):
                diverged = True
                break
            if converged_at is None and cur < threshold:
                converged_at = it
                if stop_early:
                    break
    return dict(recon=dummy_x.detach().cpu(), pred=dummy_y.detach().argmax(-1).cpu().tolist(),
                history=history, converged_at=converged_at, diverged=diverged,
                runtime_s=time.time() - start)


def match(gt, recon):
    """Hungarian matching of reconstructions to originals by MSE."""
    B = gt.size(0)
    cost = [[F.mse_loss(recon[j].clamp(0, 1), gt[i]).item() if torch.isfinite(recon[j]).all() else 1e9
             for j in range(B)] for i in range(B)]
    rows, cols = linear_sum_assignment(cost)
    return list(cols)  # cols[i] = reconstruction index matched to original i


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   choices=["cifar100", "cifar10", "mnist", "fmnist", "svhn", "lfw", "smartphone"])
    p.add_argument("--model", default="resnet20", choices=ARCHS, help="paper Table 1: resnet20")
    p.add_argument("--init", default="official", choices=["official", "default"])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--n-batches", type=int, default=2, help="batches per batch size")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--max-iters", type=int, default=3000)
    p.add_argument("--converge-threshold", type=float, default=1e-3)
    p.add_argument("--no-early-stop", action="store_true",
                   help="keep optimising after convergence (slower)")
    p.add_argument("--success-mse", type=float, default=0.01)
    p.add_argument("--model-seed", type=int, default=1234)
    p.add_argument("--smartphone-root", default=None)
    p.add_argument("--data-root", default="~/.torch")
    p.add_argument("--out", default="results")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dst, info = load_dataset(args.dataset, args.data_root, args.smartphone_root)
    net = build_model(info, args.model_seed, device, args.model, args.init)
    pool = stratified_indices(dst, args.n_batches * max(args.batch_sizes))

    run = f"{args.dataset}_{args.model}_batch" + (f"_{args.tag}" if args.tag else "")
    out_dir = os.path.join(args.out, run)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Batched DLG | {args.dataset} | {args.model} | device {device} | batch sizes {args.batch_sizes} "
          f"x {args.n_batches} batches x {len(args.seeds)} seeds | max {args.max_iters} iters\n")

    fields = ["dataset", "model", "batch_size", "batch_id", "indices", "seed", "converged",
              "iters_to_converge", "mse", "psnr", "ssim", "label_acc", "success_rate",
              "final_grad_loss", "runtime_s", "diverged"]
    rows = []
    cmap = "gray" if info["channels"] == 1 else None
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for bs in args.batch_sizes:
            for b in range(args.n_batches):
                idxs = pool[b * bs:(b + 1) * bs]
                items = [get_image_tensor(dst, i, info) for i in idxs]
                gt = torch.stack([x for x, _ in items])
                labels = [y for _, y in items]
                for seed in args.seeds:
                    r = batch_attack(net, gt, labels, info["classes"], args.max_iters, seed,
                                     args.converge_threshold, not args.no_early_stop, device)
                    perm = match(gt, r["recon"])
                    finite = bool(torch.isfinite(r["recon"]).all())
                    m = [mse(gt[i], r["recon"][perm[i]]) if finite else float("nan") for i in range(bs)]
                    row = dict(
                        dataset=args.dataset, model=args.model, batch_size=bs, batch_id=b,
                        indices=" ".join(map(str, idxs)), seed=seed,
                        converged=int(r["converged_at"] is not None),
                        iters_to_converge=r["converged_at"],
                        mse=statistics.mean(m) if finite else float("nan"),
                        psnr=statistics.mean(psnr(gt[i], r["recon"][perm[i]]) for i in range(bs)) if finite else float("nan"),
                        ssim=statistics.mean(ssim(gt[i], r["recon"][perm[i]]) for i in range(bs)) if finite else float("nan"),
                        label_acc=sum(r["pred"][perm[i]] == labels[i] for i in range(bs)) / bs,
                        success_rate=sum(x < args.success_mse for x in m) / bs if finite else 0.0,
                        final_grad_loss=r["history"][-1][1], runtime_s=r["runtime_s"],
                        diverged=int(r["diverged"]),
                    )
                    w.writerow(row)
                    f.flush()
                    rows.append(row)
                    print(f"bs {bs} batch {b} seed {seed}: converged at "
                          f"{r['converged_at'] if r['converged_at'] is not None else '-'}  "
                          f"MSE {row['mse']:.4f}  SSIM {row['ssim']:.3f}  label acc {row['label_acc']:.0%}  "
                          f"{r['runtime_s']:.0f}s")

                    if b == 0 and seed == args.seeds[0] and finite:
                        fig, axes = plt.subplots(2, bs, figsize=(1.6 * bs + 0.6, 3.6), squeeze=False)
                        for i in range(bs):
                            axes[0, i].imshow(to_pil(gt[i]), cmap=cmap)
                            axes[1, i].imshow(to_pil(r["recon"][perm[i]].clamp(0, 1)), cmap=cmap)
                            axes[1, i].set_title(f"MSE {m[i]:.4f}", fontsize=8)
                            axes[0, i].axis("off")
                            axes[1, i].axis("off")
                        axes[0, 0].text(-0.3, 0.5, "Original", transform=axes[0, 0].transAxes,
                                        rotation=90, va="center")
                        axes[1, 0].text(-0.3, 0.5, "DLG", transform=axes[1, 0].transAxes,
                                        rotation=90, va="center")
                        fig.suptitle(f"{args.dataset}, {args.model}, batch size {bs}", fontsize=9)
                        plt.tight_layout()
                        plt.savefig(os.path.join(out_dir, f"batch_bs{bs}.png"), dpi=150)
                        plt.close(fig)

    per_bs = {}
    for bs in args.batch_sizes:
        rs = [r for r in rows if r["batch_size"] == bs]
        conv = [r["iters_to_converge"] for r in rs if r["iters_to_converge"] is not None]
        fin = [r for r in rs if math.isfinite(r["mse"])]
        per_bs[str(bs)] = dict(
            trials=len(rs), converged=len(conv),
            median_iters_to_converge=statistics.median(conv) if conv else None,
            mse_mean=statistics.mean(r["mse"] for r in fin) if fin else None,
            ssim_mean=statistics.mean(r["ssim"] for r in fin) if fin else None,
            label_acc=statistics.mean(r["label_acc"] for r in rs),
            success_rate=statistics.mean(r["success_rate"] for r in rs),
            runtime_s=statistics.mean(r["runtime_s"] for r in rs),
        )
    summary = dict(type="batch", dataset=args.dataset, model=args.model, init=args.init,
                   max_iters=args.max_iters, converge_threshold=args.converge_threshold,
                   seeds=args.seeds, n_batches=args.n_batches, device=device, per_batch_size=per_bs)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n| batch size | converged | median iters to converge | MSE | SSIM | label acc | success |")
    print("|---|---|---|---|---|---|---|")
    for bs, s in per_bs.items():
        print(f"| {bs} | {s['converged']}/{s['trials']} | {s['median_iters_to_converge'] or '-'} | "
              f"{s['mse_mean'] if s['mse_mean'] is None else round(s['mse_mean'], 4)} | "
              f"{s['ssim_mean'] if s['ssim_mean'] is None else round(s['ssim_mean'], 3)} | "
              f"{s['label_acc']:.0%} | {s['success_rate']:.0%} |")
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
