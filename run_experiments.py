"""
run_experiments.py - Run DLG over many images and random seeds, and save
metrics + figures for the reproduction study.

Examples
--------
# Replicate main.py exactly (seed -1 = official random stream), incl. default index 25
python run_experiments.py --dataset cifar100 --indices 25 30 40 60 --seeds -1 --tag official

# Main replication: 20 class-balanced CIFAR-100 images x 3 seeds
python run_experiments.py --dataset cifar100 --n-images 20 --seeds 0 1 2

# New existing datasets
python run_experiments.py --dataset cifar10 --n-images 20 --seeds 0 1 2
python run_experiments.py --dataset fmnist  --n-images 20 --seeds 0 1 2

# Your smartphone photos (after prepare_smartphone.py)
python run_experiments.py --dataset smartphone --smartphone-root smartphone/processed/32 \
       --n-images 10 --seeds 0 1 2

Outputs go to results/<dataset>/ :
  results.csv        one row per attack trial
  summary.json       aggregate metrics (also printed)
  showcase.png       originals vs best reconstruction, for slides
  loss_curves.png    gradient-matching loss over iterations
  trials/            per-trial original|reconstruction images
  progress/          reconstruction-over-time strips and GIFs (first seed)
"""

import argparse
import csv
import json
import math
import os
import statistics

import torch
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dlg_lib import (load_dataset, stratified_indices, get_image_tensor, class_names,
                     build_model, run_dlg, iters_to_converge, mse, psnr, ssim)

to_pil = transforms.ToPILImage()


def tensor_to_pil(t, scale=4):
    img = to_pil(t.clamp(0, 1))
    return img.resize((img.width * scale, img.height * scale), resample=0)  # nearest


def save_pair(orig, recon, path):
    a, b = tensor_to_pil(orig), tensor_to_pil(recon)
    from PIL import Image
    canvas = Image.new(a.mode, (a.width * 2 + 8, a.height), color=255 if a.mode == "L" else (255, 255, 255))
    canvas.paste(a, (0, 0))
    canvas.paste(b, (a.width + 8, 0))
    canvas.save(path)


def save_progress(snapshots, orig, path_png, path_gif, n_frames=8):
    # Strip: evenly spaced snapshots + the original at the end
    picks = [snapshots[round(i * (len(snapshots) - 1) / (n_frames - 1))] for i in range(n_frames)]
    fig, axes = plt.subplots(1, n_frames + 1, figsize=(1.6 * (n_frames + 1), 2))
    cmap = "gray" if orig.size(0) == 1 else None
    for ax, (it, img) in zip(axes, picks):
        ax.imshow(to_pil(img.clamp(0, 1)), cmap=cmap)
        ax.set_title(f"iter {it}", fontsize=8)
        ax.axis("off")
    axes[-1].imshow(to_pil(orig), cmap=cmap)
    axes[-1].set_title("original", fontsize=8)
    axes[-1].axis("off")
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close(fig)
    # GIF of every snapshot - handy for the video
    frames = [tensor_to_pil(img, scale=8).convert("RGB") for _, img in snapshots]
    frames[0].save(path_gif, save_all=True, append_images=frames[1:], duration=120, loop=0)


def summarise(rows, success_mse):
    ok = [r for r in rows if not r["diverged"] and math.isfinite(r["mse"])]

    def stat(key):
        vals = [r[key] for r in ok if r[key] is not None and math.isfinite(r[key])]
        if not vals:
            return None
        return dict(mean=statistics.mean(vals),
                    std=statistics.stdev(vals) if len(vals) > 1 else 0.0,
                    median=statistics.median(vals))

    by_image = {}
    for r in rows:
        by_image.setdefault(r["index"], []).append(r)
    conv = [r["iters_to_converge"] for r in ok if r["iters_to_converge"] is not None]
    return dict(
        n_trials=len(rows),
        n_images=len(by_image),
        n_diverged=sum(r["diverged"] for r in rows),
        mse=stat("mse"), psnr=stat("psnr"), ssim=stat("ssim"),
        runtime_s=stat("runtime_s"),
        label_accuracy=sum(r["label_correct"] for r in rows) / len(rows),
        success_threshold_mse=success_mse,
        success_rate_per_trial=sum(r["success"] for r in rows) / len(rows),
        success_rate_best_of_seeds=sum(any(r["success"] for r in rs) for rs in by_image.values()) / len(by_image),
        median_iters_to_converge=statistics.median(conv) if conv else None,
        n_converged=len(conv),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   choices=["cifar100", "cifar10", "mnist", "fmnist", "svhn", "smartphone"])
    p.add_argument("--n-images", type=int, default=20, help="class-balanced sample size")
    p.add_argument("--indices", type=int, nargs="*", help="explicit dataset indices (overrides --n-images)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="dummy-initialisation seeds; -1 = official main.py behaviour")
    p.add_argument("--iters", type=int, default=300, help="300 = official code default; paper used 1200")
    p.add_argument("--model-seed", type=int, default=1234, help="1234 = official code")
    p.add_argument("--image-size", type=int, default=None, help="resize inputs, e.g. 64 for resolution tests")
    p.add_argument("--smartphone-root", default=None)
    p.add_argument("--data-root", default="~/.torch")
    p.add_argument("--success-mse", type=float, default=0.01,
                   help="a trial counts as a successful attack if image MSE is below this")
    p.add_argument("--converge-threshold", type=float, default=1e-3,
                   help="gradient-matching loss below which the attack is 'converged'")
    p.add_argument("--out", default="results")
    p.add_argument("--tag", default="", help="optional suffix for the output folder")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dst, info = load_dataset(args.dataset, args.data_root, args.smartphone_root, args.image_size)
    names = class_names(dst)
    indices = args.indices if args.indices else stratified_indices(dst, args.n_images)

    out_dir = os.path.join(args.out, args.dataset + (f"_{args.tag}" if args.tag else ""))
    os.makedirs(os.path.join(out_dir, "trials"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "progress"), exist_ok=True)

    print(f"Dataset {args.dataset}: {len(dst)} images, {info['classes']} classes, "
          f"{info['channels']}x{info['size']}x{info['size']} | device {device}")
    print(f"{len(indices)} images x {len(args.seeds)} seeds = {len(indices) * len(args.seeds)} trials, "
          f"{args.iters} iterations each\n")

    net = build_model(info, args.model_seed, device)
    fields = ["dataset", "index", "true_label", "class_name", "seed", "pred_label", "label_correct",
              "mse", "psnr", "ssim", "final_grad_loss", "iters_to_converge", "runtime_s",
              "diverged", "success"]
    csv_path = os.path.join(out_dir, "results.csv")
    rows, curves, best = [], [], {}

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for n, idx in enumerate(indices, 1):
            gt, label = get_image_tensor(dst, idx, info)
            for seed in args.seeds:
                res = run_dlg(net, gt, label, info["classes"], args.iters, seed, device=device,
                              model_seed=args.model_seed)
                recon = res["recon"]
                finite = torch.isfinite(recon).all().item()
                m = mse(gt, recon) if finite else float("nan")
                row = dict(
                    dataset=args.dataset, index=idx, true_label=label,
                    class_name=names[label] if names else str(label), seed=seed,
                    pred_label=res["pred_label"], label_correct=int(res["pred_label"] == label),
                    mse=m, psnr=psnr(gt, recon) if finite else float("nan"),
                    ssim=ssim(gt, recon) if finite else float("nan"),
                    final_grad_loss=res["loss_history"][-1][1],
                    iters_to_converge=iters_to_converge(res["loss_history"], args.converge_threshold),
                    runtime_s=res["runtime_s"], diverged=int(res["diverged"]),
                    success=int(finite and m < args.success_mse),
                )
                writer.writerow(row)
                f.flush()
                rows.append(row)
                curves.append(res["loss_history"])

                if finite:
                    save_pair(gt, recon, os.path.join(out_dir, "trials", f"idx{idx}_seed{seed}.png"))
                    if idx not in best or m < best[idx][0]:
                        best[idx] = (m, gt, recon)
                if seed == args.seeds[0] and finite:
                    save_progress(res["snapshots"], gt,
                                  os.path.join(out_dir, "progress", f"idx{idx}.png"),
                                  os.path.join(out_dir, "progress", f"idx{idx}.gif"))
                print(f"[{n}/{len(indices)}] idx {idx:>5} ({row['class_name']}) seed {seed}: "
                      f"MSE {m:.4f}  SSIM {row['ssim']:.3f}  label {'OK' if row['label_correct'] else 'X'}  "
                      f"{res['runtime_s']:.1f}s")

    summary = summarise(rows, args.success_mse)
    summary.update(dataset=args.dataset, iters=args.iters, seeds=args.seeds,
                   image_shape=[info["channels"], info["size"], info["size"]], device=device)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Showcase: original row vs best-of-seeds reconstruction row (up to 10 images)
    show = list(best.items())[:10]
    if show:
        fig, axes = plt.subplots(2, len(show), figsize=(1.6 * len(show), 3.6), squeeze=False)
        cmap = "gray" if info["channels"] == 1 else None
        for j, (idx, (m, gt, recon)) in enumerate(show):
            axes[0, j].imshow(to_pil(gt), cmap=cmap)
            axes[0, j].set_title(f"idx {idx}", fontsize=8)
            axes[1, j].imshow(to_pil(recon.clamp(0, 1)), cmap=cmap)
            axes[1, j].set_title(f"MSE {m:.4f}", fontsize=8)
            axes[0, j].axis("off")
            axes[1, j].axis("off")
        axes[0, 0].text(-0.3, 0.5, "Original", transform=axes[0, 0].transAxes, rotation=90, va="center")
        axes[1, 0].text(-0.3, 0.5, "DLG", transform=axes[1, 0].transAxes, rotation=90, va="center")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "showcase.png"), dpi=150)
        plt.close(fig)

    # Loss curves
    fig, ax = plt.subplots(figsize=(6, 4))
    for c in curves:
        its = [it for it, l in c if math.isfinite(l) and l > 0]
        ls = [l for it, l in c if math.isfinite(l) and l > 0]
        ax.plot(its, ls, alpha=0.4, linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("gradient-matching loss")
    ax.set_title(f"DLG on {args.dataset} ({len(curves)} trials)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=150)
    plt.close(fig)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
