"""
compare_datasets.py - Combine results/*/summary.json into one comparison table
(printed as Markdown, saved as CSV) and a bar chart, with the paper's reported
MSE alongside where one exists.

Usage:  python compare_datasets.py --results results
"""

import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Zhu et al. (2019), DLG MSE on [0,1]-normalised images (ResNet-56 variant, 1200 iters)
PAPER_MSE = {"mnist": 0.0038, "cifar100": 0.0069, "svhn": 0.0051}


def fmt(s, key="mean", digits=4):
    return "-" if not s else f"{s[key]:.{digits}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    args = p.parse_args()

    summaries = []
    for path in sorted(glob.glob(os.path.join(args.results, "*", "summary.json"))):
        with open(path) as f:
            s = json.load(f)
        s["run"] = os.path.basename(os.path.dirname(path))
        summaries.append(s)
    if not summaries:
        print("No summary.json files found.")
        return

    header = ["run", "trials", "paper MSE", "MSE mean", "MSE median", "SSIM", "PSNR (dB)",
              "label acc", "success/trial", "success best-of-seeds", "median iters to converge",
              "runtime/trial (s)"]
    table = []
    for s in summaries:
        table.append([
            s["run"], s["n_trials"],
            PAPER_MSE.get(s["dataset"], "-"),
            fmt(s["mse"]), fmt(s["mse"], "median"), fmt(s["ssim"], digits=3), fmt(s["psnr"], digits=1),
            f"{s['label_accuracy']:.0%}", f"{s['success_rate_per_trial']:.0%}",
            f"{s['success_rate_best_of_seeds']:.0%}",
            s["median_iters_to_converge"] if s["median_iters_to_converge"] is not None else "-",
            fmt(s["runtime_s"], digits=1),
        ])

    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in table:
        print("| " + " | ".join(str(c) for c in row) + " |")

    with open(os.path.join(args.results, "comparison.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(table)

    runs = [s["run"] for s in summaries]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].bar(runs, [s["mse"]["median"] if s["mse"] else 0 for s in summaries])
    axes[0].set_title("Median MSE (lower = better attack)")
    axes[1].bar(runs, [s["ssim"]["mean"] if s["ssim"] else 0 for s in summaries])
    axes[1].set_title("Mean SSIM (higher = better attack)")
    axes[1].set_ylim(0, 1)
    axes[2].bar(runs, [s["success_rate_per_trial"] for s in summaries])
    axes[2].set_title("Success rate per trial")
    axes[2].set_ylim(0, 1)
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(args.results, "comparison.png"), dpi=150)
    print(f"\nSaved {args.results}/comparison.csv and comparison.png")


if __name__ == "__main__":
    main()
