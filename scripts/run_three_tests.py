#!/usr/bin/env python3
"""Run the three assessment-relevant image experiments with automatic resume.

Run from any directory:
    python scripts/run_three_tests.py

The same command is safe to run again after a Codespaces interruption. Each
completed trial is written to CSV and its files are committed and pushed to
origin/master. Existing completed trials are skipped on the next invocation.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
COMMON = [
    "--seeds", "0", "1", "2",
    "--iters", "300",
    "--model", "lenet",
    "--init", "official",
    "--resume",
    "--git-checkpoint-every", "1",
    "--git-branch", "master",
]

TESTS = [
    (
        "1/3: CIFAR-100 baseline",
        [
            "--dataset", "cifar100",
            "--n-images", "20",
            "--tag", "codespace_resume_cifar100",
        ],
    ),
    (
        "2/3: Smartphone Natural Objects, 32x32",
        [
            "--dataset", "smartphone",
            "--smartphone-root", "smartphone/processed/32",
            "--n-images", "12",
            "--tag", "codespace_resume_smartphone32",
        ],
    ),
    (
        "3/3: Smartphone Natural Objects, 64x64",
        [
            "--dataset", "smartphone",
            "--smartphone-root", "smartphone/processed/64",
            "--n-images", "12",
            "--tag", "codespace_resume_smartphone64",
        ],
    ),
]


def main() -> int:
    if not (ROOT / ".git").exists():
        print("ERROR: this script must be run from a Git checkout of the project.", file=sys.stderr)
        return 2

    # Fail early if the branch is wrong; the worker also checks before every push.
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=ROOT, text=True, capture_output=True,
    )
    if branch.returncode != 0 or branch.stdout.strip() != "master":
        print(
            "ERROR: switch your Codespace checkout to master before running. "
            f"Current branch: {branch.stdout.strip() or '(unknown)'}",
            file=sys.stderr,
        )
        return 2

    print("DLG resumable experiment sequence", flush=True)
    print("Configuration: LeNet, official initialisation, seeds 0/1/2, 300 iterations.", flush=True)
    print("Each completed trial is checkpointed to GitHub. Re-run this exact command after timeout.", flush=True)
    print("Do not change the tags or experiment settings while resuming.", flush=True)

    for title, test_args in TESTS:
        print("\n" + "=" * 76, flush=True)
        print(title, flush=True)
        print("=" * 76, flush=True)
        command = [
            PYTHON,
            str(ROOT / "run_experiments.py"),
            *test_args,
            *COMMON,
        ]
        try:
            subprocess.run(command, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            print(
                f"\nStopped during {title} (exit code {exc.returncode}). "
                "Completed trials have already been checkpointed. "
                "Fix the reported issue and rerun: python scripts/run_three_tests.py",
                file=sys.stderr,
                flush=True,
            )
            return exc.returncode or 1

    print("\nAll three tests are complete. CSV, run configuration and summary JSON files are in results/ and have been pushed to GitHub.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
