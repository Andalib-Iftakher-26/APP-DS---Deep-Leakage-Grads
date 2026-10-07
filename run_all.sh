#!/usr/bin/env bash
# run_all.sh - every experiment for the project update, in priority order.
# Usage:   bash run_all.sh smoke   # 2-minute check that every script runs (do this first)
#          bash run_all.sh         # the real runs; safe to stop and restart
# Each block writes its own results/<run>/ folder, so a crash later on doesn't lose earlier results.
set -e
export PYTHONUNBUFFERED=1   # log file updates live
N=20          # images per dataset (class-balanced)
SEEDS="0 1 2" # attacker starting points

if [ "$1" = "smoke" ]; then
  echo "== Smoke tests (tiny runs, numbers are meaningless) =="
  python run_experiments.py --dataset cifar100 --indices 30 --seeds -1 --iters 20 --tag smoke
  python run_experiments.py --dataset mnist  --n-images 1 --seeds 0 --iters 5 --tag smoke
  python run_experiments.py --dataset svhn   --n-images 1 --seeds 0 --iters 5 --tag smoke
  python run_experiments.py --dataset lfw    --n-images 1 --seeds 0 --iters 5 --tag smoke
  python run_experiments.py --dataset cifar10 --n-images 1 --seeds 0 --iters 5 --tag smoke
  python run_experiments.py --dataset fmnist --n-images 1 --seeds 0 --iters 5 --tag smoke
  python run_experiments.py --dataset cifar100 --model resnet20 --n-images 1 --seeds 0 --iters 2 --tag smoke
  python dlg_text.py --seeds 0 --iters 3 --tag smoke
  rm -rf results/*_smoke
  echo "== All smoke tests passed =="
  exit 0
fi

# --- 1. Sanity check against main.py (~1 min)
python run_experiments.py --dataset cifar100 --indices 25 30 40 60 --seeds -1 --tag official

# --- 2. Original datasets with the official LeNet (~7 min each on CPU)
python run_experiments.py --dataset cifar100 --n-images $N --seeds $SEEDS
python run_experiments.py --dataset mnist    --n-images $N --seeds $SEEDS
python run_experiments.py --dataset svhn     --n-images $N --seeds $SEEDS
python run_experiments.py --dataset lfw      --n-images $N --seeds $SEEDS

# --- 3. New existing datasets
python run_experiments.py --dataset cifar10 --n-images $N --seeds $SEEDS
python run_experiments.py --dataset fmnist  --n-images $N --seeds $SEEDS

# --- 4. Text: BERT masked language model (paper Section 4.2)
python dlg_text.py --seeds $SEEDS                     # 3 sentences, small BERT
python dlg_text.py --wikitext 10 --seeds 0            # new text data

# --- 5. The paper's deeper model (slow: minutes per image on CPU - few images only)
python run_experiments.py --dataset cifar100 --model resnet20 --n-images 5 --seeds 0
# python run_experiments.py --dataset cifar100 --model resnet56 --n-images 3 --seeds 0   # GPU recommended

python compare_datasets.py
