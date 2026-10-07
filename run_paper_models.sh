#!/usr/bin/env bash
# run_paper_models.sh - the paper's own architectures on original + new datasets.
#   ResNet-56 (Sigmoid, no strides): single-image attacks   (paper Fig. 3)
#   ResNet-20: batched attacks, batch sizes 1/2/4/8          (paper Table 1)
# Writes to separate folders (*_resnet56, *_resnet20_batch), so LeNet/BERT results are never touched.
#
# Usage:  bash run_paper_models.sh smoke      # ~1-2 min check
#         bash run_paper_models.sh time       # how long one ResNet-56 iteration takes here
#         bash run_paper_models.sh            # everything
# Settings (optional, put in front of the command):  N=5 ITERS=1200 OUT=results bash run_paper_models.sh
set -e
export PYTHONUNBUFFERED=1           # log file updates live
N=${N:-5}                           # images per dataset for ResNet-56
ITERS=${ITERS:-1200}                # paper: 1200 iterations for images
OUT=${OUT:-results}                 # e.g. a Google Drive folder on Colab
SINGLE_DATASETS=${SINGLE_DATASETS:-"cifar100 mnist svhn lfw cifar10 fmnist"}
BATCH_DATASETS=${BATCH_DATASETS:-"cifar100 cifar10 fmnist"}

if [ "$1" = "smoke" ]; then
  python run_experiments.py --dataset cifar100 --model resnet56 --indices 30 --seeds 0 --iters 3 --out "$OUT" --tag smoke
  python run_experiments.py --dataset mnist    --model resnet56 --n-images 1 --seeds 0 --iters 3 --out "$OUT" --tag smoke
  python run_batch.py --dataset cifar100 --batch-sizes 1 2 --n-batches 1 --max-iters 3 --out "$OUT" --tag smoke
  rm -rf "$OUT"/*_smoke
  echo "== Paper-model smoke tests passed =="
  exit 0
fi

if [ "$1" = "time" ]; then
  python run_experiments.py --dataset cifar100 --model resnet56 --indices 30 --seeds 0 --iters 10 --out "$OUT" --tag timing \
    | grep -E "idx|device"
  rm -rf "$OUT"/*_timing
  echo "Seconds shown are for 10 iterations. x120 = one image at 1200 iterations."
  exit 0
fi

echo "== ResNet-56, single image: $N images x 1 seed x $ITERS iters per dataset =="
for d in $SINGLE_DATASETS; do
  python run_experiments.py --dataset $d --model resnet56 --n-images $N --seeds 0 --iters $ITERS --out "$OUT"
done

echo "== ResNet-20, batched: batch sizes 1 2 4 8, 2 batches each =="
for d in $BATCH_DATASETS; do
  python run_batch.py --dataset $d --model resnet20 --batch-sizes 1 2 4 8 --n-batches 2 --max-iters 3000 --out "$OUT"
done

python compare_datasets.py --results "$OUT"
