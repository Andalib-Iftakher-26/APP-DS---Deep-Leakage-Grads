#!/usr/bin/env bash
# run_resnet_cpu.sh - the paper's ResNet-20 / ResNet-56 on CPU (Codespace), small subset.
# Re-checks the Colab GPU diagnostics. Results go to results/resnet_20/ and results/resnet_56/.
#
#   bash run_resnet_cpu.sh time    # ~2 min: seconds per iteration on this machine
#   bash run_resnet_cpu.sh         # everything (~7 h on a 2-core CPU - too long for one Codespace session)
#   bash run_resnet_cpu.sh r20     # ResNet-20 single-image + batched only (~3 h)
#   bash run_resnet_cpu.sh r56     # ResNet-56 only (~4 h) - e.g. on Colab CPU at the same time
# Optional settings:  ITERS=300 N20=4 N56=4 OUT=results bash run_resnet_cpu.sh r20
#
# Variants (see diag_resnet.py):
#   A = paper layout: Sigmoid wraps the residual sum, no BatchNorm, official uniform(-0.5,0.5) init
#   C = identity skip (Sigmoid inside the branch only), PyTorch default init  [best in diagnostics]
# All runs: no strides (as in the paper), L-BFGS early-stop tolerances off, seed 0.
set -e
export PYTHONUNBUFFERED=1
ITERS=${ITERS:-300}     # paper: 1200 (too slow on CPU; stated as a deviation)
N20=${N20:-4}           # images per dataset, ResNet-20
N56=${N56:-4}           # images per dataset, ResNet-56
OUT=${OUT:-results}     # results go to $OUT/resnet_20 and $OUT/resnet_56
PART=${1:-all}
DATASETS=${DATASETS:-"cifar100 cifar10"}
A="--no-lbfgs-tol"
C="--no-lbfgs-tol --resnet-skip identity --init default"

if [ "$1" = "time" ]; then
  for m in resnet20 resnet56; do
    python run_experiments.py --dataset cifar100 --model $m --indices 30 --seeds 0 --iters 5 $A \
      --out results/_timing --tag t | grep -E "idx"
  done
  rm -rf results/_timing
  echo "Seconds shown are for 5 iterations. x60 = one image at 300 iterations."
  exit 0
fi

if [ "$PART" = "all" ] || [ "$PART" = "r20" ]; then
echo "== ResNet-20, variants A and C, $N20 images x $ITERS iters =="
for d in $DATASETS; do
  python run_experiments.py --dataset $d --model resnet20 --n-images $N20 --seeds 0 --iters $ITERS $A --out "$OUT/resnet_20" --tag A_post
  python run_experiments.py --dataset $d --model resnet20 --n-images $N20 --seeds 0 --iters $ITERS $C --out "$OUT/resnet_20" --tag C_identity
done

fi

if [ "$PART" = "all" ] || [ "$PART" = "r56" ]; then
echo "== ResNet-56, variant A, $N56 image(s) x $ITERS iters =="
for d in $DATASETS; do
  python run_experiments.py --dataset $d --model resnet56 --n-images $N56 --seeds 0 --iters $ITERS $A --out "$OUT/resnet_56" --tag A_post
done

fi

if [ "$PART" = "all" ] || [ "$PART" = "r20" ]; then
echo "== ResNet-20 batched (paper Table 1), batch sizes 1 and 2 =="
python run_batch.py --dataset cifar100 --model resnet20 --batch-sizes 1 2 --n-batches 1 --max-iters $ITERS \
  --no-lbfgs-tol --out "$OUT/resnet_20" --tag A_post

fi

echo "== Summary =="
[ -d "$OUT/resnet_20" ] && python compare_datasets.py --results "$OUT/resnet_20"
[ -d "$OUT/resnet_56" ] && python compare_datasets.py --results "$OUT/resnet_56"
echo "== Done =="
