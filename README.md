# Deep Leakage from Gradients — Reproduction Study

This repository contains the code used for a coursework reproduction study of **Deep Leakage from Gradients (DLG)**, together with extensions and diagnostic experiments carried out on CPU-based GitHub Codespaces.

The project studies whether an attacker can reconstruct private training inputs by matching gradients shared during model training. It includes the original-style single-image demonstration, a repeatable experiment runner, ResNet variants, batched attacks, a text/MLM experiment, and preprocessing for a small smartphone natural-objects dataset.

> **Reproduction note:** this is a reproduction study with documented deviations, not a claim that every experiment in the original paper was reproduced exactly. The experiment status and limitations are recorded in [docs/EXPERIMENT_STATUS.md](docs/EXPERIMENT_STATUS.md).

## Repository layout

| Path | Purpose |
|---|---|
| **main.py** | Minimal single-image DLG demonstration based on the original project workflow |
| **dlg_lib.py** | Shared dataset loading, model construction, metrics, and single-image attack implementation |
| **run_experiments.py** | Repeatable single-image experiments across datasets, seeds, and architectures |
| **run_batch.py** | Batched gradient inversion experiments with reconstruction matching |
| **dlg_text.py** | Experimental masked-language-model gradient inversion implementation |
| **prepare_smartphone.py** | Converts local smartphone photos into metadata-stripped 32×32 / 64×64 images |
| **compare_datasets.py** | Aggregates generated JSON summaries and creates comparison tables/plots |
| **diag_resnet.py** | ResNet diagnostic experiments |
| **models/vision.py** | Vision model definitions used by the demonstration |
| **scripts/** | Shell entry points for smoke tests and longer experiment batches |
| **smartphone/README.md** | Local smartphone dataset folder and privacy instructions |
| **docs/EXPERIMENT_STATUS.md** | Experiment matrix, completed scope, deviations, and evidence limitations |

## Environment

The reported experiments were run in **GitHub Codespaces on CPU**. The recorded environment was:

- Python 3.14.2
- PyTorch 2.14.1+cpu
- Torchvision 0.29.1+cpu
- CUDA unavailable

The versions are pinned in requirements.txt. These versions reflect the recorded project environment; installing the full requirements file may take time and requires a compatible Python / wheel combination.

### Setup

~~~bash
git clone https://github.com/Andalib-Iftakher-26/APP-DS---Deep-Leakage-Grads.git
cd APP-DS---Deep-Leakage-Grads

python3.14 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

On Windows PowerShell, activate the environment with:

~~~powershell
.venv\Scripts\Activate.ps1
~~~

### Quick smoke test

The following small run checks the main image-experiment path. It downloads the selected public dataset on first use and writes outputs under results/.

~~~bash
python run_experiments.py \
  --dataset cifar10 \
  --indices 0 \
  --seeds 0 \
  --iters 2 \
  --tag smoke
~~~

This is only a pipeline check. Two iterations are **not** a meaningful reproduction result.

## Running experiments

### Single-image DLG

The general runner supports cifar100, cifar10, mnist, fmnist, svhn, lfw, and smartphone.

Example: class-balanced sample of 20 images, 3 dummy initialisation seeds, and 300 outer iterations:

~~~bash
python run_experiments.py \
  --dataset cifar100 \
  --n-images 20 \
  --seeds 0 1 2 \
  --iters 300
~~~

Replace cifar100 with another supported dataset to run the same configuration. Dataset-specific downloads/caches are stored under ~/.torch by default. Use --data-root PATH to change that location.

Useful options include:

- --model lenet|resnet20|resnet32|resnet56
- --init official|default
- --image-size N
- --success-mse 0.01
- --out PATH and --tag NAME

The runner prints per-trial progress and writes a CSV, JSON summary, reconstruction images, loss curves, and progress visualisations beneath results/<run-name>/.

### Smartphone natural-objects dataset

The original phone photos are **not included** in this repository. Place them locally using the folder structure and privacy guidance in [smartphone/README.md](smartphone/README.md), then preprocess:

~~~bash
python prepare_smartphone.py --root smartphone --sizes 32 64 --target-per-class 30
~~~

Run the 32×32 processed images:

~~~bash
python run_experiments.py \
  --dataset smartphone \
  --smartphone-root smartphone/processed/32 \
  --n-images 12 \
  --seeds 0 1 2 \
  --iters 300
~~~

For the 64×64 set, change the root to smartphone/processed/64. The exact sample selection is determined by the runner's deterministic class-stratified sampler; reproducing a particular historical subset requires recording and reusing its explicit indices.

### ResNet and batch experiments

For ResNet single-image tests, pass --model resnet20 or --model resnet56. The scripts in scripts/ provide longer CPU runs and timing checks. These can be substantially slower than LeNet, especially on CPU.

Batched experiments use run_batch.py:

~~~bash
python run_batch.py \
  --dataset cifar100 \
  --model resnet20 \
  --batch-sizes 1 2 \
  --n-batches 1 \
  --max-iters 300
~~~

### Text experiment

dlg_text.py implements an experimental masked-language-model variant. It is not a faithful reproduction of the paper's text experiment: the model initialisation, sample sentences, and optimisation settings are documented in the script and in the experiment-status file. The default run uses a small randomly initialised BERT configuration. A pretrained model can be requested with --pretrained, but that is a separate, more expensive configuration.

## Outputs and data handling

Generated experiment outputs are written to results/. This directory is ignored by Git so that large, machine-specific outputs are not accidentally committed. At the time of this repository cleanup, the public repository does **not** include the historical CSV/JSON result archive. The reported experiment scope and the evidence that is or is not available are described in [docs/EXPERIMENT_STATUS.md](docs/EXPERIMENT_STATUS.md).

The smartphone raw photos, processed images, and metadata CSV are also excluded from Git. Processed images are re-encoded without EXIF metadata, but the original local photographs remain unchanged. Review the actual images for faces, addresses, documents, screens, or other identifying content before sharing any sample publicly.

## Reproducibility and limitations

- The project ran on CPU; CUDA was unavailable.
- The main LeNet experiments used 300 outer iterations. Some paper configurations used a larger iteration budget, so runtime and convergence are not directly comparable.
- The deeper ResNet, batched, BERT, and LLM-judge components have incomplete reproduction coverage; see the status record rather than treating the helper scripts as proof that every configured run was completed.
- Some scripts are intended for new runs and diagnostics, not a turnkey regeneration of every number in the submitted report.
- Use the experiment-status record to distinguish completed experiments, partial attempts, and uncompleted configurations.

## Acknowledgements and citation

This work builds on the original DLG implementation by MIT HAN Lab. The upstream MIT licence is retained in LICENSE.

If you use the method, cite:

> Zhu, L., Liu, Z., and Han, S. (2019). *Deep Leakage from Gradients*. Advances in Neural Information Processing Systems. [arXiv:1906.08935](https://arxiv.org/abs/1906.08935).
