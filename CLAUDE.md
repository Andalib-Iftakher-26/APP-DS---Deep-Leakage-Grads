# Project context for Claude

## Who / what
Andalib Iftakher, Master of Data Science, Macquarie University. Unit COMP8240 (Applications of Data Science),
Sem 2 2026. Novel reproduction project of **Deep Leakage from Gradients (DLG)**, Zhu, Liu & Han, NeurIPS 2019
(official code: github.com/mit-han-lab/dlg, which this repo is based on).

## Current deliverable: Project Update video, due Fri 9 Oct 2026, 11:55 pm
5 min (±10%), face visible in a corner. Novel-project rubric (40 marks):
- Recap / justification — 6
- Replication (environment, does the code work, similar results?, what did/didn't work) — 8
- New data description (existing new datasets + constructed dataset) — 8
- New data progress (collection, preprocessing, annotation, results so far) — 8  ← biggest risk
- Presentation quality — 10
Goal: full marks, i.e. "leaving no questions" in every row.

## What the proposal promised
- Reproduce DLG image attack; MSE as primary metric vs paper (MNIST 0.0038, CIFAR-100 0.0069, SVHN 0.0051, LFW 0.0055)
- Extra metrics: SSIM, PSNR, label-recovery acc, success rate, iterations to convergence, runtime; repeated random inits
- Preliminary LLM-as-judge (multimodal LLM, fixed rubric, 1–5 recognisability/similarity, repeated + reversed order)
- New existing datasets: CIFAR-10 (primary), Fashion-MNIST, STL-10 (resolution, if time), WikiText-2 (optional, BERT)
- Constructed: Smartphone Natural Objects — 10 classes (bottle, cup, book, key, shoe, backpack, plant, fruit,
  chair, computer mouse), iPhone 17, varied conditions, target 1,000 / minimum 300 (30/class), metadata per photo,
  32×32 (+64×64), privacy exclusions, GPS stripped; DLG on stratified subset (10/class)
- Defence experiment only if time allows

## What the paper actually used
- Images (MNIST, CIFAR-100, SVHN, LFW): **ResNet-56, ReLU→Sigmoid, strides removed**, random init, L-BFGS, 1200 iters
- Batched experiment (Table 1, batch sizes 1/2/4/8): **ResNet-20**
- Text: **BERT, randomly initialised**, masked LM, 15% [MASK], 3 sentences from the NeurIPS website, 100 iters
- Official repo instead ships a Sigmoid **LeNet** demo (300 iters). Its ResNet code is broken (calls `F.Sigmoid`,
  which doesn't exist) and has no ResNet-56. Paper's MSE figure doesn't say whether ResNet-56 or -20 was used.
- The student wants ResNet-56 and ResNet-20 used as in the paper, on both the original and the new datasets.

## Code (all in repo root)
- `dlg_lib.py` — models (LeNet; `ResNetSigmoid` = resnet20/32/56, no BatchNorm, no strides by default),
  loaders (cifar100, cifar10 via Hugging Face fallback, mnist, fmnist, svhn [test split], lfw [≥20 faces/person,
  62 people, via scikit-learn], smartphone ImageFolder), metrics, single-image attack `run_dlg`.
  Seed -1 = reproduce `main.py`'s exact random stream.
- `run_experiments.py` — single-image attacks over N images × seeds → `results/<dataset>[_<model>][_<tag>]/`
- `run_batch.py` — batched DLG (Table 1), Hungarian matching of reconstructions. **Written, not yet run.**
- `dlg_text.py` — BERT attack (not in official repo; reimplemented). Default small random BERT for CPU;
  `--size base` = paper-faithful. Word-embedding matrix frozen. Don't use `--pretrained` for replication.
- `prepare_smartphone.py` — photos in `smartphone/raw/<class>/<setting>_<lighting>_<background>/` →
  stripped, cropped, 32/64 px PNGs in `smartphone/processed/`, `metadata.csv`, `status.txt`, contact sheets
- `compare_datasets.py` — combines all `summary.json` into tables + `comparison.csv/png`
- `run_all.sh` — `smoke` = quick test of everything; no argument = full LeNet + BERT + ResNet-20 subset run
- Original student files: `main.py` (saves index_*_actual/reconstructed.png), `exe.py` (timing),
  `gradient_chart.py` (proposal Fig. 2)

## Environment
- GitHub Codespace, CPU only (2 cores), Python 3.14, venv **`DS`** (`source DS/bin/activate`)
- PyTorch MUST be the CPU build: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
  (plain `pip install torch` pulls ~3 GB of CUDA libs). `requirements.txt` has the extra index line.
- `.gitignore` excludes `DS/`, `smartphone/raw/`, `results/`
- Datasets cached in `~/.torch`
- ResNet-20 ≈ 2 s per L-BFGS iteration on CPU; ResNet-56 far slower → GPU (e.g. Colab) recommended

## Status (Wed 7 Oct, 13:20)
- Smoke test passed. Full `run_all.sh` running in background (`nohup … > run_log.txt`). **Don't edit the files it
  uses until it finishes** (`pgrep -f run_all.sh`).
- Preliminary finding: index 25 (repo's default) fails completely (MSE 0.37) while 30/40/60 are near-perfect
  (MSE ~1e-5). The full run checks this across seeds.
- Preliminary output PNGs were accidentally deleted in a commit — restore from git history if still missing.

## To do (priority order)
1. Smartphone photos (≥300) → prepare → DLG on 10/class × 3 seeds
2. ResNet-56 single-image runs on original + new datasets; ResNet-20 batched runs (Colab GPU if possible)
3. LLM-as-judge: blinded pairs + fixed prompt, run manually in claude.ai (Pro plan has no API), 2 runs with
   reversed order, agreement script
4. STL-10 loader (32 and 64 px)
5. Optional: defence (Gaussian noise on gradients)
6. Slides + timed 5-min script; record Thursday

## Deviations to state in the video
LeNet vs ResNet-56; 300 vs 1200 iters; LFW preprocessing unspecified; SVHN test split; BERT reimplemented,
smaller model for CPU; success = MSE < 0.01; seeds vary only the attacker's start, model init fixed at 1234.

## Working style
Student wants step-by-step guidance, one step at a time, with a clear "done when" check per step. AI use is
allowed ("Open" policy) but the student must understand and explain all code and results.
