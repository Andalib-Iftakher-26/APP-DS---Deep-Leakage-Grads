# Experiment status and reproduction record

This file records the scope of the experiments reported for this project. It is intentionally explicit about partial and unsuccessful work. A script being present in the repository does not mean every possible configuration in that script was completed.

## Execution environment

The recorded execution environment was GitHub Codespaces, Linux x86-64, CPU-only:

- Python 3.14.2
- PyTorch 2.14.1+cpu
- Torchvision 0.29.1+cpu
- CUDA: unavailable
- CPU: 2 logical CPUs (as recorded during the work)

The pinned package list is in requirements.txt.

## Completed and partial experiment scope

| Experiment | Reported scope | Status / qualification |
|---|---|---|
| LeNet single-image experiments on public datasets | MNIST, CIFAR-10, CIFAR-100, SVHN, LFW, and Fashion-MNIST; 20 images per dataset, 3 dummy seeds, 300 outer iterations (60 trials per dataset; 360 total) | Completed and analysed |
| Smartphone natural-objects preprocessing | 285 source images; processed at 32×32 and 64×64 | Completed; raw images and generated outputs are not included in this public repository |
| Smartphone DLG | 12 selected images × 3 seeds at each resolution (36 trials at 32×32 and 36 at 64×64) | Completed and analysed |
| ResNet single-image experiments | 24 trials across six tested configurations (four trials per configuration) | Diagnostic / partial reproduction; no trial met the MSE < 0.01 image-reconstruction success threshold |
| Batched DLG | One trial each for batch sizes 1 and 2 | Partial; batch sizes 4 and 8 were not completed |
| BERT / masked-language-model experiment | 38 trials across built-in sentences and WikiText-2 data | Exploratory reimplementation; the tested BERT model was randomly initialised, not the pretrained model |
| LLM-as-Judge image comparison | 12 images × 3 seeds × 2 resolutions = 72 image pairs, assessed on four criteria | Assessment was performed; pair-level judge logs are not present in the available result archive, so the individual judgements cannot be independently audited from this repository |

## Smartphone dataset inventory

The local Smartphone Natural Objects collection contained 285 images. The reported class counts were:

| Class | Images |
|---|---:|
| bag | 24 |
| bottle | 30 |
| chair | 30 |
| container_bottle | 17 |
| fruit | 34 |
| key | 30 |
| mouse | 30 |
| mug | 30 |
| pen | 29 |
| watches | 31 |
| **Total** | **285** |

The collection was described as 143 images corresponding to paths already represented in the earlier metadata source and 142 additional images. The recorded setting counts were 168 indoor and 117 outdoor. The class counts are not perfectly balanced and some classes are below the original target of 30; this was reported as a scoped-down dataset rather than a completed 1,000-image collection.

Preprocessing generated 285 files at each of 32×32 and 64×64. The processed images were re-encoded without EXIF metadata. The raw source images were kept separately and are not published in this repository.

## Interpretation and reported findings

The report used image MSE < 0.01 as the operational threshold for a successful reconstruction. This is a project-defined evaluation threshold, not a universal criterion for DLG.

For the smartphone experiment, the reported 32×32 run achieved 27/36 trials below the threshold (75%), with 83.3% label accuracy, median MSE 3.74×10⁻⁵, mean MSE 0.0743, mean SSIM 0.7698, mean PSNR 37.01 dB, and mean runtime 14.41 s. The 64×64 run achieved 0/36 trials below the threshold, despite 100% label accuracy; mean MSE was 0.2510, median MSE 0.2467, mean SSIM 0.0638, mean PSNR 6.02 dB, and mean runtime 40.01 s.

The BERT experiment did not recover tokens successfully in the reported trials. It should be interpreted as an exploratory attempt, not evidence of a successful language-model reproduction. Likewise, the ResNet and batched experiments were incomplete and should not be presented as full reproductions of the paper's deeper-model and batch-size results.

## Evidence availability in this GitHub repository

At the time of this record:

- The experiment source scripts and launch helpers are committed.
- The results/ directory is excluded by .gitignore, and the historical CSV/JSON result archive is not present on the public branch.
- The smartphone raw images, processed image folders, and metadata CSV are not present on the public branch.
- The reported LLM-judge pair-level records are not available here.

Consequently, this file is a transparent summary of the reported work, not a substitute for the missing trial-level logs. For full independent verification, the non-sensitive result CSV/JSON files should be exported from the original Codespaces workspace and committed separately after checking them for private paths or image content.
