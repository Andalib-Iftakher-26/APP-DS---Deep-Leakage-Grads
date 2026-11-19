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
| Smartphone natural-objects preprocessing | 285 source images; 285 processed images at 32×32 and 285 at 64×64 | Completed; processed PNG inputs are now included in the current repository; raw originals and full metadata CSV are excluded from the current snapshot |
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

Preprocessing generated 285 files at each of 32×32 and 64×64. The processed images were re-encoded without EXIF metadata. The 570 processed PNGs have been restored to the current repository under smartphone/processed/32/ and smartphone/processed/64/. The raw source images and full metadata CSV are not included in the current snapshot.

## Interpretation and reported findings

The report used image MSE < 0.01 as the operational threshold for a successful reconstruction. This is a project-defined evaluation threshold, not a universal criterion for DLG.

For the smartphone experiment, the reported 32×32 run achieved 27/36 trials below the threshold (75%), with 83.3% label accuracy, median MSE 3.74×10⁻⁵, mean MSE 0.0743, mean SSIM 0.7698, mean PSNR 37.01 dB, and mean runtime 14.41 s. The 64×64 run achieved 0/36 trials below the threshold, despite 100% label accuracy; mean MSE was 0.2510, median MSE 0.2467, mean SSIM 0.0638, mean PSNR 6.02 dB, and mean runtime 40.01 s.

The BERT experiment did not recover tokens successfully in the reported trials. It should be interpreted as an exploratory attempt, not evidence of a successful language-model reproduction. Likewise, the ResNet and batched experiments were incomplete and should not be presented as full reproductions of the paper's deeper-model and batch-size results.

## Evidence availability in this GitHub repository

At the time of this record:

- The experiment source scripts and launch helpers are committed.
- The results/ directory is excluded by .gitignore, and the historical CSV/JSON result archive is not present on the public branch.
- The current public branch includes 570 processed smartphone PNGs (285 per resolution); the original raw photos and full metadata CSV are not present in the current branch snapshot.
- The reported LLM-judge pair-level records are not available here.

Consequently, this file is a transparent summary of the reported work, not a substitute for the missing trial-level logs. For full independent verification, the non-sensitive result CSV/JSON files should be exported from the original Codespaces workspace and committed separately after checking them for private paths or image content.

## Historical dataset privacy warning

Although the current branch tree does not contain the source image folders or smartphone metadata CSV, earlier public commits do contain the original photographs under the repository-level raw/ directory, as well as smartphone/metadata.csv and the processed image files. The processed PNGs have since been restored to the latest branch as the reproducible input dataset. The historical metadata CSV contains 285 records, and its had_gps field is set to 1 for all 285 records. This records GPS metadata presence in the originals; it does not, by itself, disclose coordinate values in the CSV. The original image files may nevertheless retain embedded EXIF data.

The current .gitignore excludes the root raw/ folder, smartphone/raw/, the full metadata CSV, and generated status/contact-sheet files. The processed smartphone PNG folders are intentionally tracked in the current branch so the experiments can be rerun. Ignore rules do **not** erase earlier commits or copies/clones of the repository. Removing those historical files from the public repository requires rewriting Git history (for example, with git-filter-repo), force-updating the affected branches, and asking collaborators to re-clone or carefully clean existing clones. Review this as a separate, deliberate privacy operation before doing it; rotating any credentials or tokens accidentally exposed in history would also be necessary.
