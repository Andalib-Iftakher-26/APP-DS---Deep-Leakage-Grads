# Smartphone Natural Objects dataset (local files)

This folder documents the locally collected smartphone dataset used in the reproduction study. The original photographs and full metadata CSV are deliberately **not committed in the current branch snapshot**. The processed images used by the experiments are included.

## Expected local layout

Put one subfolder per class under raw/:

~~~text
smartphone/
├── raw/
│   ├── bag/
│   ├── bottle/
│   ├── chair/
│   ├── container_bottle/
│   ├── fruit/
│   ├── key/
│   ├── mouse/
│   ├── mug/
│   ├── pen/
│   └── watches/
└── ...
~~~

Optional session subfolders can describe capture conditions. The expected naming convention is setting_lighting_background, for example:

~~~text
smartphone/raw/bottle/indoor_bright_plain/photo.jpg
smartphone/raw/bottle/outdoor_dim_cluttered/photo.jpg
~~~

Use only the labels that are genuinely supported by the capture notes; the script reads the first nested session folder name.

## Preprocessing

From the repository root:

~~~bash
python prepare_smartphone.py --root smartphone --sizes 32 64 --target-per-class 30
~~~

The script creates:

- processed/32/<class>/ and processed/64/<class>/: square-cropped RGB PNGs. The 285 files at each resolution used in the reported experiment are included in this repository.
- metadata.csv: a local file containing image IDs, source-relative paths, dimensions, whether the source had GPS metadata, capture-condition labels, and manual annotation fields. This full file is not committed.
- status.txt and contact-sheet / preprocessing-example images. These generated local files are not committed.

The processed PNGs are re-encoded from pixel data after EXIF orientation is applied. They are checked for EXIF metadata. The raw source files are not modified by preprocessing.

## Reported inventory

The collected version used in the project contained 285 images across the ten classes. The recorded counts and dataset limitations are in [docs/EXPERIMENT_STATUS.md](../docs/EXPERIMENT_STATUS.md). The initial target was reduced from 1,000 images to a smaller minimum viable collection because of the project time limit.

## Privacy and publishing

Do not commit raw/ or the full metadata.csv. The processed/ folders are tracked because they are the actual experimental inputs; check pixel content before adding any new samples. The metadata file includes source-relative filenames/paths and a GPS-presence flag; raw photographs may contain identifiable people, home interiors, documents, reflections, or location clues even after EXIF is removed. Check the actual image pixels and filenames before publishing a curated sample.

**Historical warning:** previous public commits of this repository contain the original source photos and a metadata CSV. The historical CSV marks all 285 original photos as having GPS metadata. Although these files are not in the current branch snapshot and are ignored for future commits, deleting them from the latest version does not remove them from Git history. See [docs/EXPERIMENT_STATUS.md](../docs/EXPERIMENT_STATUS.md) before deciding whether to rewrite repository history.

To run the smartphone attack after preprocessing:

~~~bash
python run_experiments.py \
  --dataset smartphone \
  --smartphone-root smartphone/processed/32 \
  --n-images 12 \
  --seeds 0 1 2 \
  --iters 300
~~~

Change the path to smartphone/processed/64 for 64×64 inputs. The deterministic class-stratified sampler selects the sample; to recreate a specific historic sample exactly, record and pass the original --indices.
