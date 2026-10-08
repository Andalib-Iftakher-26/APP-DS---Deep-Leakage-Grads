"""
prepare_smartphone.py - Build the Smartphone Natural Objects dataset.

Input layout (put your phone photos here):
    smartphone/raw/<class>/<photo>.jpg
or, to get capture conditions annotated automatically, one subfolder per
shooting session named  <setting>_<lighting>_<background>:
    smartphone/raw/bottle/indoor_bright_plain/IMG_0001.jpg
    smartphone/raw/bottle/outdoor_dim_cluttered/IMG_0042.jpg

What it does, for every photo:
  1. Applies the EXIF rotation, then discards ALL metadata (GPS, device,
     timestamps) - output files are re-encoded from pixels only.
  2. Centre-crops to a square and resizes to 32x32 (and 64x64) RGB PNGs.
  3. Gives each image a stable ID (bottle_001, ...) - IDs never change when
     you add more photos later.
  4. Writes/updates metadata.csv. Columns you fill in by hand (distance,
     orientation, notes, include) are preserved on re-runs.
  5. Makes contact sheets and a per-class count report for your slides.

Usage:
    python prepare_smartphone.py --root smartphone
    python prepare_smartphone.py --root smartphone --sizes 32 64 --target-per-class 30

iPhone photos are HEIC by default. Either set Settings > Camera > Formats >
"Most Compatible" (JPEG), or:  pip install pillow-heif
"""

import argparse
import csv
import os
from collections import Counter, OrderedDict

from PIL import Image, ImageDraw, ImageOps

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_OK = True
except ImportError:
    HEIF_OK = False

CLASSES = ["bottle", "cup", "book", "key", "shoe", "backpack", "plant", "fruit", "chair", "mouse"]
EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
AUTO_COLS = ["image_id", "class", "source_file", "orig_width", "orig_height", "had_gps",
             "setting", "lighting", "background"]
MANUAL_COLS = ["distance", "orientation", "notes", "include"]
GPS_IFD = 0x8825


def find_photos(raw_dir):
    photos = []
    for cls in sorted(os.listdir(raw_dir)):
        cdir = os.path.join(raw_dir, cls)
        if not os.path.isdir(cdir):
            continue
        for dirpath, _, files in os.walk(cdir):
            for fn in sorted(files):
                if os.path.splitext(fn)[1].lower() in EXTS:
                    full = os.path.join(dirpath, fn)
                    session = os.path.relpath(dirpath, cdir)
                    photos.append((cls, os.path.relpath(full, raw_dir), "" if session == "." else session))
    return photos


def parse_session(session):
    parts = session.replace("\\", "/").split("/")[0].split("_") if session else []
    parts += [""] * (3 - len(parts))
    return parts[0], parts[1], parts[2]


def load_metadata(path):
    if not os.path.exists(path):
        return OrderedDict()
    with open(path, newline="") as f:
        return OrderedDict((r["source_file"], r) for r in csv.DictReader(f))


def has_gps(img):
    try:
        return bool(img.getexif().get_ifd(GPS_IFD))
    except Exception:
        return False


def centre_square(img):
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def contact_sheet(paths_labels, out_path, thumb=64, cols=10):
    if not paths_labels:
        return
    rows = (len(paths_labels) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (thumb + 4), rows * (thumb + 16)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (p, label) in enumerate(paths_labels):
        x, y = (i % cols) * (thumb + 4), (i // cols) * (thumb + 16)
        sheet.paste(Image.open(p).convert("RGB").resize((thumb, thumb), Image.NEAREST), (x, y))
        draw.text((x + 2, y + thumb + 2), label[:12], fill="black")
    sheet.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="smartphone")
    ap.add_argument("--sizes", type=int, nargs="+", default=[32, 64])
    ap.add_argument("--target-per-class", type=int, default=30)
    ap.add_argument("--name", default="Smartphone Natural Objects", help="dataset name for the report")
    args = ap.parse_args()

    raw_dir = os.path.join(args.root, "raw")
    meta_path = os.path.join(args.root, "metadata.csv")
    if not os.path.isdir(raw_dir):
        os.makedirs(raw_dir)
        for c in CLASSES:
            os.makedirs(os.path.join(raw_dir, c), exist_ok=True)
        print(f"Created {raw_dir}/<class>/ folders - copy your photos in and run again.")
        return

    meta = load_metadata(meta_path)
    used_ids = Counter()
    for r in meta.values():
        cls, num = r["image_id"].rsplit("_", 1)
        used_ids[cls] = max(used_ids[cls], int(num))

    photos = find_photos(raw_dir)
    unknown = sorted({c for c, _, _ in photos} - set(CLASSES))
    if unknown:
        print(f"Note: folders not in the planned class list: {unknown} (processed anyway)")

    n_new, n_fail, skipped_heic = 0, 0, 0
    for cls, rel, session in photos:
        src = os.path.join(raw_dir, rel)
        if rel.lower().endswith((".heic", ".heif")) and not HEIF_OK:
            skipped_heic += 1
            continue
        try:
            with Image.open(src) as im:
                gps = has_gps(im)
                w, h = im.size
                im = ImageOps.exif_transpose(im)               # keep correct orientation
                clean = Image.new("RGB", im.size)              # fresh image: no metadata at all
                clean.paste(im.convert("RGB"))
        except Exception as e:
            print(f"  ! could not read {rel}: {e}")
            n_fail += 1
            continue

        if rel in meta:
            row = meta[rel]
        else:
            used_ids[cls] += 1
            setting, lighting, background = parse_session(session)
            row = dict(image_id=f"{cls}_{used_ids[cls]:03d}", **{"class": cls}, source_file=rel,
                       orig_width=w, orig_height=h, had_gps=int(gps),
                       setting=setting, lighting=lighting, background=background,
                       distance="", orientation="", notes="", include="yes")
            meta[rel] = row
            n_new += 1

        square = centre_square(clean)
        for s in args.sizes:
            out_dir = os.path.join(args.root, "processed", str(s), cls)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, row["image_id"] + ".png")
            if row.get("include", "yes").strip().lower() in ("no", "n", "0", "false"):
                if os.path.exists(out_path):
                    os.remove(out_path)                         # excluded on inspection
                continue
            square.resize((s, s), Image.LANCZOS).save(out_path)  # PNG written with no EXIF

    with open(meta_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AUTO_COLS + MANUAL_COLS, lineterminator="\n")
        w.writeheader()
        for r in sorted(meta.values(), key=lambda r: r["image_id"]):
            w.writerow({k: r.get(k, "") for k in AUTO_COLS + MANUAL_COLS})

    # Verify the processed files really carry no metadata
    leaked = []
    for dirpath, _, files in os.walk(os.path.join(args.root, "processed")):
        for fn in files:
            if fn.endswith(".png"):
                with Image.open(os.path.join(dirpath, fn)) as im:
                    if len(im.getexif()) or im.info.get("exif"):
                        leaked.append(fn)

    # Report
    included = [r for r in meta.values() if r.get("include", "yes").strip().lower() not in ("no", "n", "0", "false")]
    counts = Counter(r["class"] for r in included)
    report = [f"{args.name} dataset - status", ""]
    report.append(f"{'class':<10} {'images':>6} / target {args.target_per_class}")
    present = sorted(c for c in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, c)))
    for c in sorted(set(present or CLASSES) | set(counts)):
        bar = "#" * min(counts[c], 40)
        report.append(f"{c:<10} {counts[c]:>6}   {bar}")
    report.append("")
    report.append(f"Total included: {len(included)} (excluded on inspection: {len(meta) - len(included)})")
    report.append(f"Originals that contained GPS: {sum(int(r['had_gps']) for r in meta.values())} "
                  f"- all stripped. Processed files with metadata: {len(leaked)}")
    for col in ("setting", "lighting", "background"):
        c = Counter(r[col] or "(not set)" for r in included)
        report.append(f"{col:<11}: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
    if skipped_heic:
        report.append(f"\nWARNING: skipped {skipped_heic} HEIC photos - run: pip install pillow-heif")
    report_text = "\n".join(report)
    print(report_text)
    with open(os.path.join(args.root, "status.txt"), "w") as f:
        f.write(report_text + "\n")

    # Contact sheets (32x32 versions, shown enlarged) for the video
    s = min(args.sizes)
    sheet_items = []
    for r in sorted(included, key=lambda r: r["image_id"]):
        p = os.path.join(args.root, "processed", str(s), r["class"], r["image_id"] + ".png")
        if os.path.exists(p):
            sheet_items.append((p, r["image_id"]))
    contact_sheet(sheet_items, os.path.join(args.root, f"contact_sheet_{s}px.png"))
    # One example per class, original vs processed, for the preprocessing slide
    examples = OrderedDict()
    for r in sorted(included, key=lambda r: r["image_id"]):
        examples.setdefault(r["class"], r)
    if examples:
        th = 128
        fig = Image.new("RGB", (len(examples) * (th + 6), 2 * th + 40), "white")
        d = ImageDraw.Draw(fig)
        for i, (cls, r) in enumerate(examples.items()):
            with Image.open(os.path.join(raw_dir, r["source_file"])) as im:
                im = centre_square(ImageOps.exif_transpose(im).convert("RGB"))
                fig.paste(im.resize((th, th), Image.LANCZOS), (i * (th + 6), 0))
            small = Image.open(os.path.join(args.root, "processed", str(s), cls, r["image_id"] + ".png"))
            fig.paste(small.resize((th, th), Image.NEAREST), (i * (th + 6), th + 4))
            d.text((i * (th + 6) + 2, 2 * th + 10), cls, fill="black")
        fig.save(os.path.join(args.root, "preprocessing_examples.png"))
    print(f"\nWrote {meta_path}, status.txt, contact sheet and preprocessing_examples.png in {args.root}/")


if __name__ == "__main__":
    main()
