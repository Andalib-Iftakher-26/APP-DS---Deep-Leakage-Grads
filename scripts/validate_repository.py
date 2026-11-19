#!/usr/bin/env python3
"""Validate the tracked smartphone experiment inputs without third-party packages.

Checks both resolutions, class counts, paired image IDs, PNG dimensions, and
absence of the PNG eXIf metadata chunk. This is a dataset-integrity check,
not a model-training or experiment-results test.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COUNTS = {
    "bag": 24,
    "bottle": 30,
    "chair": 30,
    "container_bottle": 17,
    "fruit": 34,
    "key": 30,
    "mouse": 30,
    "mug": 30,
    "pen": 29,
    "watches": 31,
}
RESOLUTIONS = (32, 64)


def inspect_png(path: Path) -> tuple[int, int, bool]:
    """Return width, height and whether an eXIf chunk exists."""
    with path.open("rb") as stream:
        if stream.read(8) != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Not a PNG file: {path}")
        width = height = None
        has_exif = False
        found_iend = False

        while True:
            raw_length = stream.read(4)
            if len(raw_length) != 4:
                break
            length = struct.unpack(">I", raw_length)[0]
            chunk_type = stream.read(4)
            if len(chunk_type) != 4:
                break

            if chunk_type == b"IHDR":
                data = stream.read(length)
                if len(data) != length or length < 8:
                    raise ValueError(f"Invalid IHDR chunk: {path}")
                width, height = struct.unpack(">II", data[:8])
            else:
                if chunk_type == b"eXIf":
                    has_exif = True
                stream.seek(length, 1)

            crc = stream.read(4)
            if len(crc) != 4:
                break
            if chunk_type == b"IEND":
                found_iend = True
                break

    if width is None or height is None or not found_iend:
        raise ValueError(f"Incomplete PNG structure: {path}")
    return width, height, has_exif


def main() -> int:
    errors: list[str] = []
    total = 0
    ids_by_resolution: dict[int, dict[str, set[str]]] = {}

    # Do not publish the original phone images or their detailed metadata.
    for private_path in (ROOT / "raw", ROOT / "smartphone" / "raw",
                         ROOT / "smartphone" / "metadata.csv"):
        if private_path.exists():
            errors.append(f"Private source data should not be tracked: {private_path.relative_to(ROOT)}")

    for resolution in RESOLUTIONS:
        base = ROOT / "smartphone" / "processed" / str(resolution)
        ids_by_resolution[resolution] = {}
        for class_name, expected_count in EXPECTED_COUNTS.items():
            folder = base / class_name
            images = sorted(folder.glob("*.png")) if folder.is_dir() else []
            ids = {image.stem for image in images}
            ids_by_resolution[resolution][class_name] = ids

            if len(images) != expected_count:
                errors.append(
                    f"{resolution}px/{class_name}: expected {expected_count} PNGs, found {len(images)}"
                )
            if len(ids) != len(images):
                errors.append(f"{resolution}px/{class_name}: duplicate image stems detected")

            for image_path in images:
                try:
                    width, height, has_exif = inspect_png(image_path)
                    if (width, height) != (resolution, resolution):
                        errors.append(
                            f"{image_path.relative_to(ROOT)}: expected {resolution}x{resolution}, "
                            f"found {width}x{height}"
                        )
                    if has_exif:
                        errors.append(f"{image_path.relative_to(ROOT)}: PNG eXIf chunk found")
                except (OSError, ValueError, struct.error) as exc:
                    errors.append(f"{image_path.relative_to(ROOT)}: {exc}")
            total += len(images)

    for class_name in EXPECTED_COUNTS:
        ids_32 = ids_by_resolution[32][class_name]
        ids_64 = ids_by_resolution[64][class_name]
        if ids_32 != ids_64:
            errors.append(f"{class_name}: 32px and 64px image IDs do not match")

    if errors:
        print("Repository validation FAILED:")
        for error in errors:
            print(f" - {error}")
        return 1

    print("Repository validation PASSED")
    print(f" - {total} processed PNGs checked across both resolutions")
    print(f" - {sum(EXPECTED_COUNTS.values())} paired image IDs")
    print(" - class counts, image dimensions and PNG eXIf chunks checked")
    print(" - raw source folders and full metadata CSV are absent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
