"""Build a deterministic COCO train set with one hard background crop per image.

Only images from the existing training split are used. Validation images are
never cropped into training data. Candidate 512px crops must not intersect a
ground-truth box (including a safety margin); the highest-texture valid crop is
kept as an annotation-free COCO image. Original images are hard-linked when
possible and copied as a fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", default=3407, type=int)
    parser.add_argument("--crop-size", default=512, type=int)
    parser.add_argument("--margin", default=48, type=int)
    parser.add_argument("--random-candidates", default=24, type=int)
    return parser.parse_args()


def intersects(a: tuple[int, int, int, int], b: tuple[float, float, float, float]) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def texture_score(image: Image.Image) -> float:
    array = np.asarray(image.resize((128, 128)).convert("L"), dtype=np.float32)
    grad_x = np.abs(np.diff(array, axis=1)).mean()
    grad_y = np.abs(np.diff(array, axis=0)).mean()
    return float(array.std() + 2.0 * (grad_x + grad_y))


def candidate_windows(width: int, height: int, size: int, rng: random.Random,
                      random_count: int) -> list[tuple[int, int, int, int]]:
    max_x, max_y = width - size, height - size
    if max_x < 0 or max_y < 0:
        return []
    xs = sorted({0, max_x, max_x // 2, max_x // 4, 3 * max_x // 4})
    ys = sorted({0, max_y, max_y // 2, max_y // 4, 3 * max_y // 4})
    windows = [(x, y, x + size, y + size) for y in ys for x in xs]
    windows.extend((x := rng.randint(0, max_x), y := rng.randint(0, max_y),
                    x + size, y + size) for _ in range(random_count))
    return list(dict.fromkeys(windows))


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    source_bytes = args.annotations.read_bytes()
    data = json.loads(source_bytes.decode("utf-8-sig"))
    output_images = args.output / "images"
    output_annotations = args.output / "annotations"
    output_images.mkdir(parents=True, exist_ok=True)
    output_annotations.mkdir(parents=True, exist_ok=True)

    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in data["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)

    # Repeated multibox records share file names. Build crops exactly once per
    # unique source image while retaining the original repeated records.
    unique_records: dict[str, dict] = {}
    boxes_by_file: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    for record in data["images"]:
        unique_records.setdefault(record["file_name"], record)
        for ann in annotations_by_image[record["id"]]:
            x, y, width, height = map(float, ann["bbox"])
            boxes_by_file[record["file_name"]].append((x, y, x + width, y + height))

    for name in unique_records:
        link_or_copy(args.images / name, output_images / name)

    output = {
        "info": dict(data.get("info", {}),
                     hard_negative_recipe="one deterministic non-overlap texture crop per unique train image"),
        "licenses": data.get("licenses", []),
        "images": [dict(record) for record in data["images"]],
        "annotations": [dict(annotation) for annotation in data["annotations"]],
        "categories": data["categories"],
    }
    next_image_id = max(record["id"] for record in output["images"]) + 1
    crop_records = []
    skipped = []
    for file_name in sorted(unique_records):
        source = args.images / file_name
        with Image.open(source) as opened:
            image = opened.convert("RGB")
        width, height = image.size
        crop_size = min(args.crop_size, width, height)
        rng = random.Random(f"{args.seed}:{file_name}")
        expanded = [(max(0.0, x1 - args.margin), max(0.0, y1 - args.margin),
                     min(float(width), x2 + args.margin), min(float(height), y2 + args.margin))
                    for x1, y1, x2, y2 in boxes_by_file[file_name]]
        valid = []
        for window in candidate_windows(width, height, crop_size, rng, args.random_candidates):
            if any(intersects(window, box) for box in expanded):
                continue
            crop = image.crop(window)
            valid.append((texture_score(crop), window, crop))
        if not valid:
            skipped.append(file_name)
            continue
        score, window, crop = max(valid, key=lambda item: (item[0], item[1]))
        negative_name = f"hn_{Path(file_name).stem}.jpg"
        crop.save(output_images / negative_name, format="JPEG", quality=95, subsampling=0)
        output["images"].append({
            "id": next_image_id,
            "file_name": negative_name,
            "width": crop.width,
            "height": crop.height,
        })
        crop_records.append({
            "image_id": next_image_id,
            "file_name": negative_name,
            "source_file": file_name,
            "crop_xyxy": list(window),
            "texture_score": score,
        })
        next_image_id += 1

    annotation_path = output_annotations / "instances_train_hardneg.json"
    annotation_path.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "source_annotations": str(args.annotations.resolve()),
        "source_annotations_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_unique_images": len(unique_records),
        "source_dataset_records": len(data["images"]),
        "source_annotations_count": len(data["annotations"]),
        "negative_crop_count": len(crop_records),
        "skipped_source_count": len(skipped),
        "crop_size": args.crop_size,
        "safety_margin": args.margin,
        "random_candidates": args.random_candidates,
        "combined_image_records": len(output["images"]),
        "combined_annotations": len(output["annotations"]),
        "annotation_file": str(annotation_path.resolve()),
        "skipped_sources": skipped,
        "crops": crop_records,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key not in {"crops", "skipped_sources"}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
