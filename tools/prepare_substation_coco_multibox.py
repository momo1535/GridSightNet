"""Create reproducible COCO splits while preserving real multi-object labels.

Near-identical YOLO boxes are deduplicated. Spatially distinct boxes remain
separate COCO annotations. Explicit scene groups are kept in the same split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Record:
    image_path: Path
    width: int
    height: int
    boxes_xyxy: tuple[tuple[float, float, float, float], ...]
    source_box_count: int
    deduplicated_box_count: int

    @property
    def is_multi(self) -> bool:
        return len(self.boxes_xyxy) > 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scene-groups", type=Path)
    parser.add_argument("--val-ratio", default=0.2, type=float)
    parser.add_argument("--seed", default=3407, type=int)
    parser.add_argument("--duplicate-iou", default=0.9, type=float)
    parser.add_argument("--min-val-multi-images", default=4, type=int)
    parser.add_argument(
        "--train-multibox-repeat",
        default=5,
        type=int,
        help="Total occurrences of each true multi-box training image per epoch",
    )
    return parser.parse_args()


def sha256_files(paths):
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def box_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_iou(first, second):
    intersection = max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )
    union = box_area(first) + box_area(second) - intersection
    return intersection / union if union > 0 else 0.0


def read_yolo_boxes(label_path, width, height, duplicate_iou):
    source_boxes = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected five values")
        class_id, cx, cy, box_width, box_height = map(float, parts)
        if class_id != 0:
            raise ValueError(f"{label_path}:{line_number}: expected class 0")
        if not all(math.isfinite(value) for value in (cx, cy, box_width, box_height)):
            raise ValueError(f"{label_path}:{line_number}: non-finite value")
        if box_width <= 0 or box_height <= 0:
            raise ValueError(f"{label_path}:{line_number}: non-positive size")
        box = (
            max(0.0, (cx - box_width / 2) * width),
            max(0.0, (cy - box_height / 2) * height),
            min(float(width), (cx + box_width / 2) * width),
            min(float(height), (cy + box_height / 2) * height),
        )
        if box_area(box) <= 0:
            raise ValueError(f"{label_path}:{line_number}: box is outside the image")
        source_boxes.append(box)
    if not source_boxes:
        raise ValueError(f"{label_path}: no annotations")

    # Deterministic class-agnostic NMS: keep the larger annotation when two
    # source rows describe the same physical target.
    unique_boxes = []
    for candidate in sorted(source_boxes, key=lambda box: (-box_area(box), box)):
        if all(box_iou(candidate, kept) < duplicate_iou for kept in unique_boxes):
            unique_boxes.append(candidate)
    unique_boxes.sort()
    return tuple(unique_boxes), len(source_boxes)


def load_scene_groups(path, stems):
    group_for_stem = {stem: stem for stem in stems}
    if path is None:
        return group_for_stem
    data = json.loads(path.read_text(encoding="utf-8"))
    seen = set()
    for index, members in enumerate(data.get("groups", []), 1):
        group_id = f"explicit-group-{index:03d}"
        for stem in members:
            if stem not in stems:
                raise ValueError(f"Unknown scene-group member: {stem}")
            if stem in seen:
                raise ValueError(f"Scene-group member repeated: {stem}")
            seen.add(stem)
            group_for_stem[stem] = group_id
    return group_for_stem


def split_records(records, group_for_stem, val_ratio, seed, min_val_multi_images):
    grouped = {}
    for record in records:
        grouped.setdefault(group_for_stem[record.image_path.stem], []).append(record)
    multi_groups = [group for group in grouped.values() if any(record.is_multi for record in group)]
    single_groups = [group for group in grouped.values() if not any(record.is_multi for record in group)]
    rng = random.Random(seed)
    rng.shuffle(multi_groups)
    rng.shuffle(single_groups)

    total_multi_images = sum(record.is_multi for record in records)
    target_multi = min(total_multi_images, max(round(total_multi_images * val_ratio), min_val_multi_images))
    val_groups = []
    selected_multi = 0
    while multi_groups and selected_multi < target_multi:
        group = multi_groups.pop()
        val_groups.append(group)
        selected_multi += sum(record.is_multi for record in group)

    target_val_images = round(len(records) * val_ratio)
    val_count = sum(len(group) for group in val_groups)
    while single_groups and val_count < target_val_images:
        group = single_groups.pop()
        if val_count + len(group) <= target_val_images:
            val_groups.append(group)
            val_count += len(group)
        else:
            break

    train_groups = multi_groups + single_groups
    train_records = sorted((record for group in train_groups for record in group), key=lambda record: record.image_path.name)
    val_records = sorted((record for group in val_groups for record in group), key=lambda record: record.image_path.name)
    if {record.image_path.stem for record in train_records} & {record.image_path.stem for record in val_records}:
        raise RuntimeError("Train/validation overlap detected")
    return train_records, val_records


def make_coco(records, split_name, seed, val_ratio, repeat_multibox=1):
    instances = []
    for record in records:
        repeat = repeat_multibox if split_name == "train" and record.is_multi else 1
        instances.extend([record] * repeat)

    images = []
    annotations = []
    annotation_id = 1
    for image_id, record in enumerate(instances, 1):
        images.append(
            {
                "id": image_id,
                "file_name": record.image_path.name,
                "width": record.width,
                "height": record.height,
                "source_stem": record.image_path.stem,
                "repeat_index": sum(
                    previous.image_path.stem == record.image_path.stem
                    for previous in instances[: image_id - 1]
                ),
            }
        )
        for x1, y1, x2, y2 in record.boxes_xyxy:
            width = x2 - x1
            height = y2 - y1
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 0,
                    "bbox": [x1, y1, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    return {
        "info": {
            "description": "Substation detection with preserved multi-object annotations",
            "split": split_name,
            "seed": seed,
            "val_ratio": val_ratio,
            "unique_images": len(records),
            "dataset_instances": len(instances),
            "true_multibox_images": sum(record.is_multi for record in records),
            "train_multibox_repeat": repeat_multibox if split_name == "train" else 1,
        },
        "licenses": [],
        "categories": [{"id": 0, "name": "substation", "supercategory": "facility"}],
        "images": images,
        "annotations": annotations,
    }


def main():
    args = parse_args()
    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio must be in (0, 1)")
    if not 0 <= args.duplicate_iou <= 1:
        raise ValueError("--duplicate-iou must be in [0, 1]")
    if args.train_multibox_repeat < 1:
        raise ValueError("--train-multibox-repeat must be positive")

    image_paths = sorted(
        path for path in args.images.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    label_paths = sorted(args.labels.glob("*.txt"))
    if len(image_paths) != len(label_paths):
        raise RuntimeError(f"Image/label count differs: {len(image_paths)} vs {len(label_paths)}")

    records = []
    deduplicated_rows = 0
    for image_path in image_paths:
        label_path = args.labels / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        with Image.open(image_path) as image:
            width, height = image.size
        boxes, source_count = read_yolo_boxes(label_path, width, height, args.duplicate_iou)
        deduplicated_rows += source_count - len(boxes)
        records.append(Record(image_path, width, height, boxes, source_count, len(boxes)))

    group_for_stem = load_scene_groups(args.scene_groups, {record.image_path.stem for record in records})
    train_records, val_records = split_records(
        records, group_for_stem, args.val_ratio, args.seed, args.min_val_multi_images
    )
    train_coco = make_coco(
        train_records, "train", args.seed, args.val_ratio, args.train_multibox_repeat
    )
    val_coco = make_coco(val_records, "val", args.seed, args.val_ratio)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "instances_train.json").write_text(
        json.dumps(train_coco, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "instances_val.json").write_text(
        json.dumps(val_coco, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "schema_version": 2,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "duplicate_iou": args.duplicate_iou,
        "train_multibox_repeat": args.train_multibox_repeat,
        "label_corpus_sha256": sha256_files(label_paths),
        "total_unique_images": len(records),
        "source_box_count_distribution": dict(Counter(record.source_box_count for record in records)),
        "deduplicated_box_count_distribution": dict(Counter(len(record.boxes_xyxy) for record in records)),
        "deduplicated_rows": deduplicated_rows,
        "true_multibox_images": sum(record.is_multi for record in records),
        "train_unique_images": len(train_records),
        "val_unique_images": len(val_records),
        "train_multibox_images": sum(record.is_multi for record in train_records),
        "val_multibox_images": sum(record.is_multi for record in val_records),
        "train_dataset_instances_after_repeat": len(train_coco["images"]),
        "train_files": [record.image_path.name for record in train_records],
        "val_files": [record.image_path.name for record in val_records],
        "scene_group_file": str(args.scene_groups.resolve()) if args.scene_groups else None,
    }
    (args.output / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in manifest.items() if key not in {"train_files", "val_files"}}, indent=2))


if __name__ == "__main__":
    main()
