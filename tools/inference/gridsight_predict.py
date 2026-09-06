"""Generate a max-1 submission with multi-scale horizontal-flip TTA.

For every image, infer at each requested scale using the original and a
horizontally flipped view. Restore flipped boxes to the original coordinates,
then confidence-weight the accepted top-1 boxes. If all view confidences are
below the threshold, write only the competition header.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from gridsight.core import YAMLConfig  # noqa: E402


HEADER = "class_id x_center y_center width height confidence"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class InferenceModel(nn.Module):
    def __init__(self, model: nn.Module, postprocessor: nn.Module) -> None:
        super().__init__()
        self.model = model.deploy()
        self.postprocessor = postprocessor.deploy()

    def forward(self, images: torch.Tensor, sizes: torch.Tensor):
        return self.postprocessor(self.model(images), sizes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--submission-dir", required=True, type=Path)
    parser.add_argument("--submission-prefix", default="submit_GridSight_B16_TTA")
    parser.add_argument(
        "--submission-name",
        help="Exact short ZIP filename, for example submit_GridSight.zip. Overrides the generated name.",
    )
    parser.add_argument("--scales", nargs="+", default=[640, 768], type=int)
    parser.add_argument("--confidence-threshold", default=0.5, type=float)
    parser.add_argument("--candidate-count", default=10, type=int)
    parser.add_argument("--fusion-iou", default=0.5, type=float)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def load_model(config: Path, state: dict, scale: int, candidates: int, device: str):
    cfg = YAMLConfig(str(config))
    cfg.yaml_cfg["DINOv3STAs"]["weights_path"] = None
    cfg.yaml_cfg["PostProcessor"]["num_top_queries"] = candidates
    cfg.yaml_cfg["eval_spatial_size"] = [scale, scale]
    model = cfg.model
    # Anchors and their validity mask are deterministic, resolution-dependent
    # buffers. Rebuild them from eval_spatial_size instead of loading the 640
    # caches stored in the training checkpoint.
    resolution_state = {
        key: value for key, value in state.items()
        if key not in {"decoder.anchors", "decoder.valid_mask"}
    }
    incompatible = model.load_state_dict(resolution_state, strict=False)
    allowed_missing = {"decoder.anchors", "decoder.valid_mask"}
    if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {incompatible}")
    wrapped = InferenceModel(model, cfg.postprocessor).to(device).eval()
    transform = T.Compose([
        T.Resize((scale, scale)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return wrapped, transform


def unflip(box: torch.Tensor, width: int) -> torch.Tensor:
    result = box.clone()
    result[0] = width - box[2]
    result[2] = width - box[0]
    return result


def yolo_box(box: torch.Tensor, width: int, height: int) -> tuple[float, ...]:
    x1, y1, x2, y2 = map(float, box.tolist())
    x1, x2 = sorted((min(max(x1, 0.0), width), min(max(x2, 0.0), width)))
    y1, y2 = sorted((min(max(y1, 0.0), height), min(max(y2, 0.0), height)))
    values = (((x1 + x2) / 2) / width, ((y1 + y2) / 2) / height,
              (x2 - x1) / width, (y2 - y1) / height)
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
        raise ValueError(f"Invalid box: {values}")
    return values


def pair_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    lt = torch.maximum(a[:2], b[:2])
    rb = torch.minimum(a[2:], b[2:])
    inter = torch.clamp(rb - lt, min=0).prod().item()
    aa = torch.clamp(a[2:] - a[:2], min=0).prod().item()
    ab = torch.clamp(b[2:] - b[:2], min=0).prod().item()
    return inter / max(aa + ab - inter, 1e-9)


def consensus_cluster(views: list[tuple[torch.Tensor, float, str]], threshold: float):
    """Choose the largest mutually compatible cluster, then highest score support."""
    clusters = []
    for seed_box, _, _ in views:
        cluster = [view for view in views if pair_iou(seed_box, view[0]) >= threshold]
        clusters.append(cluster)
    return max(clusters, key=lambda c: (len(c), sum(v[1] for v in c), max(v[1] for v in c)))


def validate_and_zip(submit: Path, destination: Path, stems: set[str]) -> dict[int, int]:
    files = sorted(submit.glob("*.txt"))
    if {p.stem for p in files} != stems:
        raise RuntimeError("Output filenames do not match input images")
    histogram: dict[int, int] = {}
    for path in files:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        if not 1 <= len(lines) <= 2 or lines[0] != HEADER:
            raise RuntimeError(f"Invalid submission file: {path}")
        count = len(lines) - 1
        histogram[count] = histogram.get(count, 0) + 1
        if count:
            fields = lines[1].split()
            if len(fields) != 6 or fields[0] != "0":
                raise RuntimeError(f"Invalid detection: {path}: {lines[1]}")
            if not all(0 <= float(v) <= 1 for v in fields[1:]):
                raise RuntimeError(f"Value outside [0,1]: {path}: {lines[1]}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, f"submit/{path.name}")
    return histogram


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or any(s <= 0 or s % 16 for s in args.scales):
        raise ValueError("Scales must be positive multiples of 16")
    images = sorted(p for p in args.input_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images or len({p.stem for p in images}) != len(images):
        raise RuntimeError("Missing images or duplicate stems")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("ema") is not None:
        state, state_source = checkpoint["ema"]["module"], "ema.module"
    else:
        state, state_source = checkpoint["model"], "model"
    predictions = {p.stem: [] for p in images}
    image_sizes: dict[str, tuple[int, int]] = {}
    started = time.perf_counter()

    for scale in args.scales:
        model, transform = load_model(args.config, state, scale,
                                      args.candidate_count, args.device)
        with torch.inference_mode():
            for offset in range(0, len(images), args.batch_size):
                paths = images[offset:offset + args.batch_size]
                originals, flips, sizes = [], [], []
                for path in paths:
                    with Image.open(path) as image:
                        rgb = image.convert("RGB")
                        width, height = rgb.size
                        tensor = transform(rgb)
                    image_sizes[path.stem] = (width, height)
                    originals.append(tensor)
                    flips.append(torch.flip(tensor, dims=[2]))
                    sizes.append((width, height))
                batch = torch.stack(originals + flips).to(args.device, non_blocking=True)
                size_tensor = torch.tensor(sizes + sizes, dtype=torch.float32,
                                           device=args.device)
                with torch.autocast("cuda", torch.float16,
                                    enabled=args.amp and args.device.startswith("cuda")):
                    _, boxes, scores = model(batch, size_tensor)
                n = len(paths)
                for i, path in enumerate(paths):
                    width, _ = sizes[i]
                    for view, tag in ((i, str(scale)), (i + n, f"{scale}_hflip")):
                        best = int(scores[view].argmax().item())
                        box = boxes[view][best].detach().float().cpu()
                        score = float(scores[view][best].detach().float().cpu())
                        predictions[path.stem].append(
                            (unflip(box, width) if view >= n else box, score, tag))
                print(f"scale={scale}: {min(offset + args.batch_size, len(images))}/{len(images)}",
                      flush=True)
        del model
        torch.cuda.empty_cache()

    submit = args.output_root / "submit"
    submit.mkdir(parents=True, exist_ok=True)
    for old in submit.glob("*.txt"):
        old.unlink()
    accepted_hist: dict[int, int] = {}
    consensus_hist: dict[int, int] = {}
    rejected_view_count = 0
    pair_ious, scores_out = [], []
    for image in images:
        views = predictions[image.stem]
        accepted = [(b, s, t) for b, s, t in views if s >= args.confidence_threshold]
        accepted_hist[len(accepted)] = accepted_hist.get(len(accepted), 0) + 1
        output = submit / f"{image.stem}.txt"
        if not accepted:
            output.write_text(HEADER + "\n", encoding="utf-8")
            scores_out.append(max(s for _, s, _ in views))
            continue
        consensus = consensus_cluster(accepted, args.fusion_iou)
        consensus_hist[len(consensus)] = consensus_hist.get(len(consensus), 0) + 1
        rejected_view_count += len(accepted) - len(consensus)
        weights = torch.tensor([s for _, s, _ in consensus])
        boxes = torch.stack([b for b, _, _ in consensus])
        fused = (boxes * weights[:, None]).sum(0) / weights.sum()
        score = max(s for _, s, _ in consensus)
        scores_out.append(score)
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                pair_ious.append(pair_iou(boxes[i], boxes[j]))
        width, height = image_sizes[image.stem]
        cx, cy, bw, bh = yolo_box(fused, width, height)
        output.write_text(
            f"{HEADER}\n0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f} {score:.8f}\n",
            encoding="utf-8")

    epoch = checkpoint.get("last_epoch")
    scale_tag = "-".join(map(str, args.scales))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.submission_name:
        if Path(args.submission_name).name != args.submission_name:
            raise ValueError("--submission-name must be a filename, not a path")
        submission_name = args.submission_name
        if not submission_name.lower().endswith(".zip"):
            submission_name += ".zip"
    else:
        submission_name = (
            f"{args.submission_prefix}{scale_tag}HFlip_max1_{args.checkpoint.stem}_"
            f"e{epoch}_conf0p50_{timestamp}.zip")
    destination = args.submission_dir / submission_name
    detection_hist = validate_and_zip(submit, destination, {p.stem for p in images})
    manifest = {
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_last_epoch": epoch,
        "state_source": state_source,
        "device": args.device,
        "amp": args.amp,
        "scales": args.scales,
        "horizontal_flip": True,
        "fusion": "confidence_weighted_xyxy",
        "fusion_iou": args.fusion_iou,
        "confidence_threshold": args.confidence_threshold,
        "image_count": len(images),
        "views_per_image": 2 * len(args.scales),
        "accepted_view_count_histogram": {str(k): v for k, v in sorted(accepted_hist.items())},
        "consensus_view_count_histogram": {str(k): v for k, v in sorted(consensus_hist.items())},
        "rejected_outlier_view_count": rejected_view_count,
        "detection_count_histogram": {str(k): v for k, v in sorted(detection_hist.items())},
        "mean_pairwise_iou": sum(pair_ious) / len(pair_ious) if pair_ious else None,
        "top_score_min": min(scores_out),
        "top_score_mean": sum(scores_out) / len(scores_out),
        "top_score_max": max(scores_out),
        "elapsed_seconds": time.perf_counter() - started,
        "submission_zip": str(destination.resolve()),
    }
    (args.output_root / "inference_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
