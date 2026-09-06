# GridSight-B16 model card

## Summary

GridSight-B16 is a single-class horizontal-box detector for substations in
high-resolution overhead imagery. It uses a DINOv3 ViT-B/16 backbone adapted to
the DEIMv2 real-time DETR-style detector.

## Intended use

- Research and competition evaluation on overhead RGB imagery.
- Assisted substation candidate localization.
- Batch production of normalized YOLO-format text results.

It is not intended to make safety-critical power-grid decisions without human
review. Performance may change across sensors, ground sample distances,
countries, seasons, compression levels, or non-RGB imagery.

## Training

- Pretrained backbone: `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`.
- Backbone source: Meta DINOv3, LVD-1689M pretraining.
- Detector initialization and architecture: DEIMv2 derivative.
- Task data: competition overhead images converted from YOLO to COCO format.
- Final adaptation: deterministic hard-negative fine-tuning, seed 3407.
- Final checkpoint: best EMA parameters only.

## Inference

The released recipe evaluates 640 and 768 pixel square inputs, with and without
horizontal flipping. Accepted boxes are grouped by IoU and confidence-weighted.
If no view reaches confidence 0.5, the image is treated as negative. The public
submission recipe emits at most one box per image.

## Metrics

| Split | Metric | Value |
|---|---:|---:|
| Fixed validation | COCO AP | 0.967665 |
| Fixed validation | AP50 | 0.980131 |
| Fixed validation | AP75 | 0.980131 |
| Competition public evaluation | score | 0.91386 |

## Checkpoint integrity

`gridsight_b16_substation_v1.0.0.pth`

```text
SHA-256 31200AE4A102EDF8A82CC7A8DE68F88C7EACF935AF822D9BBFFC515D68066C24
```

The checkpoint contains inference parameters and omits optimizer and scheduler
states. It is suitable for inference or weight tuning, not exact interruption
recovery.

## Attribution

GridSight-B16 is a project-level release name, not a claim that the underlying
DEIMv2 or DINOv3 methods were created by this repository. See
the source notices, `LICENSE`, and the upstream links in `README.md`.
