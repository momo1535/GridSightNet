# GridSightNet

基于 DEIMv2 与 Meta DINOv3 ViT-B/16 的遥感变电站检测模型。此目录保留最终模型、数据准备、训练及推理流程。来源与版权保留在源码及 LICENSE 中。

## 文件

- `gridsight/`：模型和训练框架。
- `configs/`：最终配置及必要依赖。
- `train.py`：训练和断点续训。
- `predict.py`：640/768 水平翻转 TTA 推理。
- `tools/`：YOLO 数据转换、难负样本生成、推理实现、权重检查。
- `MODEL_CARD.md`：模型说明。

## 环境与权重

使用 Python 3.10，执行 `pip install -r requirements.txt`。以下命令均在项目根目录执行。

将最终完整检测器权重放到 `weights/gridsight_b16_substation_v1.0.0.pth`。约 414 MiB，上传 GitHub 时单独放 Releases。加载该文件推理或微调不需要另下 Backbone 权重。

不加载检测器权重时，需从 Meta 获取 `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` 放到 `weights/`。

## 数据准备

YOLO 数据放在项目同级 `dataset/train/images`、`dataset/train/labels` 和 `dataset/test/images`。

```bash
python tools/prepare_substation_coco_multibox.py --images ../dataset/train/images --labels ../dataset/train/labels --output ../dataset/annotations_multibox_v2 --seed 3407 --min-val-multi-images 0
python tools/prepare_substation_hardneg.py --annotations ../dataset/annotations_multibox_v2/instances_train.json --images ../dataset/train/images --output ../dataset/hardneg_v1 --seed 3407 --crop-size 512 --margin 48 --random-candidates 24
```

训练读取 COCO，推理输出 YOLO。多框标签保留；同场景图片可通过 `--scene-groups` 指定分组，避免跨训练与验证集。

## 训练与续训

当前配置是 10 epoch 难负样本微调，不是原始 150 epoch 方案。从最终模型继续训练：

```bash
python train.py -c configs/gridsight/gridsight_b16_substation.yml -t weights/gridsight_b16_substation_v1.0.0.pth --seed 3407 --use-amp --device cuda:0
```

中断后使用训练生成的完整检查点：

```bash
python train.py -c configs/gridsight/gridsight_b16_substation.yml -r outputs/gridsight_b16_substation_seed3407/last.pth --seed 3407 --use-amp --device cuda:0
```

启动前设置 `PYTHONHASHSEED=3407`、`DEIM_DETERMINISTIC=1`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`，并保持数据划分、依赖与硬件一致。Release EMA 文件无优化器状态，不能作为严格断点续训文件。此精简版不含历史实验记录，不承诺完整复现历史分数。

## 推理

```bash
python predict.py --config configs/gridsight/gridsight_b16_substation.yml --checkpoint weights/gridsight_b16_substation_v1.0.0.pth --input-dir ../dataset/test/images --output-root ./inference/result --submission-dir ./submissions --submission-name submit_GridSight.zip --scales 640 768 --confidence-threshold 0.5 --fusion-iou 0.5 --device cuda:0 --batch-size 2 --amp
```

输出到 `inference/result/submit/`，同时生成 ZIP。每图一个 UTF-8 TXT，第一行为 `class_id x_center y_center width height confidence`。坐标归一化，类别为 0。所有视图置信度均低于 0.5 时只有表头，否则融合后最多输出一个框。该规则可能漏掉一图多站中的次要目标。

## 改进来源

- [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)：检测架构与训练实现。
- [DINOv3](https://github.com/facebookresearch/dinov3)：ViT-B/16 主干及预训练权重，适用 Meta 对应许可。

更名不改变底层方法来源。历史竞赛线上分数为 0.91386，详见模型说明。
