# 最简 ET 训练链路说明

## 目的

旧链路混入了较多历史实验逻辑，例如 EDL 辅助损失、HSIC、GradCAM、Deepspeed、复杂测试 merge、自动断点恢复、扰动增强等。当前准确率坍缩问题已经属于训练链路 bug 排查，因此新增一条独立 clean 链路，用来做最小可控实验。

## Clean 文件

- `datasets/multiview_action_clean.py`：只负责 CSV 读取、Decord 视频读取、时序采样、resize/crop/normalize。
- `models/videomamba_clean.py`：只暴露 clean VideoMamba small，forward 只返回单视角 logits 或双视角平均 logits。
- `run_class_finetuning_et_clean.py`：只负责训练、验证、日志和 checkpoint。
- `exp/k400/videomamba_small/run_f16x224_ann_et_clean.sh`：服务器启动器。

## 保留内容

- 双视角训练数据加载。
- VideoMamba ANN 模型。
- 预训练 checkpoint 加载。
- 双视角训练：`view1_logits`、`view2_logits`。
- 最简单融合：`fused_logits = (view1_logits + view2_logits) / 2`。
- CrossEntropyLoss。
- AdamW。
- epoch 级验证。
- `log.txt` 记录 loss、acc、类别直方图。
- `latest.pth` 和 `best.pth` 保存。

## 删除内容

- Deepspeed / DDP。
- 自动从输出目录断点恢复。
- EDL 辅助损失。
- HSIC。
- StillMix / 缩略图嵌入增强。
- GradCAM。
- 多 crop 测试 merge。
- 隐式学习率缩放。
- 旧 `kinetics_sparse_et.py` 的扰动、随机擦除、AutoAugment 等实验逻辑。
- 旧 `videomamba.py` forward 中的 GAP/EDL/HSIC 双视角逻辑。

## 重要差异

clean 链路中 `LR` 就是实际学习率，不再乘以 `batch_size / 256`。

VideoMamba 默认使用 mean pooling：

```python
hidden_states[:, 1:, :].mean(dim=1)
```

如果需要回到旧的 cls token 路径，可以给 Python 脚本传：

```bash
--use_cls
```

## 推荐 sanity check

先跑均衡小样本 overfit：

```bash
cd ~/code/VIT_4090/videomamba/video_sm && \
git pull --ff-only origin main && \
CUDA_VISIBLE_DEVICES=1 \
OUTPUT_DIR=./outputs/et_clean_tiny48 \
DEBUG_OVERFIT_SAMPLES=48 \
EPOCHS=80 \
LR=3e-4 \
WEIGHT_DECAY=0 \
DROP_PATH=0 \
TRAIN_CROP_MIN_SCALE=1.0 \
TRAIN_CROP_MAX_SCALE=1.0 \
bash exp/k400/videomamba_small/run_f16x224_ann_et_clean.sh
```

判断标准：

- 如果 `train_loss` 明显低于 `2.0` 且 `train_fused_acc1` 快速上升，说明训练链路已打通。
- 如果仍然卡在 `2.30` 左右，则优先检查数据内容、标签、模型特征是否近似常量。
- 如果训练集已经能 overfit 但 `val_pred_hist` 仍全落到单一类别，说明问题更偏向跨视角泛化。
