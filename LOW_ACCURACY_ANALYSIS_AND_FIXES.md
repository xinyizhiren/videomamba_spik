# 低准确率诊断与修复记录

## 日志结论

本次拉取并分析的最新日志来自：

- `outputs/videomamba_small_cv_train12_test3_ann/log.txt`

该日志包含 22 个 epoch。主要现象如下：

- `train_acc1`：最低约 `6.84%`，最高约 `11.11%`，第 21 个 epoch 为 `9.19%`
- `train_acc5`：大多在 `40%` 到 `49%` 区间，第 21 个 epoch 为 `45.73%`
- `val_acc1`：主要在 `8.82%` 到 `13.24%` 区间
- `val_acc5`：最高约 `52.21%`
- `train_loss`：从 `4.89` 降到 `4.62`
- `val_loss`：从 `2.40` 降到约 `2.31`

对 12 类分类任务来说，随机 Top-1 约为 `8.33%`。因此当前训练集和验证集 Top-1 都接近随机水平，不是单纯的过拟合问题，而是模型没有稳定学到有效分类边界。

## 关于“噪声是否过强”的判断

`datasets/kinetics_sparse_et.py` 中确实存在高斯噪声、雨、雾、模糊、对比度扰动等函数，但当前训练流程没有真正调用这些显式扰动：

- train 分支没有调用 `apply_perturbation`
- test 分支里的扰动调用目前是注释状态
- engine 中缩略图/跨视角嵌入增强的 `participation_rate = 0`，因此也没有生效

所以当前低准确率不是由这些显式噪声直接造成的。

不过，当前训练仍然存在较强的普通数据增强：

- `aa` 默认是 `rand-m7-n4-mstd0.5-inc1`
- 训练随机裁剪面积下限是 `0.08`
- 默认开启水平翻转

对于每个视角只有几百条样本的跨视角行为识别任务，这些增强很可能过强。尤其是随机裁剪下限 `0.08` 时，动作主体或关键交互区域可能被裁掉；水平翻转也可能改变摄像机视角几何关系。

## 关于时序特征是否有效处理的判断

VideoMamba 的主体结构是有时序建模的：

- `PatchEmbed` 使用 `Conv3d`
- 输入形状为 `B, C, T, H, W`
- patch 后保留时间维 `T`
- 代码将 token 重排为跨时间的 token 序列
- 加入 `temporal_pos_embedding`
- Mamba blocks 对完整时空 token 序列进行建模

但本次检查发现一个更关键的问题：视频级 `cls token` 的取法存在样本错位风险。

原逻辑在 `B*T` 个帧级 token 中使用：

```python
cls_tokens = x[:B, :1, :]
```

在当前 reshape 顺序下，`x[:B]` 并不等价于“每个视频取一个 cls token”。当 batch size 大于 1 时，它实际更像是取第一个视频的前若干个时间片 token。这会导致视频级分类 token 与样本不严格对齐，训练信号会被破坏，表现就会接近随机。

## 本次修复

### 1. 修复视频级 cls token

文件：

- `models/videomamba.py`

修复后不再从 `B*T` 的帧级 token 中切片拿视频级 cls，而是显式构造 batch 级视频 cls：

```python
cls_tokens = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, :1, :]
```

patch token 仍然使用空间位置编码和时间位置编码，然后与视频级 cls 拼接送入 Mamba blocks。

### 2. 默认关闭过强 RandAugment

文件：

- `datasets/kinetics_sparse_et.py`
- `exp/k400/videomamba_small/run_f16x224_ann_et_local.sh`

现在 `aa` 支持 `none/off/false/0/disabled` 等关闭值。ANN 跨视角启动脚本默认：

```bash
AUTO_AUG=none
```

### 3. 调整训练随机裁剪强度

文件：

- `run_class_finetuning_et.py`
- `datasets/kinetics_sparse_et.py`
- `exp/k400/videomamba_small/run_f16x224_ann_et_local.sh`

新增参数：

- `--train_crop_min_scale`
- `--train_crop_max_scale`
- `--train_crop_min_ratio`
- `--train_crop_max_ratio`

ANN 跨视角启动脚本默认：

```bash
TRAIN_CROP_MIN_SCALE=0.50
TRAIN_CROP_MAX_SCALE=1.0
```

这会避免训练时频繁裁到过小区域。

### 4. 默认关闭训练水平翻转

文件：

- `run_class_finetuning_et.py`
- `datasets/kinetics_sparse_et.py`
- `exp/k400/videomamba_small/run_f16x224_ann_et_local.sh`

新增参数：

```bash
--disable_train_flip
```

ANN 跨视角启动脚本默认启用该参数。若后续想重新打开水平翻转，可在运行脚本时设置：

```bash
DISABLE_TRAIN_FLIP=0 bash exp/k400/videomamba_small/run_f16x224_ann_et_local.sh
```

### 5. 保留显式断点续训机制

本次也保留了上一轮修改：默认不再自动从 `OUTPUT_DIR` 断点续训。只有显式设置 `RESUME_PATH` 时才恢复训练。

## 建议复现实验

建议使用新的输出目录，避免与旧日志和旧 checkpoint 混在一起：

```bash
cd ~/code/VIT_4090/videomamba/video_sm && \
git pull --ff-only origin main && \
CUDA_VISIBLE_DEVICES=1 \
OUTPUT_DIR=./outputs/videomamba_small_cv_train12_test3_ann_clsfix_augsoft \
LOG_DIR=./logs/videomamba_small_cv_train12_test3_ann_clsfix_augsoft \
bash exp/k400/videomamba_small/run_f16x224_ann_et_local.sh
```

## 下一步观察指标

如果修复有效，前 5 到 10 个 epoch 至少应该看到：

- `train_acc1` 明显高于随机水平 `8.33%`
- `train_acc1` 不再长期卡在 `7%` 到 `11%`
- `train_loss` 继续下降，并且 `train_acc1` 有同步上升趋势

如果仍然接近随机，下一步应重点检查：

- CSV 标签是否从 `0` 到 `11` 且类别顺序一致
- 训练视角 1 与视角 2 是否确实是同一动作样本的配对
- checkpoint 是否真实加载了 backbone 权重
- 当前 12 类是否与预训练 Kinetics 类别差异过大，需要冻结 backbone 先只训 head 做 sanity check
