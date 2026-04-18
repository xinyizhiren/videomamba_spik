# ANN 跨视角微调说明

## 实验目标

当前配置用于多视角行为识别中的跨视角泛化实验：

- 使用视角 1 和视角 2 进行训练
- 使用未参与训练的视角 3 进行验证
- 使用未参与训练的视角 3 进行测试

这个设置的核心目的不是单纯记住某个摄像头视角下的外观，而是判断模型是否学到了可以迁移到新视角的动作信息。

## 主要改动

### 1. 显式使用 ET 数据加载器

`datasets/build.py` 已支持：

- `Kinetics_sparse_et`

这个入口会直接使用 `datasets/kinetics_sparse_et.py`，对应的数据流程是：

- 训练阶段：双视角输入
- 验证阶段：单个保留视角输入
- 测试阶段：单个保留视角输入

原来的 `Kinetics_sparse` 入口仍然保留，用于兼容旧实验。

### 2. CSV 文件可通过命令行配置

`run_class_finetuning_et.py` 支持以下参数：

- `--train_view1_csv`
- `--train_view2_csv`
- `--val_view_csv`
- `--test_view_csv`

默认跨视角协议为：

- `aligned_v01_1.csv`
- `aligned_v02_2.csv`
- `v03_val_set.csv`
- `v03_test_set.csv`

也就是说，默认设置是：

- 视角 1 + 视角 2 训练
- 视角 3 验证和测试

### 3. ANN 本地启动脚本

新增或更新的启动脚本为：

- `exp/k400/videomamba_small/run_f16x224_ann_et_local.sh`

该脚本使用：

- `run_class_finetuning_et.py`
- `videomamba_small`
- 预训练权重 `videomamba_s16_k400_f16_res224.pth`
- `Kinetics_sparse_et`
- 相对路径自动定位项目根目录
- 可通过环境变量覆盖 `DATA_PATH`、`PREFIX`、`MODEL_PATH`、`OUTPUT_DIR`、`LOG_DIR`
- 测试阶段使用稳定的视频级 id，避免多 segment / 多 crop 合并时退化成按图片合并

## 推荐微调参数

当前数据集每个视角只有几百条样本，因此脚本采用偏保守的微调设置：

- 模型：`videomamba_small`
- 帧数：`16`
- 采样间隔：`4`
- 输入尺寸：`224`
- batch size：`6`
- update freq：`2`
- 单卡等效 batch size：`12`
- 训练轮数：`80`
- 命令行基础学习率：`3.2e-3`
- 仓库线性缩放后的实际学习率：`1.5e-4`
- layer decay：`0.8`
- fc drop rate：`0.0`
- drop path：`0.1`
- RandAugment：默认关闭，`AUTO_AUG=none`
- 训练随机裁剪面积范围：`0.50` 到 `1.0`
- 训练水平翻转：默认关闭
- weight decay：`0.05`
- warmup epochs：`5`
- 测试视角采样：`4` 个 temporal segment x `3` 个 spatial crop
- 默认损失：融合输出 CE + 单视角 CE
- ET/EDL 辅助损失：默认关闭，可通过 `ET_AUX_LOSS_WEIGHT` 打开

这些参数的目标是降低小数据集过拟合风险，同时让 ANN 预训练 backbone 有足够空间适配当前跨视角行为识别数据。

脚本里的 `--lr` 看起来比较大，是因为 `run_class_finetuning_et.py` 会按总 batch size 做线性缩放：

```text
lr = lr * total_batch_size / 256
```

当 `batch_size=6` 且 `update_freq=2` 时，等效 batch size 为 `12`，所以：

```text
3.2e-3 * 12 / 256 = 1.5e-4
```

这个设置能让单卡微调的实际优化步长维持在相对合理的范围。

当前脚本默认采用较弱的数据增强。原因是当前每个视角只有几百条样本，且任务目标是跨视角泛化；过强的 RandAugment、过小的随机裁剪区域或水平翻转都可能破坏动作主体和视角几何关系。如果要重新打开强增强，可以通过环境变量覆盖：

```bash
AUTO_AUG=rand-m7-n4-mstd0.5-inc1 \
TRAIN_CROP_MIN_SCALE=0.08 \
DISABLE_TRAIN_FLIP=0 \
bash exp/k400/videomamba_small/run_f16x224_ann_et_local.sh
```

## 损失函数口径

ET 训练现在会同时监督：

- 双视角融合输出
- 训练视角 1 的单视角输出
- 训练视角 2 的单视角输出

这样做很重要，因为验证和测试阶段都是单视角、保留视角评估。如果只监督融合输出，模型可能在训练时依赖双视角互补信息，到了单视角测试时表现会很差。

分类 CE 使用 raw logits 计算，包括融合输出和单视角输出。`softplus` 后得到的 evidence 只用于可选的 ET/EDL 辅助损失。这样训练、验证、测试的分类口径是一致的，避免预训练模型看起来“没有发挥作用”。

如果需要做消融实验，可以重新打开 ET/EDL 辅助损失：

```bash
ET_AUX_LOSS_WEIGHT=5.0 bash exp/k400/videomamba_small/run_f16x224_ann_et_local.sh
```

## 预训练与断点续训

这里要区分两个概念：

- `MODEL_PATH` 是预训练权重，用于从 Kinetics 预训练模型开始微调。
- `RESUME_PATH` 是断点续训权重，用于恢复某一次已经开始的训练，包括模型参数、optimizer、epoch 和 scaler。

默认情况下，脚本不会再自动从 `OUTPUT_DIR` 里的旧 checkpoint 断点续训。即使输出目录里已经有 `checkpoint-latest.pth` 或 `checkpoint-best.pth`，也会从 `MODEL_PATH` 指定的预训练模型重新开始微调。

只有显式设置 `RESUME_PATH` 时，才会断点续训。

从某个输出目录恢复：

```bash
RESUME_PATH=./outputs/videomamba_small_cv_train12_test3_ann \
bash exp/k400/videomamba_small/run_f16x224_ann_et_local.sh
```

从某个具体 checkpoint 文件恢复：

```bash
RESUME_PATH=./outputs/videomamba_small_cv_train12_test3_ann/checkpoint-latest.pth \
bash exp/k400/videomamba_small/run_f16x224_ann_et_local.sh
```

如果只是想开一个全新的实验，推荐显式指定新的输出目录：

```bash
OUTPUT_DIR=./outputs/videomamba_small_cv_train12_test3_ann_new \
LOG_DIR=./logs/videomamba_small_cv_train12_test3_ann_new \
bash exp/k400/videomamba_small/run_f16x224_ann_et_local.sh
```

## 运行方式

默认运行：

```bash
bash exp/k400/videomamba_small/run_f16x224_ann_et_local.sh
```

如果服务器上的数据集或 checkpoint 在其它位置，可以这样覆盖：

```bash
DATA_PATH=/your/dataset/root \
PREFIX=/your/dataset/root \
MODEL_PATH=/your/checkpoints/videomamba_s16_k400_f16_res224.pth \
bash exp/k400/videomamba_small/run_f16x224_ann_et_local.sh
```

## 如何更换保留视角

如果要换成其它跨视角设置，不需要改 Python 代码，只需要替换启动脚本中的 CSV 参数。

例如：

- 视角 2 + 视角 3 训练
- 视角 1 验证和测试

对应参数可以改成：

```bash
--train_view1_csv 'aligned_v02_2.csv'
--train_view2_csv 'aligned_v03_3.csv'
--val_view_csv 'v01_val_set.csv'
--test_view_csv 'v01_test_set.csv'
```

这样可以比较方便地复现实验，并测试不同保留视角下的泛化能力。
