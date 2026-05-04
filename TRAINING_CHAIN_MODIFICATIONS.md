# VideoMamba ET 训练链路与本次修改说明

更新时间：2026-05-04

## 本次修改范围

本次只做脚本注释和说明文档整理，没有改动训练逻辑、模型结构、数据读取或默认超参数。

已修改：

- `exp/k400/videomamba_small/run_f16x224_ann_et_clean.sh`
- `exp/k400/videomamba_small/run_f16x224_ann_et_clean_test.sh`
- `exp/k400/videomamba_small/run_f16x224_ann_et_local.sh`
- `TRAINING_CHAIN_MODIFICATIONS.md`

主要修改内容：

- 给三个 `.sh` 文件中的关键参数加了中文注释。
- 明确标注 `clean.sh` 是正式 clean 训练入口。
- 明确标注 `clean_test.sh` 是复用 clean 入口的评估脚本。
- 明确标注 `local.sh` 是旧 ET 本地实验链路，主要用于历史结果复现和对照。

## 三个脚本对应关系

### clean 训练脚本

`exp/k400/videomamba_small/run_f16x224_ann_et_clean.sh`

对应 Python 入口：

```bash
run_class_finetuning_et_clean.py
```

用途：

- 正式 clean 训练。
- 使用 view1 和 view2 作为训练双视角。
- 使用 view3 validation CSV 作为验证视角。
- 默认输出到 `outputs/videomamba_small_cv_train12_test3_ann_clean_full`。

核心链路：

```text
run_f16x224_ann_et_clean.sh
  -> run_class_finetuning_et_clean.py
  -> datasets/multiview_action_clean.py
     -> CrossViewTrainDataset 读取 train_view1_csv + train_view2_csv
     -> SingleViewDataset 读取 val_view_csv
  -> models/videomamba_clean.py
     -> CleanVideoMamba
     -> create_videomamba_small_clean()
  -> CrossEntropyLoss
  -> AdamW
  -> latest.pth / best.pth
```

clean 训练中的 loss：

```text
fused_logits = 0.5 * (view1_logits + view2_logits)
loss = FUSED_CE_LOSS_WEIGHT * CE(fused_logits, target)
     + VIEW_CE_LOSS_WEIGHT * 0.5 * (CE(view1_logits, target) + CE(view2_logits, target))
```

默认 `FUSED_CE_LOSS_WEIGHT=1.0`、`VIEW_CE_LOSS_WEIGHT=1.0`，也就是同时监督融合输出和两个单视角输出。

### clean 测试脚本

`exp/k400/videomamba_small/run_f16x224_ann_et_clean_test.sh`

对应 Python 入口：

```bash
run_class_finetuning_et_clean.py --eval
```

用途：

- 只做评估，不训练。
- 默认读取 clean 训练目录下的 `best.pth`。
- 默认 `EVAL_SPLIT=test`，读取 `v03_test_set.csv`。
- 可通过 `EVAL_SPLIT=validation` 切换到 `v03_val_set.csv`。

核心链路：

```text
run_f16x224_ann_et_clean_test.sh
  -> run_class_finetuning_et_clean.py --eval
  -> load_eval_checkpoint(best.pth)
  -> datasets/multiview_action_clean.py
     -> SingleViewDataset 读取 test 或 validation CSV
  -> model(video)
  -> CrossEntropyLoss + acc1/acc5
  -> test_log.txt 或 validation_log.txt
```

### local 脚本

`exp/k400/videomamba_small/run_f16x224_ann_et_local.sh`

对应 Python 入口：

```bash
run_class_finetuning_et.py
```

用途：

- 旧 ET 本地实验链路。
- 主要价值是复现历史实验、和 clean 链路做对照、做 EDL/aux loss 等旧逻辑消融。
- 正式训练不建议优先用它。

核心链路：

```text
run_f16x224_ann_et_local.sh
  -> run_class_finetuning_et.py
  -> datasets/build.py
  -> datasets/kinetics_sparse_et.py
     -> train: 双视角输入
     -> validation/test: 单视角输入
  -> timm.create_model("videomamba_small")
  -> models/videomamba.py
  -> engines/engine_for_finetuning_et.py
  -> utils.save_model()
```

local 旧链路中保留的历史功能更多：

- DDP / DeepSpeed 相关逻辑。
- TensorBoard log_dir。
- step-level cosine scheduler。
- layer decay。
- AutoAugment / RandomErasing / 多 crop 测试 merge。
- EDL 风格 `ET_AUX_LOSS_WEIGHT` 辅助损失开关。
- `models/videomamba.py` 中的 GAP/evidence 融合逻辑。

local 默认 `FUSED_CE_LOSS_WEIGHT=0.0`、`VIEW_CE_LOSS_WEIGHT=1.0`，也就是默认只监督两个单视角 CE；clean 默认同时监督 fused CE 和 view CE。

## local 和 clean 的主要区别

| 项目 | clean | local |
| --- | --- | --- |
| 是否正式推荐 | 是 | 否，历史对照用 |
| Python 入口 | `run_class_finetuning_et_clean.py` | `run_class_finetuning_et.py` |
| 数据集实现 | `datasets/multiview_action_clean.py` | `datasets/build.py` -> `datasets/kinetics_sparse_et.py` |
| 模型入口 | `models/videomamba_clean.py` | `timm.create_model("videomamba_small")` -> `models/videomamba.py` |
| 训练复杂度 | 最小可控链路 | 旧实验大链路 |
| 默认 loss | fused CE + view CE | view CE，保留 fused/ET aux 开关 |
| 学习率含义 | `LR` 是实际学习率 | `BASE_LR` 会按有效 batch size 线性缩放 |
| 默认 epoch | 40 | 80 |
| 默认 GPU | 1 | 0 |
| checkpoint 命名 | `latest.pth` / `best.pth` | 走 `utils.save_model()` 的旧命名 |
| 测试方式 | 单视角直接评估 | 旧多 segment/crop + merge |

## 是否保留 local.sh

建议暂时保留。

理由：

- 它不是正式训练入口，但可以复现 `outputs/videomamba_small_cv_train12_test3_ann` 这一类历史实验。
- 它保留了 EDL/aux loss、旧数据增强、旧测试 merge 等消融能力。
- 当前 clean 链路虽然更适合作为正式训练入口，但删除 local 会丢掉和旧结果对照的启动方式。

后续如果确认不再需要旧实验复现，可以再删除或移动到 `legacy/` 目录。

## 正式使用建议

正式训练：

```bash
bash exp/k400/videomamba_small/run_f16x224_ann_et_clean.sh
```

正式测试：

```bash
bash exp/k400/videomamba_small/run_f16x224_ann_et_clean_test.sh
```

如果只是排查训练链路是否能过拟合小样本：

```bash
DEBUG_OVERFIT_SAMPLES=48 \
EPOCHS=80 \
WEIGHT_DECAY=0 \
DROP_PATH=0 \
bash exp/k400/videomamba_small/run_f16x224_ann_et_clean.sh
```
