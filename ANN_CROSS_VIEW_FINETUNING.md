# ANN Cross-View Finetuning

## Goal

This update adds an explicit ANN fine-tuning path for the protocol:

- train on view 1 and view 2
- validate on the held-out view 3
- test on the unseen view 3

This matches the cross-view evaluation setting where the model is expected to learn action cues that transfer to a camera viewpoint not seen during training.

## What Changed

### 1. Explicit dataset entry for the ET loader

`datasets/build.py` now supports:

- `Kinetics_sparse_et`

This entry uses `datasets/kinetics_sparse_et.py` directly for:

- dual-view training
- held-out single-view validation
- held-out single-view testing

The old `Kinetics_sparse` path is still kept for compatibility.

### 2. CSV filenames are configurable from CLI

`run_class_finetuning_et.py` now accepts:

- `--train_view1_csv`
- `--train_view2_csv`
- `--val_view_csv`
- `--test_view_csv`

Default protocol:

- `aligned_v01_1.csv`
- `aligned_v02_2.csv`
- `v03_val_set.csv`
- `v03_test_set.csv`

So the default setup is:

- train on view 1 + view 2
- validate/test on view 3

### 3. New local ANN launcher

Added:

- `exp/k400/videomamba_small/run_f16x224_ann_et_local.sh`

This script uses:

- `run_class_finetuning_et.py`
- `videomamba_small`
- pretrained checkpoint `videomamba_s16_k400_f16_res224.pth`
- `Kinetics_sparse_et`
- relative project path discovery, so the repo can be moved without editing the script
- optional environment overrides for `DATA_PATH`, `PREFIX`, `MODEL_PATH`, `OUTPUT_DIR`, and `LOG_DIR`
- stable video ids for held-out-view testing, so multi-crop / multi-segment merge stays at video level

## Recommended Finetuning Settings

The dataset is relatively small, with only a few hundred samples per view, so the launcher uses a conservative fine-tuning setup:

- model: `videomamba_small`
- frames: `16`
- sampling rate: `4`
- input size: `224`
- batch size: `4`
- update freq: `2`
- effective batch size: `8` on one GPU
- epochs: `80`
- base learning rate: `3.2e-3`
- effective learning rate after repo linear scaling: `1e-4`
- layer decay: `0.8`
- fc drop rate: `0.1`
- drop path: `0.2`
- weight decay: `0.05`
- warmup epochs: `5`
- test views: `4` temporal segments x `3` spatial crops

These values are chosen to reduce overfitting risk while still letting the pretrained ANN backbone adapt to the cross-view action dataset.

The launcher uses a larger CLI `--lr` on purpose because `run_class_finetuning_et.py` rescales the learning rate by:

```text
lr = lr * total_batch_size / 256
```

With `batch_size=4` and `update_freq=2`, the effective batch size is `8`, so:

```text
3.2e-3 * 8 / 256 = 1e-4
```

This keeps the actual optimization step size in the intended range for single-GPU fine-tuning.

## How To Run

```bash
bash exp/k400/videomamba_small/run_f16x224_ann_et_local.sh
```

If the dataset or checkpoint lives elsewhere on another machine, you can override paths at runtime:

```bash
DATA_PATH=/your/dataset/root \
PREFIX=/your/dataset/root \
MODEL_PATH=/your/checkpoints/videomamba_s16_k400_f16_res224.pth \
bash exp/k400/videomamba_small/run_f16x224_ann_et_local.sh
```

## How To Change The Held-Out View

You can change the protocol without editing Python code. Just swap the CSV arguments in the launcher.

Example:

- train on view 2 + view 3
- validate/test on view 1

```bash
--train_view1_csv 'aligned_v02_2.csv'
--train_view2_csv 'aligned_v03_3.csv'
--val_view_csv 'v01_val_set.csv'
--test_view_csv 'v01_test_set.csv'
```

This makes cross-view experiments easier to repeat across different hold-out settings.
