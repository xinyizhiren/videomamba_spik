# VideoMamba 24-Block LIF SNN Training

This repository is now trimmed around the active SNN workflow while retaining
the clean ANN training/evaluation scripts used to build the teacher checkpoint:

- base network: clean VideoMamba Small
- SNN wrapper: `models/videomamba_trainable_snn.py`
- spike layer: SpikingJelly `MultiStepLIFNode`
- spike mode: unsigned `{0, 1}`
- spike scope: all 24 VideoMamba blocks, post-block insertion
- SNN timesteps: `4`
- input clip: `16` sampled video frames at `224 x 224`

Older ANN-to-SNN conversion, staged SNN sweeps, report builders, and historical
outputs are preserved under `trash_code/`.

## Active Layout

```text
.
|-- datasets/
|   `-- multiview_action_clean.py
|-- exp/
|   |-- run_f16x224_ann_et_clean.sh
|   |-- run_f16x224_ann_et_clean_scratch.sh
|   |-- run_f16x224_ann_et_clean_test.sh
|   |-- run_f16x224_ann_et_clean_valtest.sh
|   |-- run_f16x224_ann_et_clean_valtest_test.sh
|   |-- run_f16x224_lif_snn_b0-23_from_b0-11_train.sh
|   `-- run_f16x224_trainable_snn.sh
|-- models/
|   |-- mamba_simple.py
|   |-- videomamba.py
|   |-- videomamba_clean.py
|   `-- videomamba_trainable_snn.py
|-- tools/
|   `-- merge_val_test_csv.py
|-- outputs/
|   |-- videomamba_small_cv_train12_test3_ann_clean_full/
|   |-- videomamba_small_lif_unsigned_snn_b0-1-2-3-4-5-6-7-8-9-10-11_t4_from_b0-5/
|   `-- videomamba_small_lif_unsigned_snn_b0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23_t4_from_b0-11/
|-- trash_code/
|-- run_class_finetuning_et_clean.py
|-- requirements.txt
`-- .gitignore
```

The two retained non-active output folders are dependencies for the active run:

- `videomamba_small_cv_train12_test3_ann_clean_full`: clean ANN teacher.
- `videomamba_small_lif_unsigned_snn_b0-...-11_t4_from_b0-5`: default initialization checkpoint for the 24-block run.

## Multi-GPU Training

The active launcher defaults to two GPUs and uses `torchrun` through the shared
SNN launcher:

```bash
bash exp/run_f16x224_lif_snn_b0-23_from_b0-11_train.sh
```

Default distributed settings:

```text
CUDA_VISIBLE_DEVICES=0,1
NPROC_PER_NODE=2
NNODES=1
MASTER_PORT=29505
```

Single-card override:

```bash
CUDA_VISIBLE_DEVICES=2 NPROC_PER_NODE=1 bash exp/run_f16x224_lif_snn_b0-23_from_b0-11_train.sh
```

Common overrides:

```bash
EPOCHS=30 LR=1e-5 BATCH_SIZE=1 UPDATE_FREQ=2 bash exp/run_f16x224_lif_snn_b0-23_from_b0-11_train.sh
```

The active script writes to:

```text
outputs/videomamba_small_lif_unsigned_snn_b0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23_t4_from_b0-11/
```

## ANN Teacher Training

The clean ANN training scripts are kept in `exp/` because the SNN route depends
on the ANN teacher and checkpoint initialization.

```bash
bash exp/run_f16x224_ann_et_clean.sh
```

Fine-tune the ANN teacher from an existing checkpoint into a new output folder:

```bash
MODEL_PATH=outputs/videomamba_small_cv_train12_test3_ann_clean_full/best.pth \
JOB_NAME=videomamba_small_ann_refine_from_best \
OUTPUT_DIR=outputs/videomamba_small_ann_refine_from_best \
LR=1e-5 EPOCHS=20 WARMUP_EPOCHS=1 DROP_PATH=0.05 \
bash exp/run_f16x224_ann_et_clean.sh
```

Use multiple GPUs by setting one process per visible GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 BATCH_SIZE=4 bash exp/run_f16x224_ann_et_clean.sh
```

Useful ANN entries:

- `exp/run_f16x224_ann_et_clean.sh`: train/fine-tune clean ANN from K400 weights.
- `exp/run_f16x224_ann_et_clean_scratch.sh`: train clean ANN from random initialization.
- `exp/run_f16x224_ann_et_clean_test.sh`: test a trained clean ANN checkpoint.
- `exp/run_f16x224_ann_et_clean_valtest.sh`: validation/test utility.

## Git Notes

Large files remain ignored:

- `*.pth`
- `*.pt`
- `*.ckpt`
- `*.npy`
- `*.npz`
- TensorBoard and other binary artifacts

Training logs and metadata are still trackable under `outputs/`:

- `log.txt`
- `*_log.txt`
- `model_summary.txt`
- `run_metadata.json`
- `spike_stats.json`

Before committing:

```bash
git add -A :/
git status --short
git diff --cached --name-status
git commit -m "update 24-block lif snn training"
git push origin main
```
