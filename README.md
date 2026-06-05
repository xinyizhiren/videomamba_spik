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
|   |-- run_f16x224_ann_et_clean_test.sh
|   |-- run_f16x224_ann_et_clean_valtest.sh
|   |-- run_f16x224_ann_et_clean_valtest_test.sh
|   |-- run_f16x224_lif_snn_b0-23_from_b0-11_train.sh
|   |-- run_f16x224_lif_snn_b0-23_from_valtest_ann_train.sh
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
|   |-- videomamba_small_cv_train12_test3_ann_clean_scratch/
|   |-- videomamba_small_cv_train12_valtest_ann_clean_full/
|   `-- videomamba_small_lif_unsigned_snn_b0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23_t4_from_b0-11/
|-- trash_code/
|-- run_class_finetuning_et_clean.py
|-- requirements.txt
`-- .gitignore
```

The two retained non-active output folders are dependencies for the active run:

- `videomamba_small_cv_train12_test3_ann_clean_full`: clean ANN teacher.
- `videomamba_small_cv_train12_valtest_ann_clean_full`: stronger ANN teacher used by the valtest 24-block SNN run.

## Multi-GPU Training

The active launcher defaults to the third GPU as a single-card run and uses the
shared SNN launcher:

```bash
bash exp/run_f16x224_lif_snn_b0-23_from_valtest_ann_train.sh
```

Default distributed settings:

```text
CUDA_VISIBLE_DEVICES=2
NPROC_PER_NODE=1
NNODES=1
MASTER_PORT=29506
```

The valtest launcher ignores generic inherited variables such as `MODEL_PATH`,
`RESUME_PATH`, `DISTILL_WEIGHT`, `USE_CHECKPOINT`, and `CHECKPOINT_NUM`. Use the
`VALTEST_*` namespace for intentional overrides.

Two-card override:

```bash
VALTEST_CUDA_VISIBLE_DEVICES=0,1 VALTEST_NPROC_PER_NODE=2 bash exp/run_f16x224_lif_snn_b0-23_from_valtest_ann_train.sh
```

Common overrides:

```bash
VALTEST_EPOCHS=30 VALTEST_LR=1e-5 VALTEST_BATCH_SIZE=1 VALTEST_UPDATE_FREQ=2 bash exp/run_f16x224_lif_snn_b0-23_from_valtest_ann_train.sh
```

The valtest 24-block launcher keeps gradient checkpointing disabled by default:

```text
USE_CHECKPOINT=0
CHECKPOINT_NUM=0
DUMP_MODEL_SUMMARY=0
DUMP_SPIKE_STATS=0
```

If memory is still tight, use these overrides in order:

```bash
VALTEST_SNN_TIMESTEPS=2 bash exp/run_f16x224_lif_snn_b0-23_from_valtest_ann_train.sh
```

The active script writes to:

```text
outputs/videomamba_small_lif_unsigned_snn_b0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23_t4_from_valtest_ann/
```

Training prints the best validation result at the end and also writes it to
`best_result.json`.

Resume the valtest 24-block SNN from its latest checkpoint with the same launcher:

```bash
bash exp/run_f16x224_lif_snn_b0-23_from_valtest_ann_train.sh resume
```

This resolves to:

```text
outputs/videomamba_small_lif_unsigned_snn_b0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23_t4_from_valtest_ann/latest.pth
```

You can also resume any checkpoint through the same launcher:

```bash
VALTEST_RESUME_PATH=outputs/.../latest.pth bash exp/run_f16x224_lif_snn_b0-23_from_valtest_ann_train.sh
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
- `best_result.json`

Before committing:

```bash
git add -A :/
git status --short
git diff --cached --name-status
git commit -m "update 24-block lif snn training"
git push origin main
```
