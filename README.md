# VideoMamba Clean Cross-View Training

This repository is trimmed for one active workflow: clean cross-view ANN fine-tuning from
`exp/run_f16x224_ann_et_clean.sh`, with the optional fixed
SpikMamba model and ANN-to-SNN conversion utilities kept for follow-up experiments.

## Repository Layout

```text
.
├── ann2snn/                                  # ANN-to-SNN conversion helpers
├── datasets/
│   ├── __init__.py
│   └── multiview_action_clean.py             # CSV video loader and transforms
├── exp/
│   ├── run_f16x224_ann_et_clean.sh           # main clean training entry
│   ├── run_f16x224_ann_et_clean_scratch.sh   # train from random initialization
│   ├── run_f16x224_ann_et_clean_test.sh      # eval-only entry
│   ├── run_f16x224_ann_et_clean_valtest.sh   # optional val+test validation run
│   └── run_f16x224_ann_et_clean_valtest_test.sh
├── models/
│   ├── mamba_simple.py
│   ├── videomamba.py
│   ├── videomamba_clean.py
│   └── videomamba_spik_baseline_1_fixed.py   # optional SpikMamba model
├── scripts/
│   ├── run_server.sh
│   ├── sync_results_back.sh
│   └── sync_to_server.sh
├── tools/
│   └── merge_val_test_csv.py
├── run_class_finetuning_et_clean.py
├── requirements.txt
└── .gitignore
```

The untrimmed local snapshot before cleanup is preserved on the `main_old` branch.

## Training

Put CSV annotation files and videos outside Git, then point the script to those paths:

```bash
DATA_PATH=/data/users/ouyangys/data/multiview_action_videos \
PREFIX=/data/users/ouyangys/data/multiview_action_videos \
MODEL_PATH=/path/to/videomamba_s16_k400_f16_res224.pth \
bash exp/run_f16x224_ann_et_clean.sh
```

Single-node multi-GPU training uses `torchrun` automatically when `NPROC_PER_NODE` is
greater than 1:

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash exp/run_f16x224_ann_et_clean.sh
```

Train from scratch without loading K400 weights:

```bash
bash exp/run_f16x224_ann_et_clean_scratch.sh
```

The scratch launcher uses a separate output directory and defaults to longer training
(`EPOCHS=120`, `WARMUP_EPOCHS=10`, `LR=3e-4`, `DROP_PATH=0.05`). Override any value the
same way as the main launcher, for example:

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 LR=1e-4 bash exp/run_f16x224_ann_et_clean_scratch.sh
```

Outputs are written under `outputs/` by default. Checkpoints and generated artifacts are
ignored by Git, while `log.txt`, `*_log.txt`, and `.log` files under `outputs/` are
trackable. The default pretrained checkpoint name in the script is
`videomamba_s16_k400_f16_res224.pth`; keep that file local or on the server, not in the
repository.

Evaluation uses the same Python entry in `--eval` mode:

```bash
CHECKPOINT_PATH=outputs/videomamba_small_cv_train12_test3_ann_clean_full/best.pth \
bash exp/run_f16x224_ann_et_clean_test.sh
```

To instantiate the retained pulse model through the clean entry, set:

```bash
MODEL_NAME=spikmamba_fixed bash exp/run_f16x224_ann_et_clean.sh
```

## Collaboration Workflow

Use Git for source code, scripts, docs, and training logs. Do not add data, pretrained
weights, checkpoints, tensorboard events, `runs/`, or `wandb/`.

Typical local flow:

```bash
git status
bash exp/run_f16x224_ann_et_clean.sh
git diff
git add -A :/
git status --short
git diff --cached --name-status
git commit -m "describe the training change"
git push origin main
```

Before committing from the server, this quick check should print nothing:

```bash
git diff --cached --name-only | grep -E '\.(pth|pt|ckpt|npy|npz|h5|hdf5)$' || true
```

Server flow:

```bash
git pull
bash scripts/run_server.sh
```

For temporary server sync before committing, configure `REMOTE` and `REMOTE_DIR`, then use:

```bash
DRY_RUN=1 REMOTE=user@server REMOTE_DIR=/path/to/video_sm bash scripts/sync_to_server.sh
DRY_RUN=0 REMOTE=user@server REMOTE_DIR=/path/to/video_sm bash scripts/sync_to_server.sh
```

`DRY_RUN=1` is the default. Flip it only after checking the rsync plan.
