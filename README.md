# VideoMamba Clean Cross-View Training

This repository is trimmed for one active workflow: clean cross-view ANN fine-tuning from
`exp/k400/videomamba_small/run_f16x224_ann_et_clean.sh`, with the optional fixed
SpikMamba model and ANN-to-SNN conversion utilities kept for follow-up experiments.

## Repository Layout

```text
.
├── ann2snn/                                  # ANN-to-SNN conversion helpers
├── datasets/
│   ├── __init__.py
│   └── multiview_action_clean.py             # CSV video loader and transforms
├── exp/k400/videomamba_small/
│   ├── run_f16x224_ann_et_clean.sh           # main clean training entry
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
bash exp/k400/videomamba_small/run_f16x224_ann_et_clean.sh
```

Outputs are written under `outputs/` by default and are ignored by Git. The default
pretrained checkpoint name in the script is `videomamba_s16_k400_f16_res224.pth`; keep
that file local or on the server, not in the repository.

Evaluation uses the same Python entry in `--eval` mode:

```bash
CHECKPOINT_PATH=outputs/videomamba_small_cv_train12_test3_ann_clean_full/best.pth \
bash exp/k400/videomamba_small/run_f16x224_ann_et_clean_test.sh
```

To instantiate the retained pulse model through the clean entry, set:

```bash
MODEL_NAME=spikmamba_fixed bash exp/k400/videomamba_small/run_f16x224_ann_et_clean.sh
```

## Collaboration Workflow

Use Git for source code, scripts, and docs only. Do not add data, pretrained weights,
checkpoints, logs, `outputs/`, `runs/`, or `wandb/`.

Typical local flow:

```bash
git status
bash exp/k400/videomamba_small/run_f16x224_ann_et_clean.sh
git diff
git add .
git commit -m "describe the training change"
git push origin main
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
