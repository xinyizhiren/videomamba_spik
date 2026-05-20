# VideoMamba ANN2SNN Experiment Log

## 2026-05-20 Current Results

Source ANN checkpoint:

- Path: `outputs/videomamba_small_cv_train12_test3_ann_clean_full/best.pth`
- Source ANN best_acc1: `94.1176`
- No-spike sanity check: normal, around `94`, so checkpoint loading, dataset loading, wrapper forward, and evaluation are valid.

Post-training ANN-to-SNN insertion ablation:

| run | patch spike | block spikes | threshold scale | timesteps | delay | calib_mse | val ann_acc1 | val snn_acc1 | val drop | test snn_acc1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `block0` | False | `0` | 1.0 | 16 | 0 | 0.001027 | 94.1176 | 74.2647 | -19.8529 | 72.6708 |
| `block01` | False | `0,1` | 1.0 | 16 | 1 | 0.001193 | 94.1176 | 67.6471 | -26.4706 | 66.1491 |
| `block0123` | False | `0,1,2,3` | 1.0 | 16 | 3 | 0.001442 | 94.1176 | 51.4706 | -42.6471 | 55.2795 |

## Interpretation

The current post-training conversion is functional but loses too much accuracy.

- A single spike insertion after `layers.0` already drops validation accuracy from `94.1176` to `74.2647`.
- Accuracy decreases monotonically as more early blocks are spiked.
- The error accumulation is large enough that simply adding more converted layers is not a good direction.
- The clean VideoMamba backbone has no `ReLU/ReLU6` modules; it has `SiLU` inside Mamba mixers. Traditional `ReLU -> spike neuron` replacement therefore does not apply.
- Directly replacing Mamba `SiLU` is risky because it participates in Mamba convolution/gating/scan behavior, not only in a simple feed-forward activation path.

Conclusion: pure post-training ANN-to-SNN insertion is not suitable as the main path for this dataset/model.

## Next Direction

Move from pure conversion to trainable SNN fine-tuning initialized from the ANN checkpoint.

The dataset is small, so the SNN must reuse the strong ANN-pretrained representation instead of training a different SNN architecture from scratch.

Preferred route:

1. Keep the clean VideoMamba structure as the base.
2. Load all compatible ANN checkpoint weights into the SNN model.
3. Add a small number of spike modules at controlled boundaries.
4. Train the SNN model end-to-end, so the backbone can adapt to spike-induced quantization/noise.
5. Expand the spiking scope only after the smaller trainable version recovers accuracy.

This is different from the current `convert_videomamba_ann_to_snn.py` route:

- Current route: load ANN weights, calibrate spike thresholds, evaluate directly.
- Next route: load ANN weights, initialize spike layers, then fine-tune with gradients.

## Recommended Trainable SNN Design

Do not make `models/videomamba_spik_baseline_1_fixed.py` the main path yet.

Reason:

- It is a separate native SNN-style architecture.
- Its module names and tensor shapes are not close enough to the clean VideoMamba checkpoint.
- Reusing the current ANN `best.pth` may be weak or partial.

Use a trainable variant of the current converted wrapper instead:

- Base class: `ConvertedVideoMambaSNN` / clean VideoMamba.
- Reuse ANN parameters for `patch_embed`, all `layers.*.mixer`, all `layers.*.norm`, `norm_f`, and `head`.
- New trainable or stateful parts: spike modules at selected boundaries.
- Initial spike locations for training:
  - start with `block0`;
  - if it recovers, try `block0,1`;
  - only then try deeper scopes.

Training details to add:

- Reset spike state at each batch.
- Average or accumulate logits over `T` timesteps.
- Start with `T=4` or `T=8` for training cost, then evaluate with `T=16`.
- Use supervised CE loss.
- Add optional ANN-teacher distillation because the dataset is small:
  - frozen ANN teacher from the same `best.pth`;
  - KL/logit MSE loss between SNN logits and ANN logits;
  - optionally feature MSE at the spiked boundary.
- Use a small learning rate for pretrained ANN weights and a larger one for spike parameters.

Suggested first trainable experiment:

```bash
MODEL=videomamba_small_trainable_snn
SNN_BLOCK_INDICES=0
SNN_TIMESTEPS=4
FINETUNE=outputs/videomamba_small_cv_train12_test3_ann_clean_full/best.pth
```

Success criterion:

- `block0` trainable SNN should recover clearly above the post-training `74.2647`.
- If it reaches `85+`, the trainable wrapper direction is worth expanding.
- If it stays near `74`, the spike layer design/thresholding needs to change before adding more blocks.

## Implementation Plan

1. Add a trainable SNN model entry, for example `videomamba_small_trainable_snn`.
2. Reuse the converted wrapper structure but expose training-time options:
   - `snn_block_indices`
   - `snn_spike_patch`
   - `snn_timesteps`
   - `snn_signed_spikes`
3. Add spike-state reset before each forward/training batch.
4. Add ANN-teacher distillation support to the training script.
5. Add a dedicated launcher under `exp/`, initialized from the ANN `best.pth`.
6. Run tiny-overfit first, then full train.

## 2026-05-20 Implementation Update

Added the first trainable SNN fine-tuning path:

- Model entry: `videomamba_small_trainable_snn`.
- Launcher: `exp/run_f16x224_trainable_snn.sh`.
- Default setup: spike after block `0`, `T=4`, no patch spike, signed trainable thresholds.
- Initialization: load compatible weights from the trained ANN `best.pth`.
- Training aid: frozen ANN teacher distillation from the same checkpoint, controlled by `DISTILL_WEIGHT`.

First recommended smoke run:

```bash
DEBUG_OVERFIT_SAMPLES=48 EPOCHS=3 BATCH_SIZE=1 SNN_TIMESTEPS=2 bash exp/run_f16x224_trainable_snn.sh
```

First full run:

```bash
bash exp/run_f16x224_trainable_snn.sh
```

## 2026-05-20 Trainable SNN Smoke Result

Command:

```bash
DEBUG_OVERFIT_SAMPLES=48 EPOCHS=3 BATCH_SIZE=1 SNN_TIMESTEPS=2 bash exp/run_f16x224_trainable_snn.sh
```

Result:

| run | blocks | T | train samples | epoch | train_acc1 | val_acc1 | val_loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `videomamba_small_trainable_snn_b0_t2` | `0` | 2 | 48 | 0 | 100.00 | 92.6471 | 0.3122 |
| `videomamba_small_trainable_snn_b0_t2` | `0` | 2 | 48 | 1 | 100.00 | 92.6471 | 0.3030 |
| `videomamba_small_trainable_snn_b0_t2` | `0` | 2 | 48 | 2 | 100.00 | 92.6471 | 0.2923 |

Interpretation:

- The trainable SNN path is working.
- The result is far above the post-training `block0` SNN accuracy of `74.2647`.
- Validation loss decreased across the smoke run, so the spike-inserted model can still optimize after ANN initialization.
- Next step: run the full `block0, T=4` training with the default launcher.

## 2026-05-20 Full Run Memory Note

The first full `block0, T=4` attempt with `BATCH_SIZE=2` hit a backward-time cuDNN initialization error. This is likely a memory/workspace pressure issue because trainable SNN keeps graphs for both views across multiple timesteps.

Launcher defaults were adjusted for the trainable SNN path:

- `BATCH_SIZE=1`
- `UPDATE_FREQ=2`
- `CUDNN_BENCHMARK=0`

This keeps the effective batch size close to the original setting while reducing peak activation memory.

## 2026-05-20 Trainable `block0, T=4` Result

Log path:

```text
outputs/videomamba_small_trainable_snn_b0_t4/log.txt
```

Partial full-run result:

| run | blocks | T | epoch | train_acc1 | val_acc1 | val_loss |
| --- | --- | --- | --- | --- | --- | --- |
| `videomamba_small_trainable_snn_b0_t4` | `0` | 4 | 0 | 99.3671 | 92.6471 | 0.2442 |
| `videomamba_small_trainable_snn_b0_t4` | `0` | 4 | 1 | 99.5781 | 94.1176 | 0.2007 |
| `videomamba_small_trainable_snn_b0_t4` | `0` | 4 | 2 | 99.1561 | 92.6471 | 0.3337 |
| `videomamba_small_trainable_snn_b0_t4` | `0` | 4 | 3 | 98.9451 | 94.8529 | 0.1940 |
| `videomamba_small_trainable_snn_b0_t4` | `0` | 4 | 4 | 100.0000 | 93.3824 | 0.2311 |
| `videomamba_small_trainable_snn_b0_t4` | `0` | 4 | 5 | 99.7890 | 91.1765 | 0.3125 |

Interpretation:

- Best validation accuracy is `94.8529`, which is above the source ANN `94.1176`.
- Trainable SNN recovered the post-training `block0` drop (`74.2647`) and is strong enough to expand the spiking scope.
- Accuracy fluctuates because the validation set is small, so compare by best checkpoint and final test accuracy, not a single epoch.

Next spiking-scope experiment:

1. Initialize `block0,1` from the trained `block0` SNN `best.pth`.
2. Keep the frozen ANN teacher as the original clean ANN `best.pth`.
3. Keep `T=4`, `BATCH_SIZE=1`, `UPDATE_FREQ=2` first.
4. If `block0,1` can stay near `90+`, expand to `block0,1,2,3`.

Recommended command:

```bash
bash exp/run_f16x224_trainable_snn_scope_ablation.sh
```

Originally this ran `block01_from_block0` by default. After the `block0,1` result, use explicit jobs for the next expansion:

```bash
ABLATION_JOBS=block0123_from_block01 bash exp/run_f16x224_trainable_snn_scope_ablation.sh
```

## 2026-05-20 Trainable `block0,1, T=4` Result

Log path:

```text
outputs/videomamba_small_trainable_snn_b0-1_t4_from_b0/log.txt
```

Result:

| run | blocks | T | init | epoch | train_acc1 | val_acc1 | val_loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `videomamba_small_trainable_snn_b0-1_t4_from_b0` | `0,1` | 4 | `block0 best` | 0 | 99.7890 | 93.3824 | 0.2359 |
| `videomamba_small_trainable_snn_b0-1_t4_from_b0` | `0,1` | 4 | `block0 best` | 1 | 100.0000 | 91.9118 | 0.2644 |
| `videomamba_small_trainable_snn_b0-1_t4_from_b0` | `0,1` | 4 | `block0 best` | 2 | 99.3671 | 88.2353 | 0.4779 |

Interpretation:

- `block0,1` does not collapse: the first epoch reaches `93.3824`, still close to the ANN baseline.
- Later epochs drift downward as learning rate warms up to `3e-5`, so wider SNN scopes should use a shorter scout run, slightly lower LR, and stronger ANN distillation.
- Because `block0,1` is viable, the next step can be bolder than one-block-at-a-time expansion.

Updated scope-ablation defaults:

- default jobs: `block0123_from_block01 block0to7_from_block01`
- `EPOCHS=6`
- `LR=2e-5`
- `WARMUP_EPOCHS=1`
- `DISTILL_WEIGHT=0.7`

Recommended bold scout:

```bash
bash exp/run_f16x224_trainable_snn_scope_ablation.sh
```

Optional high-spike probe that also spikes the patch embedding:

```bash
ABLATION_JOBS=patch_block01_from_block01 bash exp/run_f16x224_trainable_snn_scope_ablation.sh
```

## 2026-05-20 Trainable `block0..3, T=4` Result

Log path:

```text
outputs/videomamba_small_trainable_snn_b0-3_t4_from_b0-1/log.txt
```

Script mapping confirms this run uses the trainable SNN model:

- job: `block0123_from_block01`
- model: `videomamba_small_trainable_snn`
- init checkpoint: `outputs/videomamba_small_trainable_snn_b0-1_t4_from_b0/best.pth`
- `SNN_BLOCK_INDICES=0,1,2,3`
- `SNN_SPIKE_PATCH=0`
- `SNN_TIMESTEPS=4`

Result:

| run | blocks | T | init | epoch | train_acc1 | val_acc1 | val_loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `videomamba_small_trainable_snn_b0-3_t4_from_b0-1` | `0,1,2,3` | 4 | `block0,1 best` | 0 | 99.7890 | 94.8529 | 0.1952 |
| `videomamba_small_trainable_snn_b0-3_t4_from_b0-1` | `0,1,2,3` | 4 | `block0,1 best` | 1 | 99.7890 | 90.4412 | 0.2955 |

Interpretation:

- The `0..3` trainable SNN is valid and strong: epoch 0 reaches `94.8529`, matching the best `block0` result.
- This supports a bolder expansion beyond `0..3`.
- Because epoch 1 drops, keep the scout setting conservative (`LR=2e-5`, `DISTILL_WEIGHT=0.7`) and compare best checkpoints.

Recommended next step from this result:

```bash
ABLATION_JOBS=block0to7_from_block0123 bash exp/run_f16x224_trainable_snn_scope_ablation.sh
```

If we want to skip directly toward higher spike coverage:

```bash
ABLATION_JOBS=block0to11_from_block0123 bash exp/run_f16x224_trainable_snn_scope_ablation.sh
```

## 2026-05-20 Code Path Check

The trainable SNN code does insert spike layers, but the insertion point is controlled:

- `models/videomamba_trainable_snn.py` creates `block_spikes` from `SNN_BLOCK_INDICES`.
- `forward_features()` calls `self.block_spikes[key](hidden_states)` after each selected VideoMamba block.
- The spiked `hidden_states` is then passed into later blocks and into the final residual/norm path.
- `SNN_SPIKE_PATCH=0` means the patch embedding spike is not active unless explicitly enabled.

Repeated `94.8529` first-epoch validation accuracy does not by itself mean the spike modules are inactive. The validation set has 136 samples, so:

- `94.8529` means 129/136 correct.
- `94.1176` means 128/136 correct.
- Several nearby models can land on the same integer correct count.

To make future logs easier to audit, training now prints active spike modules and writes `run_metadata.json` with:

- `model_class`
- `active_spike_modules`
- `spike_block_indices`
- `spike_patch`
- `snn_timesteps`

## 2026-05-20 Initial Validation Update

Training now evaluates the validation set once before the first optimization step.

The pre-training row is appended to `log.txt` with:

- `mode: initial_eval`
- `epoch: start_epoch - 1`
- `checkpoint: resume or finetune checkpoint`
- validation metrics only

This makes each scope expansion easier to audit:

- initial validation shows the immediate effect of adding the new spike layers;
- epoch validation shows what training did after that.

For trainable SNN runs, spike thresholds are initialized from one training batch before this initial validation, so the validation set is not used for threshold calibration.

Use `--skip_initial_eval` only when intentionally skipping this extra validation pass.

## 2026-05-20 Full Mamba-Block Spike Update

Because wider scopes still have strong pre-training validation, the default scope-ablation job now jumps from the trained `0..3` SNN directly to all 24 Mamba blocks:

```bash
bash exp/run_f16x224_trainable_snn_scope_ablation.sh
```

This defaults to:

- job: `block0to23_from_block0123`
- init checkpoint: `outputs/videomamba_small_trainable_snn_b0-3_t4_from_b0-1/best.pth`
- `SNN_BLOCK_INDICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23`
- `SNN_SPIKE_PATCH=0`
- `SNN_TIMESTEPS=4`

The patch embedding spike is still kept off. Test it separately after the all-block spike result is known.

Model structure output:

- training now calls `from torchinfo import summary` when `DUMP_MODEL_SUMMARY=1`;
- default launcher sets `DUMP_MODEL_SUMMARY=1`;
- summary is written to `outputs/<job>/model_summary.txt`;
- active spike modules are also printed at startup and saved in `run_metadata.json`.
