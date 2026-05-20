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
