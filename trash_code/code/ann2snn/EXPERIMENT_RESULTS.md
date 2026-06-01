# VideoMamba SNN 实验结果整理

更新时间：2026-05-21


## 1. 基线结果

当前所有 SNN 实验都基于 clean VideoMamba ANN 预训练模型：

- ANN checkpoint: `outputs/videomamba_small_cv_train12_test3_ann_clean_full/best.pth`
- validation best acc1: `94.1176`
- test acc1: `92.2360`
- 输入设置：`16 x 224 x 224`
- 类别数：`12`

后续 SNN 结果主要看 validation acc1；test 结果仅在 ANN2SNN 转换阶段有记录。

## 2. ANN2SNN 直接转换失败

最早尝试的是直接 ANN-to-SNN 转换：加载 ANN 权重，插入 spike 层并做阈值校准，不重新训练。转换结果明显不适配 VideoMamba。

| run | spike patch | spiked blocks | T | delay | calib mse | ANN val acc1 | SNN val acc1 | drop | SNN test acc1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `block0` | False | `0` | 16 | 0 | 0.001027 | 94.1176 | 74.2647 | -19.8529 | 72.6708 |
| `block01` | False | `0,1` | 16 | 1 | 0.001193 | 94.1176 | 67.6471 | -26.4706 | 66.1491 |
| `block0123` | False | `0,1,2,3` | 16 | 3 | 0.001442 | 94.1176 | 51.4706 | -42.6471 | 55.2795 |

结论：

- 直接转换只插入 1 个 block 就从 `94.1176` 掉到 `74.2647`。
- 插入 4 个 block 后 validation 只剩 `51.4706`。
- 说明当前 ANN2SNN 校准方式不能直接适配 VideoMamba 的 residual、LayerNorm 和 Mamba block 动态。
- 这条路线暂时不继续作为主线。

## 3. 自定义 TrainableSpike3dSeq 脉冲层

### 3.1 层定义

`TrainableSpike3dSeq` 是本项目自定义的 surrogate spike 层，不是 SpikingJelly 标准 LIF。

默认使用 signed 模式：

```text
输出: {-threshold, 0, +threshold}
threshold: 每个 embedding channel 一个阈值
shape: [1, 1, 384]
```

也就是每个 spike 层有 `384` 个可训练阈值。这个设计对精度比较友好，但因为输出带 per-channel threshold scale，不适合作为部署重参数化主线。

### 3.2 不训练直接验证 sweep

设置：

- checkpoint: ANN clean `best.pth`
- spike layer: custom `TrainableSpike3dSeq`
- signed: True
- spike position: post block
- patch spike: False
- `SNN_TIMESTEPS=4`
- `EPOCHS=0`

| spiked blocks count | spiked blocks | val acc1 | val acc5 | val loss |
| --- | --- | --- | --- | --- |
| 1 | `0` | 91.1765 | 99.2647 | 0.3186 |
| 2 | `0,1` | 88.2353 | 99.2647 | 0.3770 |
| 3 | `0,1,2` | 84.5588 | 98.5294 | 0.4669 |
| 4 | `0,1,2,3` | 80.8824 | 98.5294 | 0.5758 |
| 5 | `0..4` | 76.4706 | 97.0588 | 0.7028 |
| 6 | `0..5` | 75.0000 | 97.0588 | 0.7759 |
| 7 | `0..6` | 74.2647 | 97.0588 | 0.8306 |
| 8 | `0..7` | 71.3235 | 98.5294 | 0.8585 |
| 9 | `0..8` | 71.3235 | 99.2647 | 0.8702 |
| 10 | `0..9` | 72.7941 | 99.2647 | 0.8500 |
| 11 | `0..10` | 73.5294 | 99.2647 | 0.8422 |
| 12 | `0..11` | 75.7353 | 99.2647 | 0.8418 |
| 13 | `0..12` | 76.4706 | 99.2647 | 0.8509 |
| 14 | `0..13` | 76.4706 | 99.2647 | 0.8775 |
| 15 | `0..14` | 74.2647 | 97.7941 | 0.9167 |
| 16 | `0..15` | 76.4706 | 97.7941 | 0.9396 |
| 17 | `0..16` | 75.7353 | 97.0588 | 0.9686 |
| 18 | `0..17` | 77.2059 | 97.0588 | 0.9880 |
| 19 | `0..18` | 77.2059 | 97.0588 | 1.0227 |
| 20 | `0..19` | 77.2059 | 97.0588 | 1.0437 |
| 21 | `0..20` | 77.2059 | 97.7941 | 1.0828 |
| 22 | `0..21` | 75.7353 | 97.7941 | 1.1598 |
| 23 | `0..22` | 77.9412 | 97.7941 | 1.2180 |
| 24 | `0..23` | 77.9412 | 98.5294 | 1.3115 |

现象：

- 自定义 signed spike 直接插入后掉点比 SpikingJelly LIF 小很多。
- 24 个 block 不训练仍有 `77.9412`，说明这个自定义层对 ANN 激活分布更温和。
- 但因为输出不是 `{0,1}`，而是 `{-threshold,0,+threshold}`，后续硬件/重参数化意义弱。

### 3.3 分阶段训练结果

| run | init checkpoint | spiked blocks | initial val acc1 | best val acc1 | best epoch | latest/final uploaded val acc1 | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `b0_t4` | ANN | `0` | - | 94.8529 | 3 | 91.1765 | 早期日志无 initial row |
| `b0-1_from_b0` | `b0` | `0,1` | - | 93.3824 | 0 | 88.2353 | 早期日志无 initial row |
| `b0-3_from_b0-1` | `b0-1` | `0..3` | - | 94.8529 | 0 | 90.4412 | 早期日志无 initial row |
| `b0-7_from_b0-3` | `b0-3` | `0..7` | - | 94.8529 | 0 | 94.8529 | 上传日志只有 epoch 0 |
| `b0-11_from_b0-3` | `b0-3` | `0..11` | 91.9118 | 94.1176 | 0 | 91.1765 | 日志存在断续/重复，谨慎参考 |
| `b0-23_from_b0-3` | `b0-3` | `0..23` | 90.4412 | 96.3235 | 35 | 94.1176 | 精度最高，但 signed 自定义层不适合作为重参数化主线 |

结论：

- 自定义层训练后可以把 24 个 block 的 SNN 做到 `96.3235` validation acc1。
- 但它依赖 signed + per-channel threshold 输出，和后续希望的 `{0,1}` 脉冲数据目标不一致。

## 4. SpikingJelly MultiStepLIFNode 脉冲层

### 4.1 层定义

第二条路线使用 SpikingJelly 标准 LIF：

```python
MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
```

当前主线使用 unsigned 模式：

```text
输出: {0, 1}
SNN_SPIKE_LAYER=lif
SNN_SIGNED_SPIKES=0
SNN_TIMESTEPS=4
```

这个路线更符合部署和重参数化方向，但直接从 ANN 权重插入 LIF 后掉点更大，需要分阶段训练恢复。

### 4.2 signed LIF 不训练直接验证 sweep

这次 no-train sweep 使用的是 signed LIF：

- checkpoint: ANN clean `best.pth`
- spike layer: `lif`
- signed: True
- spike position: post block
- patch spike: False
- `SNN_TIMESTEPS=4`
- `EPOCHS=0`

| spiked blocks count | spiked blocks | val acc1 | val acc5/loss note |
| --- | --- | --- | --- |
| 1 | `0` | 91.9118 | loss 0.2879 |
| 2 | `0,1` | 88.9706 | loss 0.3679 |
| 3 | `0,1,2` | 82.3529 | loss 0.6003 |
| 4 | `0..3` | 72.7941 | loss 0.9259 |
| 5 | `0..4` | 70.5882 | loss 0.9855 |
| 6 | `0..5` | 65.4412 | loss 1.0310 |
| 7 | `0..6` | 60.2941 | loss 1.2822 |
| 8 | `0..7` | 59.5588 | loss 1.3066 |
| 9 | `0..8` | 60.2941 | loss 1.3155 |
| 10 | `0..9` | 44.8529 | loss 1.7767 |
| 11 | `0..10` | 38.2353 | loss 1.9703 |
| 12 | `0..11` | 45.5882 | loss 1.7900 |
| 13 | `0..12` | 38.9706 | loss 1.9494 |
| 14 | `0..13` | 46.3235 | loss 1.7696 |
| 15 | `0..14` | 51.4706 | loss 1.6500 |
| 16 | `0..15` | 36.0294 | loss 1.9809 |
| 17 | `0..16` | 36.0294 | loss 1.9629 |
| 18 | `0..17` | 36.7647 | loss 2.0496 |
| 19 | `0..18` | 40.4412 | loss 2.0037 |
| 20 | `0..19` | 37.5000 | loss 1.9402 |
| 21 | `0..20` | 30.1471 | loss 2.0450 |
| 22 | `0..21` | 30.8824 | loss 2.1846 |
| 23 | `0..22` | 22.7941 | loss 2.2423 |
| 24 | `0..23` | 31.6176 | loss 2.2119 |

现象：

- 标准 LIF 对 ANN 激活分布更敏感。
- 3 个 block 仍有 `82.3529`，4 个 block 开始明显崩。
- 24 个 block no-train 只有 `31.6176`。

### 4.3 unsigned LIF 分阶段训练结果

虽然 no-train sweep 是 signed LIF，为了部署目标，训练主线切到了 unsigned LIF，也就是输出严格 `{0,1}`。

| stage | init checkpoint | spiked blocks | initial val acc1 | best val acc1 | best epoch | latest/final uploaded val acc1 | uploaded epochs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `b0-2_from_ann` | ANN | `0,1,2` | 52.2059 | 91.1765 | 2 / 4 | 91.1765 | 0..4 |
| `b0-5_from_b0-2` | `b0-2` | `0..5` | 59.5588 | 88.9706 | 8 | 88.9706 | 0..8 |
| `b0-11_from_b0-5` | `b0-5` | `0..11` | 38.9706 | 78.6765 | 2 | 78.6765 | 0..2 |
| `b0-23_from_b0-11` | `b0-11` | `0..23` | 25.7353 | 83.0882 | 3 | 81.6176 | 0..5 |

对应输出目录：

- `outputs/videomamba_small_lif_unsigned_snn_b0-1-2_t4_from_ann`
- `outputs/videomamba_small_lif_unsigned_snn_b0-1-2-3-4-5_t4_from_b0-2`
- `outputs/videomamba_small_lif_unsigned_snn_b0-1-2-3-4-5-6-7-8-9-10-11_t4_from_b0-5`
- `outputs/videomamba_small_lif_unsigned_snn_b0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23_t4_from_b0-11`

### 4.4 spike 输出确认

已上传的 `spike_stats.json` 确认 unsigned LIF 输出为 `{0,1}`。

`b0-2_from_ann`：

| layer | nonzero fraction | unique values |
| --- | --- | --- |
| `block_spikes.0` | 0.2486 | `[0.0, 1.0]` |
| `block_spikes.1` | 0.1194 | `[0.0, 1.0]` |
| `block_spikes.2` | 0.1972 | `[0.0, 1.0]` |

`b0-5_from_b0-2`：

| layer | nonzero fraction | unique values |
| --- | --- | --- |
| `block_spikes.0` | 0.3231 | `[0.0, 1.0]` |
| `block_spikes.1` | 0.1708 | `[0.0, 1.0]` |
| `block_spikes.2` | 0.2684 | `[0.0, 1.0]` |
| `block_spikes.3` | 0.1933 | `[0.0, 1.0]` |
| `block_spikes.4` | 0.2563 | `[0.0, 1.0]` |
| `block_spikes.5` | 0.0923 | `[0.0, 1.0]` |

`b0-23_from_b0-11`：

- 24 个 active spike layers 全部输出 `{0,1}`。
- nonzero fraction 范围：`0.1383` 到 `0.4732`。
- 平均 nonzero fraction：`0.2918`。
- best uploaded validation acc1: `83.0882` at epoch `3`。

## 5. 当前结论

1. 直接 ANN2SNN 转换失败。
   只插入 1 个 block 就掉到 `74.2647`，4 个 block 掉到 `51.4706`，不适合作为主线。

2. 自定义 `TrainableSpike3dSeq` 精度最好，但部署意义弱。
   signed + per-channel threshold 的设计可以把 24 个 block 训练到 `96.3235`，但输出不是纯 `{0,1}`。

3. SpikingJelly unsigned LIF 是当前更合适的主线。
   它输出严格 `{0,1}`，虽然掉点更大，但通过分阶段训练可以恢复：
   `0..2` 到 `91.1765`，`0..5` 到 `88.9706`，`0..11` 已从 `38.9706` 恢复到 `78.6765`，`0..23` 已从 `25.7353` 恢复到 `83.0882`。

4. 下一步重点。
   继续观察 full `0..23` 是否能在更长训练中恢复到 `85+` 或 `88+`。当前可继续同一输出目录训练，或以 full `0..23` 的 best checkpoint 为起点降低学习率微调。

```bash
bash exp/run_f16x224_lif_snn_b0-23_from_b0-11_train.sh
```
