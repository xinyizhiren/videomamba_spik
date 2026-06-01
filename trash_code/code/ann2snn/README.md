# VideoMamba ANN 到 SNN 转换说明

## 目录结构

当前与 ANN 到 SNN 转换相关的文件已经统一封装到 `ann2snn/` 目录下：

- `ann2snn/slayers.py`
- `ann2snn/spike_utils.py`
- `ann2snn/videomamba_ann2snn.py`
- `ann2snn/convert_videomamba_ann_to_snn.py`
- `ann2snn/dump_videomamba_layer_order.py`
- `ann2snn/VIDEOMAMBA_LAYER_ORDER_REFERENCE.txt`

这样后续转换、校准、层顺序导出和说明文档都在同一个地方，便于维护。

## 设计思路

这次不是重新训练一个原生 SNN VideoMamba，而是把 `run_f16x224_ann_et_clean.sh` 训练得到的 clean ANN checkpoint 转成可做脉冲推理的版本。

参考的是 `AutoPhaseNN-main/PyTorch/train.py` 的转换思路，但没有直接照搬。原因是 `VideoMamba` 和典型 ReLU CNN 差别很大：

- `patch_embed` 后没有显式 ReLU。
- Mamba block 的 token 特征既有正值也有负值。
- 主干不是简单的 `Conv-BN-ReLU` 串联结构，不能直接做逐层 ReLU 替换。

所以这里采用的是“插入式转换”：

- 在 `patch_embed` 输出后可选插入一个脉冲层。
- 在若干个 `Mamba block` 输出后插入脉冲层。
- 采用 `signed spike` 方案，尽量保留原始正负特征信息。

## 为什么要单独导出层顺序

ANN 到 SNN 转换对层执行顺序很敏感。为了避免只靠人工猜测，这里额外提供了层顺序导出脚本，用来打印模型在真实前向过程中的模块顺序和输出形状。

这对于确认以下内容很重要：

- `patch_embed` 的真实插入位置。
- 哪些 block 更适合加入脉冲层。
- 后续如果要把 `spike_block_indices` 从 `0,1` 扩展到 `0,1,2,3`，能否有依据地调整。

## 默认转换策略

第一版默认策略偏保守：

- `spike_patch=True`
- `spike_block_indices=0,1`
- `signed_spikes=True`
- `timesteps=16`

这样做的目的是先保证转换链路稳定，再逐步扩大 SNN 化范围。

## 使用方法

### 1. 导出层顺序

```bash
python ann2snn/dump_videomamba_layer_order.py \
  --checkpoint ./outputs/videomamba_small_cv_train12_test3_ann_clean_full/best.pth \
  --output ./outputs/ann2snn_videomamba/videomamba_layer_order.txt
```

### 2. 执行 ANN 到 SNN 转换

```bash
python ann2snn/convert_videomamba_ann_to_snn.py \
  --checkpoint ./outputs/videomamba_small_cv_train12_test3_ann_clean_full/best.pth \
  --data_path /data_hdd/oyys/VIT_4090/dataset/data/multiview_action_videos/ \
  --prefix /data_hdd/oyys/VIT_4090/dataset/data/multiview_action_videos/ \
  --output_dir ./outputs/ann2snn_videomamba \
  --calib_view_csv aligned_v01_1.csv \
  --val_view_csv v03_val_set.csv \
  --test_view_csv v03_test_set.csv \
  --dump_layer_order
```

默认流程会做这些事情：

- 加载 ANN clean checkpoint。
- 构造对应的 SNN 版本模型。
- 使用校准集做阈值平衡。
- 自动估计 `delay`。
- 在验证集和测试集上做一次 SNN 推理评估。
- 保存转换后的 `videomamba_ann2snn.pth` 和 `conversion_summary.json`。

## 常用参数

### `--spike_block_indices`

控制在哪些 `Mamba block` 后插入脉冲层，例如：

```bash
--spike_block_indices 0,1
--spike_block_indices 0,1,2,3
```

### `--no_spike_patch`

关闭 `patch_embed` 后的脉冲层。如果发现精度下降明显，建议优先尝试这个开关。

### `--unsigned_spikes`

关闭 `signed spike`。通常不建议在 `VideoMamba` 上这么做，因为它的 token 特征明显带有正负值。

## 建议实验顺序

建议按下面顺序尝试：

1. `spike_patch=True, spike_block_indices=0,1`
2. `spike_patch=False, spike_block_indices=0,1`
3. `spike_patch=True, spike_block_indices=0,1,2,3`

先看哪种配置在 `val/test` 上最接近 ANN 结果，再决定是否继续扩大 SNN 化范围。
