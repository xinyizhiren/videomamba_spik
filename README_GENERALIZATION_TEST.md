# VideoMamba 泛化测试

本文档介绍如何使用 VideoMamba 的泛化测试功能，测试模型在不同扰动条件下的性能表现。

## 功能概述

泛化测试功能可以评估模型在各种视频扰动下的鲁棒性，包括：

1. **高斯模糊** - 模拟失焦或运动模糊
2. **雨点效果** - 模拟雨天条件
3. **雾效果** - 模拟雾天条件
4. **高斯噪声** - 模拟传感器噪声或低光照条件
5. **对比度变化** - 模拟不同光照和摄像设备条件

每种扰动都有不同的严重程度，用于全面评估模型的泛化能力。

## 使用方法

### 1. 直接在测试函数中添加扰动

如果你想在现有的测试流程中添加扰动，可以直接修改`final_test`函数的调用，添加扰动参数：

```python
test_stats = final_test(
    data_loader=data_loader_test,
    model=model,
    device=device,
    file='output.txt',
    amp_autocast=torch.cuda.amp.autocast(),
    perturbation_type='gaussian_blur',  # 指定扰动类型
    perturbation_severity=3  # 指定扰动严重程度(1-5)
)
```

### 2. 使用泛化测试脚本

使用提供的`run_generalization_test.py`脚本可以自动测试多种扰动类型和严重程度：

```bash
python run_generalization_test.py \
  --model videomamba_small \
  --resume /path/to/checkpoint.pth \
  --data-path /path/to/dataset \
  --data-set Kinetics_sparse \
  --output-dir ./generalization_results \
  --nb-classes 10 \
  --num-frames 16 \
  --sampling-rate 2 \
  --input-size 224 \
  --short-side-size 224 \
  --test-num-segment 1 \
  --test-num-crop 1 \
  --batch-size 8
```

## 扰动类型详解

### 1. 高斯模糊 (gaussian_blur)

应用高斯模糊滤波，模拟失焦或运动模糊。

- 严重程度2: 中等模糊 (kernel_size=5)
- 严重程度4: 强烈模糊 (kernel_size=9)

### 2. 雨点效果 (rain)

添加随机雨点，模拟雨天条件。

- 严重程度2: 中等雨量 (rain_density=0.01)
- 严重程度4: 大雨量 (rain_density=0.02)

### 3. 雾效果 (fog)

添加雾效果，模拟雾天条件。

- 严重程度2: 中等雾量 (fog_density=0.2)
- 严重程度4: 浓雾 (fog_density=0.4)

### 4. 高斯噪声 (gaussian_noise)

添加随机高斯噪声，模拟传感器噪声或低光照条件下的噪点。

- 严重程度2: 中等噪声 (std=20)
- 严重程度4: 强烈噪声 (std=40)

### 5. 对比度变化 (contrast)

调整图像对比度，模拟不同光照和摄像设备条件。

- 严重程度2: 中等对比度变化 (factor=1.0)
- 严重程度4: 强烈对比度变化 (factor=1.5)

## 结果分析

测试完成后，在指定的输出目录中将生成以下文件：

1. `generalization_test_results.csv` - 包含所有扰动类型和严重程度下的测试结果
2. `generalization_acc1.png` - 不同扰动类型和严重程度下的Top-1准确率对比图
3. `generalization_heatmap.png` - 准确率热力图
4. `generalization_acc_drop.png` - 相对于原始准确率的下降百分比图
5. `generalization_summary.txt` - 结果摘要报告
6. 每种扰动类型和严重程度的详细结果文件（如 `gaussian_blur_s2_results.txt`）

## 示例结果

泛化测试可以帮助我们了解模型在不同扰动条件下的鲁棒性。例如，以下是一个典型的结果示例：

```
基准准确率 (无扰动): 85.2%

各扰动类型的平均准确率:
  gaussian_blur: 78.5% (下降 7.9%)
  rain: 76.3% (下降 10.4%)
  fog: 72.1% (下降 15.4%)
  gaussian_noise: 68.7% (下降 19.4%)
  contrast: 80.1% (下降 6.0%)

最显著的性能下降:
  gaussian_noise (严重程度 4): 62.3%
  相对于基准下降: 26.9%

最稳健的扰动类型 (平均准确率最高):
  contrast: 80.1% (下降 6.0%)
```

这些结果可以帮助我们:
1. 了解模型对哪些扰动类型更敏感
2. 确定模型的弱点，为进一步改进提供方向
3. 与其他模型进行对比，评估泛化能力的差异 