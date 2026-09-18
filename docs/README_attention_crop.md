# Attention-Guided Crop 使用说明

# 经过对比测试，该策略在本项目中**未带来稳定的性能提升，已从主实验中停用**，仅保留代码和说明供研究参考。

## 概述

Attention-Guided Crop 是一个**可插拔的增强模块**，已经集成到现有的 `scripts/train.py`（XRV 模型）和 `train_timm_models.py`（timm 模型）中。

在当前代码中，默认配置为 **关闭**（`USE_ATTENTION_CROP = False`），不会参与正式实验。如需自行探索，可以通过简单的配置开关启用此功能，**无需使用单独的训练脚本**。

## 核心思路

```
输入图像
    │
    ├─→ 低分辨率 (224×224) ─→ 全局分支 ─→ 全局特征 + Attention Map
    │                                              │
    │                                              ▼
    └─→ 高分辨率 (512×512) ─────────────→ Attention-Guided Crop
                                                   │
                                                   ▼
                                        局部 Patches → 局部特征
                                                   │
                    ┌──────────────────────────────┴──────────────────────────────┐
                    │                                                              │
               全局特征 ─────────────→ Attention-Weighted Fusion ←───────── 局部特征
                                              │
                                              ▼
                                         分类头 (14类)
```

## 快速使用

### 方式 1：修改配置（推荐，仅供自行实验）

在 `scripts/train.py` 中找到配置区域，设置：

```python
# ================== Attention-Guided Crop 配置 ==================
USE_ATTENTION_CROP = True     # 启用 Attention-Guided Crop
AG_HIGH_RES = 512             # 高分辨率图像尺寸
AG_NUM_CROPS = 2              # 裁剪的局部区域数量
AG_CROP_SIZE = 224            # 裁剪后的尺寸
AG_FUSION_TYPE = "concat_attention"  # 融合方式
AG_SHARE_BACKBONE = True      # 是否共享 backbone
AG_BATCH_SIZE = 12            # batch size（需要更小）
```

然后正常运行：

```bash
python train.py
```

### 方式 2：使用便捷入口（已停用，仅供参考）

```bash
python train_attention_crop.py
```

这会自动设置 `USE_ATTENTION_CROP = True` 并运行训练。

### 方式 3：在代码中包装现有模型（研究用示例）

```python
from src.models import TorchXRayVisionDenseNet
from src.models_attention_crop import wrap_model_with_attention_crop

# 创建基础模型
base_model = TorchXRayVisionDenseNet(num_classes=14, weights="densenet121-res224-chex")

# 包装成 Attention-Guided 版本
ag_model = wrap_model_with_attention_crop(
    base_model,
    backbone_type="xrv",
    num_classes=14,
    feature_dim=1024,
    num_crops=2,
    high_res_size=512,
)

# 训练时传入双分辨率图像
logits = ag_model(x_low, x_high)  # x_low: 224x224, x_high: 512x512
```

## 配置说明

### scripts/train.py (XRV 模型)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `USE_ATTENTION_CROP` | False | 是否启用 |
| `AG_HIGH_RES` | 512 | 高分辨率尺寸 |
| `AG_NUM_CROPS` | 2 | 裁剪区域数 |
| `AG_CROP_SIZE` | 224 | 裁剪尺寸 |
| `AG_FUSION_TYPE` | concat_attention | 融合方式 |
| `AG_SHARE_BACKBONE` | True | 共享backbone |
| `AG_BATCH_SIZE` | 12 | batch size |

### train_timm_models.py (timm 模型)

同上配置项。注意：仅支持在 `TIMM_FEATURE_DIMS` 中已定义的 CNN 类模型（如 ConvNeXt 等），**不支持 ViT**。

## 融合方式

### `concat_attention`（推荐）

使用可学习的注意力权重融合：

```
weights = softmax(attention_net([global, local_1, local_2]))
fused = weights[0] * global + weights[1] * local_1 + weights[2] * local_2
```

### `concat`

简单拼接后全连接：

```
fused = FC([global, local_1, local_2])
```

### `add`

投影后相加：

```
fused = proj_global(global) + mean(proj_local([local_1, local_2]))
```

## 显存需求（基于早期实验）

| 模式 | Batch Size | 显存需求 |
|------|------------|----------|
| 标准 (224) | 48 | ~8 GB |
| AG-Crop | 12 | ~10 GB |
| AG-Crop | 8 | ~7 GB |
| AG-Crop (share_backbone) | 12 | ~8 GB |

如果显存不足（仅在自行开启 AG-Crop 实验时需要考虑）：
1. 减小 `AG_BATCH_SIZE`
2. 设置 `AG_SHARE_BACKBONE = True`
3. 减小 `AG_HIGH_RES`（如 448）

## 实验结果与结论

在本项目的数据规模和模型配置下，我们对 Attention-Guided Crop 进行了多轮对比实验，发现：

- 训练时间和显存开销增加约 2–2.5 倍；
- 对整体 AUC / F1 指标**没有带来稳定、显著的提升**，个别类别的波动也缺乏一致性。

因此，最终在主实验中**选择直接使用高分辨率单分支模型（例如 384/512 分辨率的 ConvNeXt 等）**，并将 AG-Crop 标记为**已停用的探索性策略**。

## 文件结构

```
chestxray_multilabel/
├── train.py                    # 主训练脚本（已集成 AG-Crop 支持）
├── train_timm_models.py        # timm 训练脚本（已集成 AG-Crop 支持）
├── train_attention_crop.py     # 便捷入口（可选）
├── models_attention_crop.py    # AG-Crop 核心模块
│   ├── AttentionGuidedWrapper      # 通用包装器
│   ├── wrap_model_with_attention_crop()  # 便捷函数
│   └── visualize_attention()       # 可视化工具
└── dataset.py                  # 数据集（含多分辨率支持）
    ├── ChestXrayMultiResDatasetV2  # 多分辨率数据集
    └── ...
```

## 常见问题

### Q: 训练速度变慢了？

A: 正常现象，高分辨率处理和多次特征提取会增加计算量。早期实验中，训练时间大约是标准训练的 1.5–2 倍。

### Q: 显存不足？

A: 
1. 减小 `AG_BATCH_SIZE`（推荐 8-12）
2. 设置 `AG_SHARE_BACKBONE = True`
3. 减小 `AG_HIGH_RES`

### Q: 如何只对特定模型启用？

A: 在 `train.py` 的 `train_single_model` 函数中，可以根据 `model_name` 判断是否启用：

```python
# 只对 chex 模型启用
if model_name == "xrv-chex":
    USE_ATTENTION_CROP = True
```

### Q: 如何可视化注意力图？

A: 使用 `visualize_attention` 函数：

```python
from src.models_attention_crop import visualize_attention

logits, attention_map = model(x_low, x_high, return_attention=True)
visualize_attention(x_low[0], attention_map[0], save_path="attention.png")
```
