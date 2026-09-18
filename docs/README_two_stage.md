# 两阶段训练策略使用说明

# 经过对比测试，该策略在本项目中**未带来稳定的性能提升，已从主实验中停用**，仅保留代码和说明供研究参考。

## 概述

两阶段训练策略将训练过程分为两个阶段：

```
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 1: 稳定学习表征                         │
│                    (Epoch 1 → STAGE1_EPOCHS)                    │
├─────────────────────────────────────────────────────────────────┤
│  目标：学习通用的特征表示                                        │
│  损失：HybridLoss (BCE + Focal + LabelSmooth)                   │
│  学习率：正常 (LR = 1e-4)                                        │
│  数据增强：Mixup 启用                                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 Stage 2: 指标对齐微调                            │
│               (Epoch STAGE1_EPOCHS+1 → NUM_EPOCHS)              │
├─────────────────────────────────────────────────────────────────┤
│  目标：优化具体评测指标（F1、AUC）                               │
│  损失：ASL / LogitAdjusted / Focal                              │
│  学习率：降低 (LR × 0.1)                                         │
│  数据增强：Mixup 禁用（更精细调整）                              │
└─────────────────────────────────────────────────────────────────┘
```

## 设计原理

### Stage 1：稳定学习表征

- **HybridLoss**：BCE 提供稳定梯度，Focal 关注难分样本，LabelSmooth 防止过拟合
- **正常学习率**：快速学习通用特征
- **Mixup**：增强泛化能力，学习更鲁棒的表示

### Stage 2：指标对齐微调

- **ASL (Asymmetric Loss)**：对负样本更激进的惩罚，优化召回率
- **低学习率**：精细调整，避免破坏已学到的表征
- **禁用 Mixup**：更精确地优化决策边界

## 配置说明

### scripts/train.py / train_timm_models.py 中的配置

```python
# ================== 两阶段训练策略配置 ==================
USE_TWO_STAGE = False         # 是否启用两阶段训练（默认关闭）
STAGE1_EPOCHS = 15            # Stage 1 的 epoch 数
STAGE2_LOSS = "asl"           # Stage 2 损失函数: asl, la, focal
STAGE2_LR_FACTOR = 0.1        # Stage 2 学习率倍数
STAGE2_DISABLE_MIXUP = True   # Stage 2 是否禁用 Mixup
STAGE2_RESET_EMA = False      # Stage 2 是否重置 EMA
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `USE_TWO_STAGE` | False | 总开关，当前项目默认关闭 |
| `STAGE1_EPOCHS` | 15 | Stage 1 epoch 数（建议总 epoch 的 40%–60%） |
| `STAGE2_LOSS` | "asl" | Stage 2 损失函数 |
| `STAGE2_LR_FACTOR` | 0.1 | Stage 2 学习率 = LR × 此值 |
| `STAGE2_DISABLE_MIXUP` | True | Stage 2 禁用 Mixup |
| `STAGE2_RESET_EMA` | False | 是否重置 EMA |

### Stage 2 损失函数选项

| 选项 | 说明 | 适用场景 |
|------|------|----------|
| `asl` | Asymmetric Loss | 默认推荐，对负样本更激进 |
| `la` | Logit Adjusted Loss | 类别极度不平衡时 |
| `focal` | Focal Loss | 难分样本较多时 |

## 使用方式

### 方式 1：修改配置后运行（仅供自行实验）

```python
# train.py 中设置
USE_TWO_STAGE = True
STAGE1_EPOCHS = 15
STAGE2_LOSS = "asl"
```

然后正常运行：

```bash
python train.py
```

> 当前代码未实现命令行参数形式的两阶段开关，需通过修改脚本内的配置变量启用。

## 训练日志示例（示意）

```
[xrv-chex] Epoch 15 [Stage 1]:
  Train Loss: 0.2345
  Val Macro-F1: 0.4123
  Current LR: 0.000100

============================================================
[xrv-chex] 切换到 Stage 2 (指标对齐微调)
============================================================
  [Stage 2 Loss] ASL (gamma_neg=4.0, clip=0.05)
  学习率: 0.0001 -> 1e-05
  早停计数器已重置
============================================================

[xrv-chex] Epoch 16 [Stage 2]:
  Train Loss: 0.1876
  Val Macro-F1: 0.4234
  Current LR: 0.000010
```

## 实验结果与结论

在本项目的数据规模和当前模型配置下，我们对两阶段训练策略进行了多轮实验，观察到：

- 在个别配置下，指标存在小幅波动，但**未能稳定、显著地优于单阶段训练**；
- 引入了额外的超参数（如切换 epoch、第二阶段损失类型和学习率因子），增加了调参复杂度。

因此，最终在主实验中**不再启用两阶段训练**，而是采用简化的一阶段训练流程。本说明文档主要保留该策略的设计思路和接口，方便日后如需在其他数据集或更大规模实验中继续探索。

## 与其他策略配合

### 与 Attention-Guided Crop 配合

两阶段训练可以与 AG-Crop 同时启用：

```python
USE_ATTENTION_CROP = True
USE_TWO_STAGE = True
```

### 与集成学习配合

每个集成模型都会独立进行两阶段训练，最后再进行集成：

```
Model 1: Stage 1 → Stage 2 → Best checkpoint
Model 2: Stage 1 → Stage 2 → Best checkpoint
Model 3: Stage 1 → Stage 2 → Best checkpoint
...
Ensemble: 加权平均
```

## 调参建议

### Stage 1 Epoch 数

- **太少**（<10）：表征学习不充分
- **太多**（>20）：可能过拟合，Stage 2 调整空间小
- **推荐**：总 epoch 的 40%-60%

### Stage 2 学习率

- **太高**（>0.2）：可能破坏已学表征
- **太低**（<0.01）：调整效果不明显
- **推荐**：0.05-0.2 倍

### 是否重置 EMA

- **重置**：Stage 2 从当前模型重新开始 EMA，适合 Stage 1 后期震荡较大时
- **不重置**（默认）：保持平滑，更稳定

## 常见问题

### Q: Stage 2 后指标反而下降？

A: 可能原因：
1. Stage 2 学习率太高 → 降低 `STAGE2_LR_FACTOR`
2. Stage 1 epoch 太少 → 增加 `STAGE1_EPOCHS`
3. ASL 太激进 → 尝试 `focal` 或调整 gamma

### Q: 两阶段没有明显提升？

A: 可能原因：
1. Stage 1 已经过拟合 → 减少 `STAGE1_EPOCHS`
2. 数据集较小 → 两阶段优势不明显
3. 尝试不同的 `STAGE2_LOSS`

### Q: 如何确定最佳 Stage 1 Epoch 数？

A: 观察 Stage 1 的验证指标：
1. 当验证 loss 开始上升或 F1 趋于平稳时，是切换的好时机
2. 可以先用单阶段训练观察曲线，确定切换点
