# ChestX-ray14 多标签分类项目

基于深度学习的胸部 X 光多标签分类系统，使用 NIH ChestX-ray14 数据集进行 14 种疾病的多标签分类。

## 📋 项目概述

本项目实现了多种先进的深度学习策略，用于胸部 X 光图像的多标签疾病分类：

- **多模型集成学习**：支持 torchxrayvision 预训练模型和 timm 模型
- **Attention-Guided Crop**：基于注意力机制的高分辨率局部特征提取
- **两阶段训练策略**：分阶段优化表征学习和指标对齐
- **多种损失函数**：BCE、Focal Loss、Asymmetric Loss、Logit Adjusted Loss 等

### 支持的疾病类别（14类）

| 疾病名称 | 中文名称 |
|----------|----------|
| Atelectasis | 肺不张 |
| Cardiomegaly | 心脏肥大 |
| Effusion | 胸腔积液 |
| Infiltration | 浸润 |
| Mass | 肿块 |
| Nodule | 结节 |
| Pneumonia | 肺炎 |
| Pneumothorax | 气胸 |
| Consolidation | 实变 |
| Edema | 肺水肿 |
| Emphysema | 肺气肿 |
| Fibrosis | 纤维化 |
| Pleural_Thickening | 胸膜增厚 |
| Hernia | 膈疝 |

## 🏗️ 项目结构

```
chestxray_multilabel/
├── train.py                    # XRV 模型集成训练（DenseNet121）
├── train_timm_models.py        # timm 模型训练（ConvNeXt, ViT, EfficientNet）
├── train_attention_crop.py     # Attention-Guided Crop 便捷入口
│
├── models.py                   # 模型定义（DenseNet, torchxrayvision）
├── models_timm.py              # timm 模型包装器
├── models_attention_crop.py    # Attention-Guided Crop 模块
│
├── dataset.py                  # 数据集类（标准/多分辨率）
├── loss_utils.py               # 损失函数（Focal, ASL, HybridLoss 等）
├── metrics_utils.py            # 评估指标（AUC, F1, 阈值搜索）
│
├── ensemble_timm.py            # timm 模型集成评估
├── ensemble_search.py          # 集成权重搜索
├── ensemble_val.py             # 验证集集成评估
├── infer.py                    # 推理脚本
│
├── cxr_config.py               # 配置文件（超参数、路径等）
├── requirements.txt            # 依赖包列表
│
├── data/                       # 数据目录
│   └── filtered_labels.csv     # 标签文件
├── saved_models/               # 模型保存目录
└── docs/                       # 文档
    ├── README_attention_crop.md
    └── README_two_stage.md
```

## 🚀 快速开始

### 环境配置

```bash
# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 必需依赖

- Python >= 3.8
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- timm >= 0.9.0
- torchxrayvision >= 1.2.0
- scikit-learn >= 1.2.0
- pandas >= 2.0.0
- numpy >= 1.24.0

### 数据准备

1. 下载 NIH ChestX-ray14 数据集
2. 修改 `cxr_config.py` 中的数据路径：

```python
NIH_DATA_ROOT = r"C:\path\to\NIH_DATA_ROOT"
```

3. 确保目录结构如下：
```
NIH_DATA_ROOT/
├── images/          # 所有图像文件
└── filtered_labels.csv  # 标签文件
```

## 🎯 训练模型

### 方式 1：XRV 模型集成训练

使用 torchxrayvision 预训练的 DenseNet121 模型：

```bash
python train.py
```

支持的预训练权重（不含 NIH 数据，避免数据泄漏）：
- `densenet121-res224-chex`：CheXpert 数据集
- `densenet121-res224-pc`：PadChest 数据集  
- `densenet121-res224-mimic_nb`：MIMIC-CXR (NoBBox)
- `densenet121-res224-mimic_ch`：MIMIC-CXR (ChestOnly)

### 方式 2：timm 模型训练

使用 timm 库的现代模型架构：

```bash
python train_timm_models.py
```

支持的模型：
- **ConvNeXt**：convnext_base_in22k, convnext_base_in22k_384, convnext_base_in22k_512
- **ConvNeXtV2**：convnextv2_base_fcmae_384
- **EfficientNetV2**：tf_efficientnetv2_m

### 方式 3：Attention-Guided Crop

启用高分辨率局部特征提取：

```bash
python train_attention_crop.py
```

或在配置中启用：
```python
# train.py 中设置
USE_ATTENTION_CROP = True
AG_HIGH_RES = 512
AG_NUM_CROPS = 2
```

## ⚙️ 核心配置

### 训练超参数 (`cxr_config.py`)

```python
# DenseNet 系列
BATCH_SIZE = 48
NUM_EPOCHS = 30
LR = 1e-4
MIXUP_ALPHA = 0.2

# Timm 模型
TIMM_BATCH_SIZE = 8
TIMM_NUM_EPOCHS = 20
TIMM_LR = 1e-4
```

### 两阶段训练策略

```python
TWO_STAGE_ENABLED = True           # 启用两阶段训练
TWO_STAGE_STAGE1_EPOCHS = 15       # Stage 1 epoch 数
TWO_STAGE_STAGE2_LOSS = "asl"      # Stage 2 损失函数
TWO_STAGE_LR_FACTOR = 0.1          # Stage 2 学习率倍数
```

**训练流程**：
```
Stage 1 (稳定学习表征)       Stage 2 (指标对齐微调)
━━━━━━━━━━━━━━━━━━━━━━━      ━━━━━━━━━━━━━━━━━━━━━━━
• HybridLoss                • Asymmetric Loss (ASL)
• 正常学习率                 • 低学习率 (LR × 0.1)
• Mixup 启用                 • Mixup 禁用
```

### Attention-Guided Crop 配置

```python
AG_LOW_RES = 224           # 全局分支输入分辨率
AG_HIGH_RES = 512          # 高分辨率图
AG_NUM_CROPS = 2           # 裁剪区域数量
AG_CROP_SIZE = 224         # 裁剪后尺寸
AG_FUSION_TYPE = "concat_attention"  # 融合方式
```

## 📊 模型集成

### 运行集成评估

```bash
python ensemble_timm.py
```

### 支持的集成策略

1. **简单平均**：所有模型预测概率的平均值
2. **加权平均**：网格搜索最佳权重组合
3. **分层加权**：高/低分辨率模型分组加权
4. **Logistic Stacking**：使用交叉验证训练元分类器
5. **Per-class 自适应权重**：根据各模型在每个类别上的 AUC 分配权重

## 📈 评估指标

- **Macro-AUC**：各类别 AUC 的宏平均
- **Macro-F1**：各类别 F1 的宏平均
- **Weighted-F1**：加权 F1 分数
- **Per-class 阈值优化**：针对每个类别独立搜索最优阈值

## 🔧 损失函数

| 损失函数 | 说明 | 适用场景 |
|----------|------|----------|
| `BCEWithLogitsLoss` | 标准二元交叉熵 | 基础训练 |
| `HybridLoss` | BCE + Focal + LabelSmooth | Stage 1 稳定学习 |
| `AsymmetricLoss` | 正负样本不对称惩罚 | Stage 2 优化召回 |
| `MultiLabelFocalLoss` | 聚焦难分样本 | 类别不平衡 |
| `LogitAdjustedLoss` | Logit 调整 | 极度不平衡 |

## 📁 输出文件

训练完成后，模型和预测结果保存在 `saved_models/` 目录：

```
saved_models/
├── xrv-chex_best.pt           # XRV 模型权重
├── xrv-pc_best.pt
├── convnext_base_in22k_best.pt # timm 模型权重
├── convnext_base_in22k_384_best.pt
│
├── xrv-chex_val_preds.npz     # 验证集预测
├── xrv-chex_val_labels.npz    # 验证集标签
└── ...
```

## 📚 更多文档

- [Attention-Guided Crop 使用说明](docs/README_attention_crop.md)
- [两阶段训练策略说明](docs/README_two_stage.md)

## 🔗 参考资料

- [NIH ChestX-ray14 数据集](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- [torchxrayvision](https://github.com/mlmed/torchxrayvision)
- [timm](https://github.com/huggingface/pytorch-image-models)
- [Asymmetric Loss (CVPR 2021)](https://arxiv.org/abs/2009.14119)

## 📄 许可证

本项目仅用于学术研究目的。
