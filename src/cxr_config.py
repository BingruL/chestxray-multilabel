# cxr_config.py

import os
import torch

# 获取当前配置文件的目录 (src/)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根目录 (chestxray_multilabel/)
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)

NIH_DATA_ROOT = r"C:\Users\libin\Desktop\NIH_DATA_ROOT"

IMAGES_DIR = os.path.join(NIH_DATA_ROOT, "images")
LABEL_CSV = os.path.join(NIH_DATA_ROOT, "filtered_labels.csv")

# 模型保存目录（使用绝对路径）
SAVE_DIR = os.path.join(_PROJECT_ROOT, "saved_models")
os.makedirs(SAVE_DIR, exist_ok=True)

# 日志保存目录（使用绝对路径）
LOGS_DIR = os.path.join(_PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# ====== 类别信息（ChestX-ray14 的 14 类） ======
CLASS_NAMES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]
NUM_CLASSES = len(CLASS_NAMES)

# ====== train.py 训练超参数（DenseNet 系列） ======
BATCH_SIZE = 48
NUM_EPOCHS = 30
WARMUP_EPOCHS = 2
LR = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
MIXUP_ALPHA = 0.2
USE_EMA = True
EARLY_STOP = True
EARLY_STOP_PATIENCE = 3
EARLY_STOP_MIN_DELTA = 0.0

# ====== train_timm_models.py 训练超参数（Timm 模型） ======
TIMM_BATCH_SIZE = 8
TIMM_NUM_EPOCHS = 20
TIMM_WARMUP_EPOCHS = 2
TIMM_LR = 1e-4
TIMM_WEIGHT_DECAY = 1e-4
TIMM_GRAD_CLIP_NORM = 1.0
TIMM_MIXUP_ALPHA = 0.1
TIMM_USE_EMA = True
TIMM_EARLY_STOP = True
TIMM_EARLY_STOP_PATIENCE = 3
TIMM_EARLY_STOP_MIN_DELTA = 0.0

# ------ 架构特定超参数配置（已禁用，经测试统一配置性能更好）------
# 如需启用，将 TIMM_USE_ARCH_SPECIFIC_CONFIG 设为 True
TIMM_USE_ARCH_SPECIFIC_CONFIG = False

# # 默认配置（作为 fallback）
# TIMM_DEFAULT_CONFIG = {
#     "lr": 1e-4,
#     "weight_decay": 5e-2,         # 提升到 0.05（官方推荐）
#     "warmup_epochs": 2,
#     "mixup_alpha": 0.2,
#     "layer_decay": None,          # 不使用层级学习率衰减
# }

# # ConvNeXt 系列配置（现代 CNN，训练稳定）
# TIMM_CONVNEXT_CONFIG = {
#     "lr": 1e-4,                   # 官方推荐
#     "weight_decay": 5e-2,         # 官方推荐 0.05
#     "warmup_epochs": 2,           # 足够
#     "mixup_alpha": 0.3,           # 适度增强
#     "layer_decay": 0.85,          # 层级学习率衰减
# }

# # Swin Transformer 系列配置（对超参数敏感）
# TIMM_SWIN_CONFIG = {
#     "lr": 5e-5,                   # 比 CNN 小一半
#     "weight_decay": 5e-2,         # 官方推荐
#     "warmup_epochs": 5,           # 更长的 warmup
#     "mixup_alpha": 0.4,           # 更强的数据增强
#     "layer_decay": 0.75,          # 更强的层级衰减
# }

# # CoaT 混合架构配置（介于 CNN 和 Transformer 之间）
# TIMM_COAT_CONFIG = {
#     "lr": 8e-5,                   # 略低于纯 CNN
#     "weight_decay": 3e-2,         # 中等正则化
#     "warmup_epochs": 3,           # 中等 warmup
#     "mixup_alpha": 0.2,           # 适度增强
#     "layer_decay": 0.8,           # 中等层级衰减
# }


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# 数据集划分比例（按病人划分）
TRAIN_RATIO = 0.8   # 训练集比例
VAL_RATIO = 0.1     # 验证集比例
TEST_RATIO = 0.1    # 测试集比例

# 稀有类别列表（样本数 < 200 的类别，需要分层抽样保证每个集合都有样本）
RARE_CLASSES = ["Hernia", "Pneumonia", "Fibrosis", "Edema", "Emphysema"]


# ====== Attention-Guided Crop 配置 ======
# 分辨率设置
AG_LOW_RES = 224           # 全局分支输入分辨率
AG_HIGH_RES = 512          # 高分辨率图（用于裁剪局部区域）

# 裁剪设置
AG_NUM_CROPS = 2           # 裁剪的局部区域数量
AG_CROP_SIZE = 224         # 裁剪后 resize 到的尺寸

# 融合设置
AG_FUSION_TYPE = "concat_attention"  # concat, add, concat_attention
AG_FUSION_DIM = 512        # 融合后的特征维度

# 训练设置（因为高分辨率需要更多显存）
AG_BATCH_SIZE = 16         # 减小 batch size
AG_USE_SIMPLE_MODEL = False  # 是否使用简化版模型

# 小病灶类别（高分辨率策略主要针对这些类别）
SMALL_LESION_CLASSES = ["Nodule", "Mass", "Pneumothorax"]

# ====== 两阶段训练配置 ======
# Stage 1: 稳定学习表征（使用温和的 HybridLoss，正常学习率）
# Stage 2: 指标对齐微调（切换到 ASL，降低学习率）
TWO_STAGE_ENABLED = False      # 是否启用两阶段训练
TWO_STAGE_STAGE1_EPOCHS = 15   # Stage 1 的 epoch 数
TWO_STAGE_STAGE2_LOSS = "asl"  # Stage 2 损失函数: asl, la, focal
TWO_STAGE_LR_FACTOR = 0.1      # Stage 2 学习率相对于 Stage 1 的倍数
TWO_STAGE_DISABLE_MIXUP = True # Stage 2 是否禁用 Mixup
