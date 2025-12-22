# cxr_config.py

import os
import torch

NIH_DATA_ROOT = r"C:\Users\libin\Desktop\NIH_DATA_ROOT"

IMAGES_DIR = os.path.join(NIH_DATA_ROOT, "images")
LABEL_CSV = os.path.join(NIH_DATA_ROOT, "filtered_labels.csv")

# 模型保存目录
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

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


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42
VAL_RATIO = 0.1   # 从 21844 张里再划 10% 做验证


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
TWO_STAGE_ENABLED = True       # 是否启用两阶段训练
TWO_STAGE_STAGE1_EPOCHS = 15   # Stage 1 的 epoch 数
TWO_STAGE_STAGE2_LOSS = "asl"  # Stage 2 损失函数: asl, la, focal
TWO_STAGE_LR_FACTOR = 0.1      # Stage 2 学习率相对于 Stage 1 的倍数
TWO_STAGE_DISABLE_MIXUP = True # Stage 2 是否禁用 Mixup
