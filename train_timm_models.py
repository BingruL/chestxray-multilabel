# train_timm_models.py
"""
使用 timm 训练多个 backbone：
- ConvNeXt (ImageNet-1K / 21K / 21K-384)
- Vision Transformer
- EfficientNet
并在验证集上保存预测，用于后续集成。
"""

import os
import random
from copy import deepcopy

# 设置 Hugging Face 镜像站，解决国内下载权重慢或失败的问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import pandas as pd
from tqdm import tqdm

# AMP
from torch import amp

autocast = amp.autocast
GradScaler = amp.GradScaler


import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from src.cxr_config import (
    LABEL_CSV, IMAGES_DIR, SAVE_DIR,
    DEVICE, RANDOM_SEED, VAL_RATIO, CLASS_NAMES,
    # Timm 模型专用配置
    TIMM_BATCH_SIZE as BATCH_SIZE,
    TIMM_NUM_EPOCHS as NUM_EPOCHS,
    TIMM_WARMUP_EPOCHS as WARMUP_EPOCHS,
    TIMM_LR as LR,
    TIMM_WEIGHT_DECAY as WEIGHT_DECAY,
    TIMM_MIXUP_ALPHA as MIXUP_ALPHA,
    TIMM_GRAD_CLIP_NORM as GRAD_CLIP_NORM,
    TIMM_USE_EMA as USE_EMA,
    TIMM_EARLY_STOP as EARLY_STOP,
    TIMM_EARLY_STOP_PATIENCE as EARLY_STOP_PATIENCE,
    TIMM_EARLY_STOP_MIN_DELTA as EARLY_STOP_MIN_DELTA,
    # 架构特定超参数配置（已禁用）
    # TIMM_USE_ARCH_SPECIFIC_CONFIG,
    # TIMM_DEFAULT_CONFIG,
    # TIMM_CONVNEXT_CONFIG,
    # TIMM_SWIN_CONFIG,
    # TIMM_COAT_CONFIG,
)
from src.dataset import ChestXrayDataset, ChestXrayMultiResDatasetV2
from src.metrics_utils import compute_metrics, search_best_thresholds
from src.models_timm import TimmMultiLabelModel
from src.models_attention_crop import wrap_model_with_attention_crop
from src.loss_utils import (
    compute_pos_weight,
    HybridLoss,
    AsymmetricLoss,
    LogitAdjustedLoss,
    get_logit_adjustment,
)
from src.log_utils import setup_logger, close_logger



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 架构特定超参数和 Layer-wise LR Decay（已禁用，经测试统一配置性能更好）
# 如需启用，取消下面的注释并在 cxr_config.py 中设置 TIMM_USE_ARCH_SPECIFIC_CONFIG = True
# ============================================================

# def get_arch_config(model_name: str) -> dict:
#     """
#     根据模型名称返回架构特定的超参数配置。
#     
#     Args:
#         model_name: timm 模型名称（如 "convnext_base_in22k_512"）
#     
#     Returns:
#         包含 lr, weight_decay, warmup_epochs, mixup_alpha, layer_decay 的字典
#     """
#     if not TIMM_USE_ARCH_SPECIFIC_CONFIG:
#         return TIMM_DEFAULT_CONFIG.copy()
#     
#     model_name_lower = model_name.lower()
#     
#     if "convnext" in model_name_lower:
#         config = TIMM_CONVNEXT_CONFIG.copy()
#         arch_name = "ConvNeXt"
#     elif "swin" in model_name_lower:
#         config = TIMM_SWIN_CONFIG.copy()
#         arch_name = "Swin"
#     elif "coat" in model_name_lower:
#         config = TIMM_COAT_CONFIG.copy()
#         arch_name = "CoaT"
#     else:
#         config = TIMM_DEFAULT_CONFIG.copy()
#         arch_name = "Default"
#     
#     print(f"  >> 使用 {arch_name} 架构配置:")
#     print(f"     LR={config['lr']}, WD={config['weight_decay']}, "
#           f"Warmup={config['warmup_epochs']}, Mixup={config['mixup_alpha']}, "
#           f"LayerDecay={config['layer_decay']}")
#     
#     return config


# def get_layer_id_for_convnext(name: str, num_layers: int = 12) -> int:
#     """
#     获取 ConvNeXt 模型参数所属的层 ID。
#     ConvNeXt 结构：stem -> stages[0-3] -> head
#     """
#     if name.startswith("backbone."):
#         name = name[len("backbone."):]
#     
#     if "stem" in name or "downsample_layers.0" in name:
#         return 0
#     elif "stages.0" in name or "downsample_layers.1" in name:
#         return 1
#     elif "stages.1" in name or "downsample_layers.2" in name:
#         return 2
#     elif "stages.2" in name or "downsample_layers.3" in name:
#         return 3
#     elif "stages.3" in name:
#         return 4
#     elif "head" in name or "norm" in name or "classifier" in name:
#         return num_layers  # 顶层
#     else:
#         return num_layers  # 未知层归为顶层


# def get_layer_id_for_swin(name: str, num_layers: int = 12) -> int:
#     """
#     获取 Swin Transformer 模型参数所属的层 ID。
#     Swin 结构：patch_embed -> layers[0-3] -> head
#     """
#     if name.startswith("backbone."):
#         name = name[len("backbone."):]
#     
#     if "patch_embed" in name:
#         return 0
#     elif "layers.0" in name:
#         return 1
#     elif "layers.1" in name:
#         return 2
#     elif "layers.2" in name:
#         return 3
#     elif "layers.3" in name:
#         return 4
#     elif "head" in name or "norm" in name or "classifier" in name:
#         return num_layers
#     else:
#         return num_layers


# def get_layer_id_for_coat(name: str, num_layers: int = 12) -> int:
#     """
#     获取 CoaT 模型参数所属的层 ID。
#     CoaT 结构：patch_embed -> serial_blocks -> parallel_blocks -> head
#     """
#     if name.startswith("backbone."):
#         name = name[len("backbone."):]
#     
#     if "patch_embed" in name:
#         return 0
#     elif "serial_blocks1" in name:
#         return 1
#     elif "serial_blocks2" in name:
#         return 2
#     elif "serial_blocks3" in name:
#         return 3
#     elif "serial_blocks4" in name:
#         return 4
#     elif "parallel" in name:
#         return 5
#     elif "head" in name or "norm" in name or "classifier" in name:
#         return num_layers
#     else:
#         return num_layers


# def get_layer_id(name: str, model_name: str, num_layers: int = 12) -> int:
#     """
#     根据模型架构获取参数所属的层 ID。
#     """
#     model_name_lower = model_name.lower()
#     
#     if "convnext" in model_name_lower:
#         return get_layer_id_for_convnext(name, num_layers)
#     elif "swin" in model_name_lower:
#         return get_layer_id_for_swin(name, num_layers)
#     elif "coat" in model_name_lower:
#         return get_layer_id_for_coat(name, num_layers)
#     else:
#         # 默认：所有层使用相同学习率
#         return num_layers


# def build_optimizer_with_layer_decay(
#     model: torch.nn.Module,
#     model_name: str,
#     base_lr: float,
#     weight_decay: float,
#     layer_decay: float | None = None,
#     num_layers: int = 6,
# ) -> torch.optim.Optimizer:
#     """
#     构建带有 Layer-wise LR Decay 的 AdamW 优化器。
#     
#     Args:
#         model: PyTorch 模型
#         model_name: 模型名称（用于识别架构）
#         base_lr: 基础学习率（顶层使用）
#         weight_decay: 权重衰减
#         layer_decay: 层级学习率衰减因子（如 0.75），None 表示不使用
#         num_layers: 模型总层数（用于计算衰减）
#     
#     Returns:
#         配置好的 AdamW 优化器
#     """
#     if layer_decay is None or layer_decay >= 1.0:
#         # 不使用 Layer Decay，返回普通优化器
#         print(f"  >> 优化器: AdamW (lr={base_lr}, wd={weight_decay}, 无 Layer Decay)")
#         return optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
#     
#     # 使用 Layer-wise LR Decay
#     param_groups = []
#     param_group_names = {}  # 用于调试
#     
#     for name, param in model.named_parameters():
#         if not param.requires_grad:
#             continue
#         
#         # 获取层 ID
#         layer_id = get_layer_id(name, model_name, num_layers)
#         
#         # 计算该层的学习率缩放因子
#         # 底层（layer_id 小）使用更小的学习率
#         lr_scale = layer_decay ** (num_layers - layer_id)
#         lr = base_lr * lr_scale
#         
#         # bias 和 norm 层不使用 weight decay
#         if "bias" in name or "norm" in name or "bn" in name:
#             wd = 0.0
#         else:
#             wd = weight_decay
#         
#         param_groups.append({
#             "params": [param],
#             "lr": lr,
#             "weight_decay": wd,
#             "name": name,  # 用于调试
#         })
#         
#         # 记录每层的学习率（用于打印）
#         if layer_id not in param_group_names:
#             param_group_names[layer_id] = {"lr": lr, "count": 0}
#         param_group_names[layer_id]["count"] += 1
#     
#     # 打印 Layer Decay 信息
#     print(f"  >> 优化器: AdamW with Layer-wise LR Decay (layer_decay={layer_decay})")
#     print(f"     层级学习率分布:")
#     for layer_id in sorted(param_group_names.keys()):
#         info = param_group_names[layer_id]
#         print(f"       Layer {layer_id}: lr={info['lr']:.2e}, params={info['count']}")
#     
#     return optim.AdamW(param_groups)


def get_transforms_for_size(
    img_size: int,
    train: bool = True,
    use_jitter: bool = True,
    sharpness: float = 0.0,
):
    """根据输入尺寸构造数据增强/预处理（不改 dataset.py，单独写一个版本）"""
    import torchvision.transforms as T

    color_blocks = []
    if use_jitter:
        color_blocks.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02))
        color_blocks.append(T.RandomAutocontrast())
    if sharpness > 0:
        color_blocks.append(T.RandomAdjustSharpness(sharpness_factor=1.0 + sharpness, p=0.5))

    if train:
        return T.Compose([
            T.Resize(int(img_size * 1.14)),            # 先放大一点
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            *color_blocks,
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.25, 0.25, 0.25]),
        ])
    else:
        return T.Compose([
            T.Resize(int(img_size * 1.14)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.25, 0.25, 0.25]),
        ])


# ===== 定义要训练的模型列表 =====
MODEL_CONFIGS = [
    # -------- ConvNeXt 系列 --------
    # 注意：新版 timm 使用 model_name.pretrained_tag 格式
    {
        "name": "convnext_base_in1k",
        "backbone": "convnext_base.fb_in1k",      # ImageNet-1K
        "img_size": 224,
    },
    {
        "name": "convnext_base_in22k",
        "backbone": "convnext_base.fb_in22k",   # ImageNet-22K
        "img_size": 224,
    },
    {
        "name": "convnext_base_in22k_384",
        "backbone": "convnext_base.fb_in22k_ft_in1k_384",  # 22K pretrain + 384 finetune
        "img_size": 384,
    },
    {
        "name": "convnext_base_in22k_512",
        "backbone": "convnext_base.fb_in22k",  # 22K + 512 输入
        "img_size": 512,
    },
    {
        "name": "convnext_small_in22k_384",
        "backbone": "convnext_small.fb_in22k_ft_in1k_384",  # 50M 参数，384 分辨率
        "img_size": 384,
    },
    {
        "name": "convnext_small_in22k_512",
        "backbone": "convnext_small.fb_in22k_ft_in1k",  # 50M 参数，自定义 512 分辨率
        "img_size": 512,
    },
    {
        "name": "convnext_tiny_in22k_384",
        "backbone": "convnext_tiny.fb_in22k_ft_in1k_384",  # 28.6M 参数，384 分辨率
        "img_size": 384,
    },
    {
        "name": "convnext_large_in22k",
        "backbone": "convnext_large.fb_in22k_ft_in1k",  # 198M 参数
        "img_size": 224,
    },
    {
        "name": "convnext_large_in22k_384",
        "backbone": "convnext_large.fb_in22k_ft_in1k_384",  # 198M 参数，384 分辨率
        "img_size": 384,
    },
    {
        "name": "convnext_large_in22k_512",
        "backbone": "convnext_large.fb_in22k",  # 198M 参数，自定义 512 分辨率
        "img_size": 512,
    },
    {
        "name": "convnext_xlarge_in22k",
        "backbone": "convnext_xlarge.fb_in22k_ft_in1k",  # 350M 参数
        "img_size": 224,
    },
    {
        "name": "convnext_xlarge_in22k_384",
        "backbone": "convnext_xlarge.fb_in22k_ft_in1k_384",  # 350M 参数，384 分辨率
        "img_size": 384,
    },

    # -------- Vision Transformer 系列 --------
    {
        "name": "vit_base_in1k",
        "backbone": "vit_base_patch16_224.augreg_in1k",  # ImageNet-1K
        "img_size": 224,
    },
    {
        "name": "vit_base_in21k",
        "backbone": "vit_base_patch16_224.augreg_in21k",  # 21K
        "img_size": 224,
    },
    {
        "name": "vit_base_in21k_384",
        "backbone": "vit_base_patch16_384.augreg_in21k_ft_in1k",  # 21K pretrain + 384 finetune
        "img_size": 384,
    },
    {
        "name": "vit_base_in21k_512",
        "backbone": "vit_base_patch16_384.augreg_in21k_ft_in1k",  # 21K + 512 输入（patch16 可兼容）
        "img_size": 512,
    },

    # -------- EfficientNet 系列 --------
    {
        "name": "efficientnet_b0_in1k",
        "backbone": "efficientnet_b0.ra_in1k",   # ImageNet-1K
        "img_size": 224,
    },
    {
        "name": "efficientnet_b4_ns",
        "backbone": "tf_efficientnet_b4.ns_jft_in1k",  # Noisy Student (JFT pretrain)
        "img_size": 380,
    },
    {
        "name": "efficientnet_b4_ns_512",
        "backbone": "tf_efficientnet_b4.ns_jft_in1k",
        "img_size": 512,
    },
    # -------- EfficientNetV2 (新增策略) --------
    {
        "name": "effv2_s_in21k",
        "backbone": "tf_efficientnetv2_s.in21k",  # ImageNet-21k 预训练，泛化性更强
        "img_size": 384,                          # 推荐分辨率
    },

    # -------- Swin Transformer 系列 --------
    {
        "name": "swin_base_in22k_384",
        "backbone": "swin_base_patch4_window12_384.ms_in22k_ft_in1k",
        "img_size": 384,
    },
    {
        "name": "swin_large_in22k_384",
        "backbone": "swin_large_patch4_window12_384.ms_in22k_ft_in1k",
        "img_size": 384,
    },

    # -------- MaxViT 系列 --------
    {
        "name": "maxvit_base_in21k_384",
        "backbone": "maxvit_base_tf_384.in21k_ft_in1k",
        "img_size": 384,
    },
    {
        "name": "maxvit_base_in21k_512",
        "backbone": "maxvit_base_tf_512.in21k_ft_in1k",
        "img_size": 512,
    },

    # -------- ConvNeXt V2 --------
    {
        "name": "convnextv2_base_fcmae_384",
        "backbone": "convnextv2_base.fcmae_ft_in22k_in1k_384",
        "img_size": 384,
    },
    {
        "name": "convnextv2_base_in22k_512",
        "backbone": "convnextv2_base.fcmae_ft_in22k_in1k",  # 使用 22K→1K 权重 + 512 分辨率
        "img_size": 512,
    },
    {
        "name": "convnextv2_large_in22k_384",
        "backbone": "convnextv2_large.fcmae_ft_in22k_in1k_384",  # 198M 参数，384 分辨率
        "img_size": 384,
    },
    {
        "name": "convnextv2_tiny_in22k_384",
        "backbone": "convnextv2_tiny.fcmae_ft_in22k_in1k_384",  # 28.6M 参数，384 分辨率
        "img_size": 384,
    },
    {
        "name": "convnextv2_tiny_in22k_512",
        "backbone": "convnextv2_tiny.fcmae_ft_in22k_in1k",  # 28.6M 参数，自定义 512 分辨率
        "img_size": 512,
    },

    # -------- EVA-02 系列 --------
    {
        "name": "eva02_small_224",
        "backbone": "eva02_small_patch14_224.mim_in22k",
        "img_size": 224,
    },
    {
        "name": "eva02_small_336",
        "backbone": "eva02_small_patch14_336.mim_in22k_ft_in1k",
        "img_size": 336,
    },
    {
        "name": "eva02_base_224",
        "backbone": "eva02_base_patch14_224.mim_in22k",
        "img_size": 224,
    },
    {
        "name": "eva02_base_448",
        "backbone": "eva02_base_patch14_448.mim_in22k_ft_in1k",
        "img_size": 448,
    },
    {
        "name": "eva02_large_224",
        "backbone": "eva02_large_patch14_224.mim_in22k",
        "img_size": 224,
    },
    {
        "name": "eva02_large_448",
        "backbone": "eva02_large_patch14_448.mim_in22k_ft_in1k",
        "img_size": 448,
    },

    # -------- Swin Transformer V2 系列 --------
    {
        "name": "swinv2_base_in22k_256",
        "backbone": "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k",
        "img_size": 256,
    },
    {
        "name": "swinv2_base_in22k_384",
        "backbone": "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k",
        "img_size": 384,
    },
    {
        "name": "swinv2_large_in22k_256",
        "backbone": "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k",
        "img_size": 256,
    },
    {
        "name": "swinv2_large_in22k_384",
        "backbone": "swinv2_large_window12to24_192to384.ms_in22k_ft_in1k",
        "img_size": 384,
    },
    {
        "name": "swinv2_small_in1k_256",
        "backbone": "swinv2_small_window16_256.ms_in1k",
        "img_size": 256,
    },

    # -------- Swin S3 (AutoFormerV2) 系列 --------
    {
        "name": "swin_s3_base_224",
        "backbone": "swin_s3_base_224.ms_in1k",
        "img_size": 224,
    },
    {
        "name": "swin_s3_small_224",
        "backbone": "swin_s3_small_224.ms_in1k",
        "img_size": 224,
    },
    {
        "name": "swin_s3_tiny_224",
        "backbone": "swin_s3_tiny_224.ms_in1k",
        "img_size": 224,
    },

    # -------- CoAtNet 系列 (ImageNet-12k 预训练) --------
    {
        "name": "coatnet_2_rw_in12k",
        "backbone": "coatnet_2_rw_224.sw_in12k",
        "img_size": 224,
    },
    {
        "name": "coatnet_2_rw_in12k_ft",              #AUC=0.7862，F1很低
        "backbone": "coatnet_2_rw_224.sw_in12k_ft_in1k",
        "img_size": 224,
    },
    {
        "name": "coatnet_3_rw_in12k",
        "backbone": "coatnet_3_rw_224.sw_in12k",
        "img_size": 224,
    },
    {
        "name": "coatnet_rmlp_1_rw2_in12k",
        "backbone": "coatnet_rmlp_1_rw2_224.sw_in12k",
        "img_size": 224,
    },
    {
        "name": "coatnet_rmlp_1_rw2_in12k_ft",          #AUC=0.7893，F1很低
        "backbone": "coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k",
        "img_size": 224,
    },
    {
        "name": "coatnet_rmlp_2_rw_in12k",
        "backbone": "coatnet_rmlp_2_rw_224.sw_in12k",
        "img_size": 224,
    },
    {
        "name": "coatnet_rmlp_2_rw_in12k_ft",
        "backbone": "coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k",
        "img_size": 224,
    },
    {
        "name": "coatnet_rmlp_2_rw_384",
        "backbone": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
        "img_size": 384,
    },

    # -------- CoaT (Co-Scale Conv-Attentional) 系列 --------
    {
        "name": "coat_tiny",
        "backbone": "coat_tiny.in1k",
        "img_size": 224,
    },
    {
        "name": "coat_mini",
        "backbone": "coat_mini.in1k",
        "img_size": 224,
    },
    {
        "name": "coat_small",
        "backbone": "coat_small.in1k",
        "img_size": 224,
    },
    {
        "name": "coat_lite_tiny",
        "backbone": "coat_lite_tiny.in1k",
        "img_size": 224,
    },
    {
        "name": "coat_lite_mini",
        "backbone": "coat_lite_mini.in1k",
        "img_size": 224,
    },
    {
        "name": "coat_lite_small",
        "backbone": "coat_lite_small.in1k",
        "img_size": 224,
    },
    {
        "name": "coat_lite_medium",
        "backbone": "coat_lite_medium.in1k",
        "img_size": 224,
    },
    {
        "name": "coat_lite_medium_384",
        "backbone": "coat_lite_medium_384.in1k",
        "img_size": 384,
    },

    # -------- BoTNet (Bottleneck Transformers) 系列 --------
    {
        "name": "botnet26t_256",
        "backbone": "botnet26t_256.c1_in1k",
        "img_size": 256,
    },
    {
        "name": "botnet50ts_256",
        "backbone": "botnet50ts_256.c1_in1k",
        "img_size": 256,
    },
    {
        "name": "eca_botnext26ts_256",
        "backbone": "eca_botnext26ts_256.c1_in1k",
        "img_size": 256,
    },
]

# 通过这个名单控制要训练的模型
MODELS_TO_TRAIN = [
   #"convnext_base_in22k",      #AUC=0.8037，F1=0.2871
   #"convnext_base_in1k",       #AUC=0.7998，F1=0.3050
   #"convnext_base_in22k_384",    #AUC=0.8172，F1=0.3163
   "convnext_base_in22k_512",    #AUC=0.8180，F1=0.3454
   #"convnextv2_base_fcmae_384",   #AUC=0.8130，F1=0.2714

   # ConvNeXt Tiny/Small 系列
   "convnext_small_in22k_384",     #AUC=0.8125，F1=0.3420
   "convnext_small_in22k_512",     #AUC=0.8186，F1=0.2977

   # ConvNeXt V2 Tiny 系列
   "convnextv2_tiny_in22k_512",     #AUC=0.8176，F1=0.2792

   #"convnext_large_in22k_384",    #AUC=0.8146，F1=0.2728

   "swin_base_in22k_384",  #AUC=0.8101 还可以，F1=0.2636 很低

   #"swinv2_small_in1k_256",   #AUC=0.7992，F1很低

   #"swin_s3_small_224",    #AUC=0.7967，F1很低


    # CoaT (Co-Scale Conv-Attentional) 系列
    #"coat_tiny",    #AUC=0.8065，F1很低
    #"coat_mini",    #AUC=0.8063，F1=0.2842，但二者的峰值epoch相差4
    #"coat_lite_medium",    #AUC=0.8052，F1很低
    "coat_lite_medium_384",    #AUC=0.8199，F1=0.2762
    
]


# ================== EfficientNetV2 策略配置 ==================
# 强力推荐：使用 EfficientNetV2-S (ImageNet-21k) 配合 SE 模块和 AG-Crop
USE_EFFICIENTNET_V2 = False     # 是否将 EfficientNetV2 加入训练列表

if USE_EFFICIENTNET_V2:
    if "effv2_s_in21k" not in MODELS_TO_TRAIN:
        MODELS_TO_TRAIN.append("effv2_s_in21k")


LOSS_MODE = "hybrid"  # 选项: hybrid | asl | la
USE_TTA = False       # 验证时是否做轻量 TTA（翻转）
USE_JITTER = True
SHARPNESS = 0.2
MIXUP_PROB = 0.5
EMA_DECAY = 0.999
LA_TAU = 1.0

# ================== Attention-Guided Crop 配置 ==================
# 高分辨率+局部Patch策略（注意：仅支持 CNN 类 timm 模型，不支持 ViT）
USE_ATTENTION_CROP = False    # 是否启用 Attention-Guided Crop
AG_HIGH_RES = 512             # 高分辨率图像尺寸
AG_NUM_CROPS = 2              # 裁剪的局部区域数量
AG_CROP_SIZE = 224            # 裁剪后的尺寸
AG_FUSION_TYPE = "concat_attention"
AG_SHARE_BACKBONE = True
AG_BATCH_SIZE = 8             # Attention-Guided 模式的 batch size

# timm 模型的特征维度映射（用于 Attention-Guided Crop）
TIMM_FEATURE_DIMS = {
    "convnext_base": 1024,
    "convnext_base_in22k": 1024,
    "convnext_base_384_in22k": 1024,
    "efficientnet_b0": 1280,
    "tf_efficientnet_b4_ns": 1792,
    "resnet50": 2048,
    # === EfficientNetV2 ===
    "tf_efficientnetv2_s": 1280,
    "tf_efficientnetv2_m": 1280,
    "tf_efficientnetv2_l": 1280,
    "tf_efficientnetv2_s.in21k": 1280,
    # ViT 系列暂不支持（需要特殊处理）
}

# ================== 两阶段训练策略配置 ==================
USE_TWO_STAGE = False         # 是否启用两阶段训练
STAGE1_EPOCHS = 15            # Stage 1 的 epoch 数
STAGE2_LOSS = "asl"           # Stage 2 使用的损失函数: asl, la, focal
STAGE2_LR_FACTOR = 0.1        # Stage 2 学习率倍数
STAGE2_DISABLE_MIXUP = True   # Stage 2 是否禁用 Mixup


def prepare_data():
    """
    读取 CSV + 过滤出 images 目录中真实存在的样本 + 按病人划分 train/val/test
    
    采用分层抽样策略，确保稀有类别在验证集和测试集中都有样本：
    1. 先处理稀有类别的患者，确保每个稀有类别的患者按比例分配到三个集合
    2. 再处理剩余患者，进行随机划分
    """
    set_seed(RANDOM_SEED)

    df_all = pd.read_csv(LABEL_CSV)

    available_imgs = {
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    }

    df_labels = df_all[df_all["Image Index"].isin(available_imgs)].copy()
    df_labels = df_labels.reset_index(drop=True)

    print(f"CSV 总行数: {len(df_all)}")
    print("df_labels.columns =", list(df_labels.columns))
    print(f"images 目录中图片数: {len(available_imgs)}")
    print(f"用于训练/验证/测试的样本数(交集): {len(df_labels)}")

    # === patient-wise stratified split ===
    PATIENT_COL = "Patient ID"

    # 获取每个患者的疾病标签（用于分层）
    def get_patient_labels(patient_id):
        """获取某个患者的所有疾病标签"""
        patient_findings = df_labels[df_labels[PATIENT_COL] == patient_id]["Finding Labels"].values
        labels = set()
        for findings in patient_findings:
            if findings != "No Finding":
                for disease in str(findings).split("|"):
                    labels.add(disease.strip())
        return labels

    unique_patients = df_labels[PATIENT_COL].unique()
    print(f"总病人数: {len(unique_patients)}")

    # 构建患者到疾病的映射
    patient_to_labels = {p: get_patient_labels(p) for p in unique_patients}
    
    # 构建疾病到患者的映射（用于分层抽样）
    from src.cxr_config import RARE_CLASSES, TRAIN_RATIO, TEST_RATIO
    
    disease_to_patients = {cls: [] for cls in RARE_CLASSES}
    for patient, labels in patient_to_labels.items():
        for disease in labels:
            if disease in RARE_CLASSES:
                disease_to_patients[disease].append(patient)
    
    # 打印稀有类别的患者数量
    print("\n稀有类别患者分布:")
    for disease, patients in disease_to_patients.items():
        print(f"  {disease}: {len(patients)} 个患者")
    
    rng = np.random.RandomState(RANDOM_SEED)
    
    # 初始化三个集合
    train_patients = set()
    val_patients = set()
    test_patients = set()
    assigned_patients = set()  # 已分配的患者
    
    # Step 1: 对稀有类别进行分层抽样
    # 确保每个稀有类别在 val 和 test 中至少有 min_samples 个患者
    min_samples_per_split = 2  # 每个集合至少保证有样本
    
    for disease in RARE_CLASSES:
        disease_patients = [p for p in disease_to_patients[disease] if p not in assigned_patients]
        if len(disease_patients) == 0:
            continue
            
        rng.shuffle(disease_patients)
        n_total = len(disease_patients)
        
        # 计算每个集合应该分配的数量
        n_test = max(min_samples_per_split, int(n_total * TEST_RATIO))
        n_val = max(min_samples_per_split, int(n_total * VAL_RATIO))
        n_train = n_total - n_test - n_val
        
        # 确保 train 至少有一定数量
        if n_train < 1:
            # 如果患者太少，优先保证 test 和 val 各有 1 个
            if n_total >= 3:
                n_test = 1
                n_val = 1
                n_train = n_total - 2
            elif n_total == 2:
                n_test = 1
                n_val = 1
                n_train = 0
            else:  # n_total == 1
                # 只有一个患者，放入训练集（无法保证 val/test 有样本）
                n_train = 1
                n_val = 0
                n_test = 0
        
        # 分配患者
        idx = 0
        for p in disease_patients[idx:idx + n_test]:
            test_patients.add(p)
            assigned_patients.add(p)
        idx += n_test
        
        for p in disease_patients[idx:idx + n_val]:
            val_patients.add(p)
            assigned_patients.add(p)
        idx += n_val
        
        for p in disease_patients[idx:idx + n_train]:
            train_patients.add(p)
            assigned_patients.add(p)
    
    # Step 2: 对剩余患者进行随机划分
    remaining_patients = [p for p in unique_patients if p not in assigned_patients]
    rng.shuffle(remaining_patients)
    
    n_remaining = len(remaining_patients)
    n_test_remaining = int(n_remaining * TEST_RATIO)
    n_val_remaining = int(n_remaining * VAL_RATIO)
    
    for p in remaining_patients[:n_test_remaining]:
        test_patients.add(p)
    for p in remaining_patients[n_test_remaining:n_test_remaining + n_val_remaining]:
        val_patients.add(p)
    for p in remaining_patients[n_test_remaining + n_val_remaining:]:
        train_patients.add(p)
    
    # 获取每个集合的文件列表
    train_files = df_labels[df_labels[PATIENT_COL].isin(train_patients)]["Image Index"].tolist()
    val_files = df_labels[df_labels[PATIENT_COL].isin(val_patients)]["Image Index"].tolist()
    test_files = df_labels[df_labels[PATIENT_COL].isin(test_patients)]["Image Index"].tolist()

    print(f"\n数据集划分结果:")
    print(f"  训练集: {len(train_patients)} 个病人, {len(train_files)} 张图片 ({len(train_files)/len(df_labels)*100:.1f}%)")
    print(f"  验证集: {len(val_patients)} 个病人, {len(val_files)} 张图片 ({len(val_files)/len(df_labels)*100:.1f}%)")
    print(f"  测试集: {len(test_patients)} 个病人, {len(test_files)} 张图片 ({len(test_files)/len(df_labels)*100:.1f}%)")

    # 校验：三个集合是否完全不重叠
    assert len(train_patients & val_patients) == 0, "train / val 病人集合有重叠!"
    assert len(train_patients & test_patients) == 0, "train / test 病人集合有重叠!"
    assert len(val_patients & test_patients) == 0, "val / test 病人集合有重叠!"
    
    # 验证稀有类别在每个集合中的分布
    print("\n各集合中稀有类别的样本数:")
    for split_name, split_files in [("训练集", train_files), ("验证集", val_files), ("测试集", test_files)]:
        split_df = df_labels[df_labels["Image Index"].isin(split_files)]
        print(f"  {split_name}:")
        for disease in RARE_CLASSES:
            count = split_df["Finding Labels"].str.contains(disease, na=False).sum()
            print(f"    {disease}: {count}")

    return df_labels, train_files, val_files, test_files



from torch import amp
autocast = amp.autocast
GradScaler = amp.GradScaler


def mixup_data(x, y, alpha: float = 0.0):
    """
    Mixup 数据增强。
    
    Args:
        x: 输入图像 tensor
        y: 标签 tensor
        alpha: Mixup 混合系数（从 Beta(alpha, alpha) 分布采样）
    
    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class ModelEma:
    """简易 EMA 实现，用于平滑权重"""
    def __init__(self, model, decay=0.999):
        self.ema = deepcopy(model)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    def update(self, model):
        with torch.no_grad():
            ema_params = dict(self.ema.named_parameters())
            model_params = dict(model.named_parameters())
            for name, param in model_params.items():
                ema_params[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
            # 同步 buffers（如 BN 统计量）
            for ema_buf, buf in zip(self.ema.buffers(), model.buffers()):
                ema_buf.copy_(buf)

    def state_dict(self):
        return self.ema.state_dict()

    def to(self, device):
        self.ema.to(device)
        return self

    def forward(self, x):
        return self.ema(x)


def build_loss(df_labels, stage: int = 1):
    """
    构建损失函数（支持两阶段训练）
    
    Args:
        df_labels: 标签数据
        stage: 训练阶段 (1 或 2)
    """
    # pos_weight 温和化处理
    pos_weight = compute_pos_weight(
        df_labels, CLASS_NAMES,
        smoothing="clamp",
        clamp_max=20.0,
    ).to(DEVICE)
    
    # Stage 2: 使用指标对齐损失
    if USE_TWO_STAGE and stage == 2:
        if STAGE2_LOSS == "asl":
            print(f"  [Stage 2 Loss] ASL (gamma_neg=4.0, clip=0.05)")
            return AsymmetricLoss(gamma_pos=0.0, gamma_neg=4.0, clip=0.05)
        elif STAGE2_LOSS == "la":
            counts = []
            for cls in CLASS_NAMES:
                c = df_labels["Finding Labels"].str.contains(cls, na=False).sum()
                counts.append(c)
            freq = torch.tensor(counts, dtype=torch.float32)
            logit_bias = get_logit_adjustment(freq, tau=LA_TAU).to(DEVICE)
            print(f"  [Stage 2 Loss] LogitAdjustedLoss")
            return LogitAdjustedLoss(logit_bias=logit_bias)
        elif STAGE2_LOSS == "focal":
            from src.loss_utils import MultiLabelFocalLoss
            print(f"  [Stage 2 Loss] Focal Loss")
            return MultiLabelFocalLoss(gamma=2.0)
        else:
            print(f"  [Stage 2 Loss] HybridLoss (激进)")
            return HybridLoss(
                pos_weight=pos_weight,
                focal_gamma=2.0,
                smooth=0.05,
                bce_weight=1.0,
                focal_weight=1.0,
                smooth_weight=0.5,
            )
    
    # Stage 1: 稳定学习
    print(f"  [Stage 1 Loss] HybridLoss (温和)")
    if LOSS_MODE == "asl":
        return AsymmetricLoss(gamma_pos=0.0, gamma_neg=4.0, clip=0.05)
    if LOSS_MODE == "la":
        counts = []
        for cls in CLASS_NAMES:
            c = df_labels["Finding Labels"].str.contains(cls, na=False).sum()
            counts.append(c)
        freq = torch.tensor(counts, dtype=torch.float32)
        logit_bias = get_logit_adjustment(freq, tau=LA_TAU).to(DEVICE)
        return LogitAdjustedLoss(logit_bias=logit_bias)
    
    return HybridLoss(
        pos_weight=pos_weight,
        focal_gamma=2.0,
        smooth=0.05,
        bce_weight=1.0,
        focal_weight=0.5,
        smooth_weight=0.1,
    )


def adjust_learning_rate_for_stage2(optimizer, base_lr):
    """
    调整 Stage 2 的学习率。
    
    支持 Layer-wise LR Decay：按比例缩放每个参数组的学习率，
    保持各层之间的相对比例不变。
    """
    # 检查是否使用了 Layer-wise LR Decay（多个参数组且有不同的 lr）
    if len(optimizer.param_groups) > 1:
        # Layer-wise LR Decay 模式：按比例缩放每个参数组的 lr
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = old_lr * STAGE2_LR_FACTOR
            param_group['lr'] = new_lr
        new_base_lr = base_lr * STAGE2_LR_FACTOR
        print(f"  [Layer Decay] 所有层学习率按 {STAGE2_LR_FACTOR}x 缩放")
        return new_base_lr
    else:
        # 普通模式：直接设置新的学习率
        new_lr = base_lr * STAGE2_LR_FACTOR
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr


def inference_with_tta(model, imgs):
    """轻量 TTA：原图 + 水平翻转 平均"""
    if not USE_TTA:
        return torch.sigmoid(model(imgs))
    probs = []
    logits = model(imgs)
    probs.append(torch.sigmoid(logits))
    imgs_flip = torch.flip(imgs, dims=[3])
    logits_flip = model(imgs_flip)
    probs.append(torch.sigmoid(logits_flip))
    return torch.stack(probs, dim=0).mean(dim=0)


def get_multi_res_transforms_for_timm(low_res: int, high_res: int, train: bool = True):
    """为 timm 模型构造多分辨率数据增强（使用标准 ImageNet 归一化）"""
    import torchvision.transforms as T
    
    # 基础增强
    if train:
        base_augment = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
        ])
    else:
        base_augment = None
    
    # 归一化（timm 标准）
    normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
    
    return base_augment, low_res, high_res, normalize


class ChestXrayMultiResDatasetTimm(ChestXrayMultiResDatasetV2):
    """为 timm 模型定制的多分辨率数据集（使用标准归一化而非 XRV 归一化）"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 覆盖归一化方式
        import torchvision.transforms as T
        self.normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
        self.to_tensor = T.ToTensor()
    
    def __getitem__(self, idx):
        import torchvision.transforms.functional as TF
        import random
        
        img_name = self.file_list[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        
        if self.train:
            if random.random() > 0.5:
                img = TF.hflip(img)
            angle = random.uniform(-10, 10)
            img = TF.rotate(img, angle)
        
        img_high = img.resize((self.high_res, self.high_res))
        img_low = img.resize((self.low_res, self.low_res))
        
        img_low = self.normalize(self.to_tensor(img_low))
        img_high = self.normalize(self.to_tensor(img_high))
        
        findings = self.df_labels.loc[img_name]["Finding Labels"]
        label_vec = self._get_label_vector(findings)
        
        return img_low, img_high, label_vec, img_name


def train_one_model(cfg, df_labels, train_files, val_files):
    # 每个模型训练前重新设置随机种子，确保可重复性
    # 避免因训练顺序不同导致随机状态累积差异
    set_seed(RANDOM_SEED)
    
    name = cfg["name"]
    backbone = cfg["backbone"]
    img_size = cfg["img_size"]
    checkpoint_path = cfg.get("checkpoint", None)

    print(f"\n========== 开始训练模型: {name} ==========")
    print(f"Backbone: {backbone}, img_size={img_size}")
    
    # 检查是否支持 Attention-Guided Crop
    use_ag_crop = USE_ATTENTION_CROP and backbone in TIMM_FEATURE_DIMS
    if USE_ATTENTION_CROP and backbone not in TIMM_FEATURE_DIMS:
        print(f"  [Warning] {backbone} 不在 TIMM_FEATURE_DIMS 中，禁用 Attention-Guided Crop")
        use_ag_crop = False
    
    if use_ag_crop:
        print(f"  模式: Attention-Guided Crop (high_res={AG_HIGH_RES}, crops={AG_NUM_CROPS})")
    
    if USE_TWO_STAGE:
        print(f"  两阶段训练: Stage 1 (epoch 1-{STAGE1_EPOCHS}) -> Stage 2 (epoch {STAGE1_EPOCHS+1}-{NUM_EPOCHS})")
        print(f"    Stage 2 损失: {STAGE2_LOSS}, 学习率因子: {STAGE2_LR_FACTOR}")

    # ========= A. 数据集 & DataLoader =========
    if use_ag_crop:
        # 多分辨率数据集
        train_dataset = ChestXrayMultiResDatasetTimm(
            file_list=train_files,
            df_labels=df_labels,
            train=True,
            low_res=img_size,
            high_res=AG_HIGH_RES,
        )
        val_dataset = ChestXrayMultiResDatasetTimm(
            file_list=val_files,
            df_labels=df_labels,
            train=False,
            low_res=img_size,
            high_res=AG_HIGH_RES,
        )
        batch_size = AG_BATCH_SIZE
    else:
        # 标准单分辨率数据集
        train_dataset = ChestXrayDataset(
            file_list=train_files,
            df_labels=df_labels,
            transform=get_transforms_for_size(img_size, train=True, use_jitter=USE_JITTER, sharpness=SHARPNESS),
        )
        val_dataset = ChestXrayDataset(
            file_list=val_files,
            df_labels=df_labels,
            transform=get_transforms_for_size(img_size, train=False),
        )
        batch_size = BATCH_SIZE

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ========= B. 模型 & 损失函数 =========
    base_model = TimmMultiLabelModel(backbone_name=backbone, pretrained=True, checkpoint_path=checkpoint_path)
    
    if use_ag_crop:
        feature_dim = TIMM_FEATURE_DIMS[backbone]
        model = wrap_model_with_attention_crop(
            base_model,
            backbone_type="timm",
            num_classes=len(CLASS_NAMES),
            feature_dim=feature_dim,
            num_crops=AG_NUM_CROPS,
            crop_size=AG_CROP_SIZE,
            high_res_size=AG_HIGH_RES,
            fusion_type=AG_FUSION_TYPE,
            share_backbone=AG_SHARE_BACKBONE,
        ).to(DEVICE)
        print(f"  >> 已包装为 AttentionGuidedWrapper (feature_dim={feature_dim})")
    else:
        model = base_model.to(DEVICE)

    criterion = build_loss(df_labels)

    # 使用统一的超参数（经测试，统一配置性能更好）
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # ------ 架构特定超参数配置（已禁用）------
    # 如需启用，取消下面的注释并在 cxr_config.py 中设置 TIMM_USE_ARCH_SPECIFIC_CONFIG = True
    # arch_config = get_arch_config(name)
    # LR = arch_config["lr"]
    # WEIGHT_DECAY = arch_config["weight_decay"]
    # WARMUP_EPOCHS = arch_config["warmup_epochs"]
    # MIXUP_ALPHA = arch_config["mixup_alpha"]
    # LAYER_DECAY = arch_config["layer_decay"]
    # optimizer = build_optimizer_with_layer_decay(
    #     model=model,
    #     model_name=name,
    #     base_lr=LR,
    #     weight_decay=WEIGHT_DECAY,
    #     layer_decay=LAYER_DECAY,
    #     num_layers=6,
    # )

    schedulers = []
    milestones = []
    if WARMUP_EPOCHS > 0:
        schedulers.append(LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS))
        schedulers.append(CosineAnnealingLR(optimizer, T_max=max(1, NUM_EPOCHS - WARMUP_EPOCHS)))
        milestones = [WARMUP_EPOCHS]
    else:
        schedulers.append(CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS))
        milestones = [NUM_EPOCHS]
    scheduler = SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)
    scaler = GradScaler()  # AMP scaler
    ema = None
    if USE_EMA:
        ema = ModelEma(model, decay=EMA_DECAY).to(DEVICE)

    # ===== F1 最优跟踪 =====
    best_f1_score = -1.0
    best_f1_thresholds = None
    model_path_f1 = os.path.join(SAVE_DIR, f"{name}_best_f1.pth")
    val_pred_path_f1 = os.path.join(SAVE_DIR, f"val_preds_{name}_f1.npz")
    no_improve_f1_epochs = 0
    
    # ===== AUC 最优跟踪 =====
    best_auc_score = -1.0
    best_auc_thresholds = None
    model_path_auc = os.path.join(SAVE_DIR, f"{name}_best_auc.pth")
    val_pred_path_auc = os.path.join(SAVE_DIR, f"val_preds_{name}_auc.npz")
    no_improve_auc_epochs = 0
    
    # 两阶段训练状态
    current_stage = 1
    stage2_criterion = None

    # ========= C. 训练循环 =========
    for epoch in range(1, NUM_EPOCHS + 1):
        # ========== 检查是否需要切换到 Stage 2 ==========
        if USE_TWO_STAGE and epoch == STAGE1_EPOCHS + 1 and current_stage == 1:
            current_stage = 2
            print(f"\n{'='*60}")
            print(f"[{name}] 切换到 Stage 2 (指标对齐微调)")
            print(f"{'='*60}")
            
            # 1. 创建 Stage 2 损失函数
            stage2_criterion = build_loss(df_labels, stage=2)
            
            # 2. 调整学习率
            new_lr = adjust_learning_rate_for_stage2(optimizer, LR)
            print(f"  学习率: {LR} -> {new_lr}")
            
            # 3. 重置早停计数器
            no_improve_f1_epochs = 0
            no_improve_auc_epochs = 0
            print(f"  早停计数器已重置")
            print(f"{'='*60}\n")
        
        # 选择当前阶段的损失函数
        current_criterion = stage2_criterion if current_stage == 2 else criterion
        
        # 是否使用 Mixup
        use_mixup_this_epoch = MIXUP_ALPHA > 0
        if USE_TWO_STAGE and current_stage == 2 and STAGE2_DISABLE_MIXUP:
            use_mixup_this_epoch = False
        
        model.train()
        train_loss_sum = 0.0

        stage_tag = f"S{current_stage}" if USE_TWO_STAGE else ""
        pbar = tqdm(train_loader, desc=f"[{name}] Epoch {epoch}/{NUM_EPOCHS} {stage_tag} [Train]")
        for batch in pbar:
            # 根据数据集类型解包
            if use_ag_crop:
                imgs_low, imgs_high, labels, _ = batch
                imgs_low = imgs_low.to(DEVICE)
                imgs_high = imgs_high.to(DEVICE)
                labels = labels.to(DEVICE)
                current_batch_size = imgs_low.size(0)
            else:
                imgs, labels, _ = batch
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)
                current_batch_size = imgs.size(0)

            optimizer.zero_grad()

            # AMP 半精度加速 + 降显存
            with autocast(device_type="cuda"):
                if use_ag_crop:
                    logits = model(imgs_low, imgs_high)
                    loss = current_criterion(logits, labels)
                elif use_mixup_this_epoch and random.random() < MIXUP_PROB:
                    imgs_m, y_a, y_b, lam = mixup_data(imgs, labels, alpha=MIXUP_ALPHA)
                    logits = model(imgs_m)
                    loss = lam * current_criterion(logits, y_a) + (1 - lam) * current_criterion(logits, y_b)
                else:
                    logits = model(imgs)
                    loss = current_criterion(logits, labels)

            scaler.scale(loss).backward()
            if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            if ema:
                ema.update(model)

            train_loss_sum += loss.item() * current_batch_size
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "stage": current_stage})

        scheduler.step()
        avg_train_loss = train_loss_sum / len(train_dataset)

        # ========= D. 验证 =========
        model.eval()
        eval_model = ema.ema if ema is not None else model
        all_labels = []
        all_probs = []

        val_desc = f"[{name}] Epoch {epoch}/{NUM_EPOCHS} [Val"
        if use_ag_crop:
            val_desc += " AG-Crop]"
        elif USE_TTA:
            val_desc += " TTA]"
        else:
            val_desc += "]"

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=val_desc):
                if use_ag_crop:
                    imgs_low, imgs_high, labels, _ = batch
                    imgs_low = imgs_low.to(DEVICE)
                    imgs_high = imgs_high.to(DEVICE)
                    logits = eval_model(imgs_low, imgs_high)
                    probs = torch.sigmoid(logits)
                else:
                    imgs, labels, _ = batch
                    imgs = imgs.to(DEVICE)
                    probs = inference_with_tta(eval_model, imgs)

                all_labels.append(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        y_true = np.concatenate(all_labels, axis=0)
        y_pred_prob = np.concatenate(all_probs, axis=0)

        auc_macro, f1_macro, f1_weighted = compute_metrics(y_true, y_pred_prob, thresholds=None)
        thresholds = search_best_thresholds(y_true, y_pred_prob)
        auc_macro_t, f1_macro_t, f1_weighted_t = compute_metrics(y_true, y_pred_prob, thresholds)

        stage_info = f" [Stage {current_stage}]" if USE_TWO_STAGE else ""
        print(f"\n[{name}] Epoch {epoch}{stage_info}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val AUC:        {auc_macro:.4f}")
        print(f"  Val Macro-F1 (thr=0.5):   {f1_macro:.4f}")
        print(f"  Val Macro-F1 (best thr):  {f1_macro_t:.4f}")
        if USE_TWO_STAGE:
            print(f"  Current LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 获取当前模型状态（用于保存）
        current_model_state = ema.state_dict() if ema is not None else model.state_dict()
        
        # ===== 检查 F1 是否提升并保存 =====
        if f1_macro_t > best_f1_score + EARLY_STOP_MIN_DELTA:
            best_f1_score = f1_macro_t
            best_f1_thresholds = thresholds
            no_improve_f1_epochs = 0
            torch.save(
                {
                    "model_state": current_model_state,
                    "thresholds": best_f1_thresholds,
                    "backbone": backbone,
                    "img_size": img_size,
                    "metric": "f1",
                    "best_f1": best_f1_score,
                    "auc_at_best_f1": auc_macro,
                    "stage": current_stage if USE_TWO_STAGE else 1,
                    "two_stage_config": {
                        "enabled": USE_TWO_STAGE,
                        "stage1_epochs": STAGE1_EPOCHS,
                        "stage2_loss": STAGE2_LOSS,
                    } if USE_TWO_STAGE else None,
                },
                model_path_f1,
            )
            np.savez(
                val_pred_path_f1,
                y_true=y_true,
                y_pred_prob=y_pred_prob,
                best_f1=best_f1_score,
                auc_at_best_f1=auc_macro,
                thresholds=best_f1_thresholds,
            )
            print(f"  >> [F1最优] 已保存: F1={best_f1_score:.4f}, AUC={auc_macro:.4f}")
        else:
            no_improve_f1_epochs += 1
        
        # ===== 检查 AUC 是否提升并保存 =====
        if auc_macro > best_auc_score + EARLY_STOP_MIN_DELTA:
            best_auc_score = auc_macro
            best_auc_thresholds = thresholds
            no_improve_auc_epochs = 0
            torch.save(
                {
                    "model_state": current_model_state,
                    "thresholds": best_auc_thresholds,
                    "backbone": backbone,
                    "img_size": img_size,
                    "metric": "auc",
                    "best_auc": best_auc_score,
                    "f1_at_best_auc": f1_macro_t,
                    "stage": current_stage if USE_TWO_STAGE else 1,
                    "two_stage_config": {
                        "enabled": USE_TWO_STAGE,
                        "stage1_epochs": STAGE1_EPOCHS,
                        "stage2_loss": STAGE2_LOSS,
                    } if USE_TWO_STAGE else None,
                },
                model_path_auc,
            )
            np.savez(
                val_pred_path_auc,
                y_true=y_true,
                y_pred_prob=y_pred_prob,
                best_auc=best_auc_score,
                f1_at_best_auc=f1_macro_t,
                thresholds=best_auc_thresholds,
            )
            print(f"  >> [AUC最优] 已保存: AUC={best_auc_score:.4f}, F1={f1_macro_t:.4f}")
        else:
            no_improve_auc_epochs += 1

        # ===== Early stopping: 两个指标都没有改进才停止 =====
        no_improve_epochs = min(no_improve_f1_epochs, no_improve_auc_epochs)
        if EARLY_STOP and epoch > WARMUP_EPOCHS and no_improve_epochs >= EARLY_STOP_PATIENCE:
            print(f"\n[EarlyStop] {name}: 两指标均无改进超过 {EARLY_STOP_PATIENCE} epochs")
            print(f"    Best F1={best_f1_score:.4f}, Best AUC={best_auc_score:.4f}")
            print(f"    停止于 epoch {epoch}")
            break

    print(f"\n[{name}] 训练结束:")
    print(f"    Best F1:  {best_f1_score:.4f} (保存于 {model_path_f1})")
    print(f"    Best AUC: {best_auc_score:.4f} (保存于 {model_path_auc})")



def main():
    # 设置日志记录，所有 print 输出同时保存到 logs/ 目录
    setup_logger("train_timm_models")
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    df_labels, train_files, val_files, test_files = prepare_data()
    
    # 保存数据集划分信息，供后续测试使用
    split_info_path = os.path.join(SAVE_DIR, "dataset_split.npz")
    np.savez(
        split_info_path,
        train_files=np.array(train_files),
        val_files=np.array(val_files),
        test_files=np.array(test_files),
    )
    print(f"\n数据集划分信息已保存到: {split_info_path}")

    cfg_map = {cfg["name"]: cfg for cfg in MODEL_CONFIGS}

    for name in MODELS_TO_TRAIN:
        if name not in cfg_map:
            print(f"[跳过] 未在 MODEL_CONFIGS 中找到配置: {name}")
            continue
        train_one_model(cfg_map[name], df_labels, train_files, val_files)

    # ========== 所有模型训练完成后，自动运行集成评估 ==========
    print("\n" + "=" * 70)
    print("所有模型训练完成，开始集成评估...")
    print("=" * 70)
    
    from ensemble_timm import main as ensemble_main
    ensemble_main()
    
    # 关闭日志记录
    close_logger()


if __name__ == "__main__":
    main()
