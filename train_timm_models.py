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
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from cxr_config import (
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
)
from dataset import ChestXrayDataset, ChestXrayMultiResDatasetV2
from metrics_utils import compute_metrics, search_best_thresholds
from models_timm import TimmMultiLabelModel
from models_attention_crop import wrap_model_with_attention_crop
from loss_utils import (
    compute_pos_weight,
    HybridLoss,
    AsymmetricLoss,
    LogitAdjustedLoss,
    get_logit_adjustment,
)
from log_utils import setup_logger, close_logger



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

    # -------- EVA-02 --------
    {
        "name": "eva02_large_448",
        "backbone": "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
        "img_size": 448,
    },
]

# 通过这个名单控制要训练的模型
MODELS_TO_TRAIN = [
   "convnext_base_in22k",
   "convnext_base_in1k",

   "convnext_base_in22k_384",
   "convnext_base_in22k_512",

   # 新增模型
   #"swin_base_in22k_384",  #性能不如convnext
   "convnextv2_base_fcmae_384",
   #"maxvit_base_in21k_512",  #很大，还未训练
   #"eva02_large_448",       #很大，还未训练
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
    """读取 CSV + 过滤出 images 目录中真实存在的样本 + 按病人划分 train/val"""
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
    print(f"用于训练/验证的样本数(交集): {len(df_labels)}")

    # === patient-wise split ===
    PATIENT_COL = "Patient ID"   # 如果列名不同，在这里改

    unique_patients = df_labels[PATIENT_COL].unique()
    print(f"总病人数: {len(unique_patients)}")

    rng = np.random.RandomState(RANDOM_SEED)
    rng.shuffle(unique_patients)

    n_val_patients = max(1, int(len(unique_patients) * VAL_RATIO))
    val_patients = set(unique_patients[:n_val_patients])
    train_patients = set(unique_patients[n_val_patients:])

    train_files = df_labels[df_labels[PATIENT_COL].isin(train_patients)]["Image Index"].tolist()
    val_files   = df_labels[df_labels[PATIENT_COL].isin(val_patients)]["Image Index"].tolist()

    print(f"训练集: {len(train_patients)} 个病人, {len(train_files)} 张图片")
    print(f"验证集: {len(val_patients)} 个病人, {len(val_files)} 张图片")

    # 简单校验：train / val 病人是否完全不重叠
    assert len(train_patients & val_patients) == 0, "train / val 病人集合有重叠, 请检查划分逻辑!"

    return df_labels, train_files, val_files



from torch import amp
autocast = amp.autocast
GradScaler = amp.GradScaler


def mixup_data(x, y, alpha=MIXUP_ALPHA):
    if alpha <= 0:
        return x, y, 1.0
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
            from loss_utils import MultiLabelFocalLoss
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
    """调整 Stage 2 的学习率"""
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

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
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

    best_val_score = -1.0
    best_thresholds = None
    model_path = os.path.join(SAVE_DIR, f"{name}_best.pth")
    val_pred_path = os.path.join(SAVE_DIR, f"val_preds_{name}.npz")
    no_improve_epochs = 0
    
    # 跟踪整个训练过程中的最佳指标
    best_auc_ever = -1.0
    best_f1_macro_ever = -1.0
    best_f1_weighted_ever = -1.0
    
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
            no_improve_epochs = 0
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

        # 更新整个训练过程中的最佳指标
        if auc_macro > best_auc_ever:
            best_auc_ever = auc_macro
        if f1_macro_t > best_f1_macro_ever:
            best_f1_macro_ever = f1_macro_t
        if f1_weighted_t > best_f1_weighted_ever:
            best_f1_weighted_ever = f1_weighted_t

        stage_info = f" [Stage {current_stage}]" if USE_TWO_STAGE else ""
        print(f"\n[{name}] Epoch {epoch}{stage_info}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val AUC:        {auc_macro:.4f}")
        print(f"  Val Macro-F1 (thr=0.5):   {f1_macro:.4f}")
        print(f"  Val Macro-F1 (best thr):  {f1_macro_t:.4f}")
        # print(f"  Val Weighted-F1 (best thr):{f1_weighted_t:.4f}")
        if USE_TWO_STAGE:
            print(f"  Current LR: {optimizer.param_groups[0]['lr']:.6f}")

        val_score = f1_macro_t
        if val_score > best_val_score + EARLY_STOP_MIN_DELTA:
            best_val_score = val_score
            best_thresholds = thresholds
            no_improve_epochs = 0
            torch.save(
                {
                    "model_state": ema.state_dict() if ema is not None else model.state_dict(),
                    "thresholds": best_thresholds,
                    "backbone": backbone,
                    "img_size": img_size,
                    "stage": current_stage if USE_TWO_STAGE else 1,
                    "two_stage_config": {
                        "enabled": USE_TWO_STAGE,
                        "stage1_epochs": STAGE1_EPOCHS,
                        "stage2_loss": STAGE2_LOSS,
                    } if USE_TWO_STAGE else None,
                },
                model_path,
            )
            np.savez(
                val_pred_path,
                y_true=y_true,
                y_pred_prob=y_pred_prob,
                best_auc=best_auc_ever,
                best_f1_macro=best_f1_macro_ever,
                best_f1_weighted=best_f1_weighted_ever,
            )
            print(f"  >> 新的最优模型已保存: {model_path}")
            print(f"  >> 验证集预测已保存: {val_pred_path}")
        else:
            no_improve_epochs += 1

        if EARLY_STOP and epoch > WARMUP_EPOCHS and no_improve_epochs >= EARLY_STOP_PATIENCE:
            print(f"\n[EarlyStop] {name}: no improvement in {EARLY_STOP_PATIENCE} epochs (best F1={best_val_score:.4f}). Stop at epoch {epoch}.")
            break

    print(f"\n[{name}] 训练结束，整个训练过程最佳指标:")
    print(f"    Best AUC:         {best_auc_ever:.4f}")
    print(f"    Best Macro-F1:    {best_f1_macro_ever:.4f}")
    # print(f"    Best Weighted-F1: {best_f1_weighted_ever:.4f}")



def main():
    # 设置日志记录，所有 print 输出同时保存到 logs/ 目录
    setup_logger("train_timm_models")
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    df_labels, train_files, val_files = prepare_data()

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
