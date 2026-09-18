# train.py
"""
DenseNet121 多标签分类训练脚本 - 集成学习版本。
训练多个 torchxrayvision 医学预训练模型，并通过加权平均进行集成。

预训练权重（均不含 NIH ChestX-ray14 数据，避免数据泄漏）：
- densenet121-res224-chex:     CheXpert 数据集
- densenet121-res224-pc:       PadChest 数据集
- densenet121-res224-mimic_nb: MIMIC-CXR (NoBBox)
- densenet121-res224-mimic_ch: MIMIC-CXR (ChestOnly)
"""

import os
import random
import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from copy import deepcopy

from src.cxr_config import (
    LABEL_CSV, SAVE_DIR, IMAGES_DIR,
    DEVICE, RANDOM_SEED,
    BATCH_SIZE, NUM_EPOCHS, WARMUP_EPOCHS, LR, WEIGHT_DECAY, 
    VAL_RATIO, TEST_RATIO, TRAIN_RATIO, RARE_CLASSES, CLASS_NAMES,
    MIXUP_ALPHA, GRAD_CLIP_NORM, USE_EMA,
    EARLY_STOP, EARLY_STOP_PATIENCE, )
from src.dataset import ChestXrayDataset, get_xrv_transforms, ChestXrayMultiResDatasetV2
from src.models import TorchXRayVisionDenseNet
from src.models_attention_crop import wrap_model_with_attention_crop
from src.metrics_utils import compute_metrics, search_best_thresholds
from src.loss_utils import (
    compute_pos_weight,
    HybridLoss,
    AsymmetricLoss,
    LogitAdjustedLoss,
    get_logit_adjustment,
)
from src.log_utils import setup_logger, close_logger


# ================== 集成学习配置 ==================
# 要训练的所有模型（均不含 NIH 数据）
ENSEMBLE_MODELS = {
    "xrv-chex": "densenet121-res224-chex",         # CheXpert 数据集
    "xrv-pc": "densenet121-res224-pc",             # PadChest 数据集
    "xrv-mimic_nb": "densenet121-res224-mimic_nb", # MIMIC-CXR (NoBBox)
    "xrv-mimic_ch": "densenet121-res224-mimic_ch", # MIMIC-CXR (ChestOnly)
}

# 是否冻结 backbone（仅训练分类头）- 对于医学预训练通常设为 False
FREEZE_BACKBONE = False

# 损失函数模式：hybrid / asl / la
LOSS_MODE = "hybrid"

# ================== TTA 配置 ==================
USE_TTA = False                # 验证/测试时是否启用 TTA
TTA_SCALES = [224, 256]       # 2-scale resize TTA（原始尺寸 + 放大尺寸）
TTA_HFLIP = True              # 是否启用水平翻转 TTA

# ================== Attention-Guided Crop 配置 ==================
# 高分辨率+局部Patch策略，针对小病灶（Nodule, Mass, Pneumothorax）优化
USE_ATTENTION_CROP = False    # 是否启用 Attention-Guided Crop（显存需求更高）
AG_HIGH_RES = 512             # 高分辨率图像尺寸
AG_NUM_CROPS = 2              # 裁剪的局部区域数量
AG_CROP_SIZE = 224            # 裁剪后的尺寸
AG_FUSION_TYPE = "concat_attention"  # 融合方式: concat, add, concat_attention
AG_SHARE_BACKBONE = True      # 是否共享 backbone（节省显存）
AG_BATCH_SIZE = 12            # Attention-Guided 模式的 batch size（需要更小）

# ================== 两阶段训练策略配置 ==================
# Stage 1: 稳定学习表征（使用温和的 HybridLoss）
# Stage 2: 指标对齐微调（切换到 ASL，降低学习率）
USE_TWO_STAGE = False         # 是否启用两阶段训练
STAGE1_EPOCHS = 15            # Stage 1 的 epoch 数（之后切换到 Stage 2）
STAGE2_LOSS = "asl"           # Stage 2 使用的损失函数: asl, la, focal
STAGE2_LR_FACTOR = 0.1        # Stage 2 学习率相对于 Stage 1 的倍数
STAGE2_DISABLE_MIXUP = True   # Stage 2 是否禁用 Mixup（更精细调整）
STAGE2_RESET_EMA = False      # Stage 2 是否重置 EMA（从当前模型重新开始）


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def inference_with_tta(model, imgs, scales=TTA_SCALES, use_hflip=TTA_HFLIP):
    """
    XRV 模型的 TTA 推理策略：
    1. Horizontal Flip TTA - 水平翻转
    2. 2-scale resize TTA - 多尺度推理
    
    Args:
        model: 模型
        imgs: 输入图像 (B, C, H, W)，假设已经是 224x224
        scales: 多尺度列表，如 [224, 256]
        use_hflip: 是否使用水平翻转
    
    Returns:
        平均后的预测概率 (B, num_classes)
    """
    if not USE_TTA:
        return torch.sigmoid(model(imgs))
    
    import torch.nn.functional as F
    
    all_probs = []
    original_size = imgs.shape[-1]  # 原始尺寸（通常是 224）
    
    for scale in scales:
        # 如果需要 resize 到不同尺度
        if scale != original_size:
            imgs_scaled = F.interpolate(imgs, size=(scale, scale), mode='bilinear', align_corners=False)
        else:
            imgs_scaled = imgs
        
        # 原图推理
        logits = model(imgs_scaled)
        all_probs.append(torch.sigmoid(logits))
        
        # 水平翻转推理
        if use_hflip:
            imgs_flip = torch.flip(imgs_scaled, dims=[3])
            logits_flip = model(imgs_flip)
            all_probs.append(torch.sigmoid(logits_flip))
    
    # 对所有 TTA 结果取平均
    return torch.stack(all_probs, dim=0).mean(dim=0)


def mixup_data(x, y, alpha=MIXUP_ALPHA):
    if alpha <= 0:
        return x, y, None, None
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class ModelEma:
    """简易 EMA，用于权重平滑"""
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
            for ema_buf, buf in zip(self.ema.buffers(), model.buffers()):
                ema_buf.copy_(buf)

    def to(self, device):
        self.ema.to(device)
        return self

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, state_dict):
        self.ema.load_state_dict(state_dict)


def prepare_data():
    """
    准备数据集，返回 df_labels, train_files, val_files, test_files
    
    采用分层抽样策略，确保稀有类别在验证集和测试集中都有样本
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
    print(f"images 目录中图片数: {len(available_imgs)}")
    print(f"用于训练/验证/测试的样本数(交集): {len(df_labels)}")
    
    # === patient-wise stratified split ===
    PATIENT_COL = "Patient ID"
    
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
    
    # 构建疾病到患者的映射
    disease_to_patients = {cls: [] for cls in RARE_CLASSES}
    for patient, labels in patient_to_labels.items():
        for disease in labels:
            if disease in RARE_CLASSES:
                disease_to_patients[disease].append(patient)
    
    rng = np.random.RandomState(RANDOM_SEED)
    
    # 初始化三个集合
    train_patients = set()
    val_patients = set()
    test_patients = set()
    assigned_patients = set()
    
    # Step 1: 对稀有类别进行分层抽样
    min_samples_per_split = 2
    
    for disease in RARE_CLASSES:
        disease_patients = [p for p in disease_to_patients[disease] if p not in assigned_patients]
        if len(disease_patients) == 0:
            continue
            
        rng.shuffle(disease_patients)
        n_total = len(disease_patients)
        
        n_test = max(min_samples_per_split, int(n_total * TEST_RATIO))
        n_val = max(min_samples_per_split, int(n_total * VAL_RATIO))
        n_train = n_total - n_test - n_val
        
        if n_train < 1:
            if n_total >= 3:
                n_test = 1
                n_val = 1
                n_train = n_total - 2
            elif n_total == 2:
                n_test = 1
                n_val = 1
                n_train = 0
            else:
                n_train = 1
                n_val = 0
                n_test = 0
        
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
    
    return df_labels, train_files, val_files, test_files


def build_criterion(df_labels, model_name=None, stage: int = 1):
    """
    构建损失函数（根据模型和训练阶段）
    
    Args:
        df_labels: 标签数据
        model_name: 模型名称
        stage: 训练阶段 (1 或 2)
    
    Stage 1 策略分配：
    - xrv-pc: 激进策略（原始 pos_weight + 1:1:1 相加）
    - 其他模型: 温和策略（clamp pos_weight + 加权相加）
    
    Stage 2 策略：
    - 使用 STAGE2_LOSS 指定的损失函数（默认 ASL）
    """
    # 计算 pos_weight（温和化处理）
    use_aggressive = (model_name == "xrv-pc")
    
    if use_aggressive:
        pos_weight = compute_pos_weight(
            df_labels, CLASS_NAMES,
            smoothing="none",
        ).to(DEVICE)
    else:
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
            logit_bias = get_logit_adjustment(freq, tau=1.0).to(DEVICE)
            print(f"  [Stage 2 Loss] LogitAdjustedLoss (tau=1.0)")
            return LogitAdjustedLoss(logit_bias=logit_bias)
        elif STAGE2_LOSS == "focal":
            from src.loss_utils import MultiLabelFocalLoss
            print(f"  [Stage 2 Loss] Focal Loss (gamma=2.0)")
            return MultiLabelFocalLoss(gamma=2.0)
        else:
            # 默认使用更激进的 HybridLoss
            print(f"  [Stage 2 Loss] HybridLoss (激进配置)")
            return HybridLoss(
                pos_weight=pos_weight,
                focal_gamma=2.0,
                smooth=0.05,
                bce_weight=1.0,
                focal_weight=1.0,
                smooth_weight=0.5,
            )
    
    # Stage 1: 使用稳定的表征学习损失
    if model_name:
        if use_aggressive:
            strategy_tag = "Stage 1 激进策略 (smoothing=none, weights=1:1:1)"
        else:
            strategy_tag = "Stage 1 温和策略 (smoothing=clamp, weights=1:0.5:0.1)"
        print(f"  [Stage 1 Loss] {model_name} 使用 {strategy_tag}")
    
    if LOSS_MODE == "asl":
        return AsymmetricLoss(gamma_pos=0.0, gamma_neg=4.0, clip=0.05)
    elif LOSS_MODE == "la":
        counts = []
        for cls in CLASS_NAMES:
            c = df_labels["Finding Labels"].str.contains(cls, na=False).sum()
            counts.append(c)
        freq = torch.tensor(counts, dtype=torch.float32)
        logit_bias = get_logit_adjustment(freq, tau=1.0).to(DEVICE)
        return LogitAdjustedLoss(logit_bias=logit_bias)
    else:
        if use_aggressive:
            return HybridLoss(
                pos_weight=pos_weight,
                focal_gamma=2.0,
                smooth=0.05,
                bce_weight=1.0,
                focal_weight=1.0,
                smooth_weight=1.0,
            )
        else:
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
    调整 Stage 2 的学习率
    """
    new_lr = base_lr * STAGE2_LR_FACTOR
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr


def train_single_model(model_name, xrv_weights, df_labels, train_files, val_files, criterion):
    """
    训练单个模型，返回最佳验证集预测和指标
    
    支持的模式：
    1. 标准模式：单分辨率(224)输入
    2. Attention-Guided Crop 模式：双分辨率输入，自动裁剪高分辨率局部区域
    3. 两阶段训练：Stage 1 稳定学习表征，Stage 2 指标对齐微调
    """
    print(f"\n{'='*60}")
    print(f"开始训练模型: {model_name}")
    print(f"预训练权重: {xrv_weights}")
    if USE_ATTENTION_CROP:
        print(f"模式: Attention-Guided Crop (high_res={AG_HIGH_RES}, crops={AG_NUM_CROPS})")
    else:
        print(f"模式: 标准训练")
    if USE_TWO_STAGE:
        print(f"两阶段训练: Stage 1 (epoch 1-{STAGE1_EPOCHS}) -> Stage 2 (epoch {STAGE1_EPOCHS+1}-{NUM_EPOCHS})")
        print(f"  Stage 2 损失: {STAGE2_LOSS}, 学习率因子: {STAGE2_LR_FACTOR}")
    print(f"{'='*60}")
    
    # 根据模式选择数据集和 batch size
    if USE_ATTENTION_CROP:
        # 使用多分辨率数据集
        train_dataset = ChestXrayMultiResDatasetV2(
            file_list=train_files,
            df_labels=df_labels,
            train=True,
            low_res=224,
            high_res=AG_HIGH_RES,
        )
        val_dataset = ChestXrayMultiResDatasetV2(
            file_list=val_files,
            df_labels=df_labels,
            train=False,
            low_res=224,
            high_res=AG_HIGH_RES,
        )
        batch_size = AG_BATCH_SIZE
    else:
        # 标准单分辨率数据集
        train_transform = get_xrv_transforms(train=True)
        val_transform = get_xrv_transforms(train=False)
        
        train_dataset = ChestXrayDataset(
            file_list=train_files,
            df_labels=df_labels,
            transform=train_transform,
        )
        val_dataset = ChestXrayDataset(
            file_list=val_files,
            df_labels=df_labels,
            transform=val_transform,
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
    
    # 构建基础模型
    base_model = TorchXRayVisionDenseNet(
        num_classes=len(CLASS_NAMES),
        weights=xrv_weights,
        freeze_backbone=FREEZE_BACKBONE,
    )
    
    # 如果启用 Attention-Guided Crop，包装基础模型
    if USE_ATTENTION_CROP:
        model = wrap_model_with_attention_crop(
            base_model,
            backbone_type="xrv",
            num_classes=len(CLASS_NAMES),
            feature_dim=1024,
            num_crops=AG_NUM_CROPS,
            crop_size=AG_CROP_SIZE,
            high_res_size=AG_HIGH_RES,
            fusion_type=AG_FUSION_TYPE,
            share_backbone=AG_SHARE_BACKBONE,
        ).to(DEVICE)
        print(f"  >> 已包装为 AttentionGuidedWrapper")
    else:
        model = base_model.to(DEVICE)
    
    # 优化器和调度器
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
    
    ema = ModelEma(model, decay=0.999) if USE_EMA else None
    
    best_val_score = -1.0
    best_thresholds = None
    best_y_true = None
    best_y_pred_prob = None
    model_save_name = f"densenet121_{model_name.replace('-', '_')}"
    best_model_path = os.path.join(SAVE_DIR, f"{model_save_name}_best.pth")
    no_improve_epochs = 0
    
    # 两阶段训练状态
    current_stage = 1
    stage2_criterion = None  # Stage 2 的损失函数（延迟创建）
    
    # 训练循环
    for epoch in range(1, NUM_EPOCHS + 1):
        # ========== 检查是否需要切换到 Stage 2 ==========
        if USE_TWO_STAGE and epoch == STAGE1_EPOCHS + 1 and current_stage == 1:
            current_stage = 2
            print(f"\n{'='*60}")
            print(f"[{model_name}] 切换到 Stage 2 (指标对齐微调)")
            print(f"{'='*60}")
            
            # 1. 创建 Stage 2 损失函数
            stage2_criterion = build_criterion(df_labels, model_name=model_name, stage=2)
            
            # 2. 调整学习率
            new_lr = adjust_learning_rate_for_stage2(optimizer, LR)
            print(f"  学习率: {LR} -> {new_lr}")
            
            # 3. 可选：重置 EMA
            if STAGE2_RESET_EMA and ema is not None:
                ema = ModelEma(model, decay=0.999)
                print(f"  EMA 已重置")
            
            # 4. 重置早停计数器（Stage 2 重新开始计数）
            no_improve_epochs = 0
            print(f"  早停计数器已重置")
            print(f"{'='*60}\n")
        
        # 选择当前阶段的损失函数
        current_criterion = stage2_criterion if current_stage == 2 else criterion
        
        # 是否使用 Mixup（Stage 2 可以禁用）
        use_mixup_this_epoch = MIXUP_ALPHA > 0
        if USE_TWO_STAGE and current_stage == 2 and STAGE2_DISABLE_MIXUP:
            use_mixup_this_epoch = False
        
        model.train()
        train_loss_sum = 0.0
        
        # 训练进度描述
        stage_tag = f"S{current_stage}" if USE_TWO_STAGE else ""
        pbar = tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch}/{NUM_EPOCHS} {stage_tag} [Train]")
        
        for batch in pbar:
            # 根据数据集类型解包数据
            if USE_ATTENTION_CROP:
                imgs_low, imgs_high, labels, _ = batch
                imgs_low = imgs_low.to(DEVICE)
                imgs_high = imgs_high.to(DEVICE)
                labels = labels.to(DEVICE)
                batch_size = imgs_low.size(0)
            else:
                imgs, labels, _ = batch
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)
                batch_size = imgs.size(0)
            
            optimizer.zero_grad()
            
            # Attention-Guided 模式不使用 Mixup
            if USE_ATTENTION_CROP:
                logits = model(imgs_low, imgs_high)
                loss = current_criterion(logits, labels)
            elif use_mixup_this_epoch:
                imgs_m, y_a, y_b, lam = mixup_data(imgs, labels, alpha=MIXUP_ALPHA)
                if lam is not None:
                    logits = model(imgs_m)
                    loss = lam * current_criterion(logits, y_a) + (1 - lam) * current_criterion(logits, y_b)
                else:
                    logits = model(imgs)
                    loss = current_criterion(logits, labels)
            else:
                logits = model(imgs)
                loss = current_criterion(logits, labels)
            
            loss.backward()
            if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            if ema:
                ema.update(model)
            
            train_loss_sum += loss.item() * batch_size
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "stage": current_stage})
        
        scheduler.step()
        avg_train_loss = train_loss_sum / len(train_dataset)
        
        # 验证（带 TTA，或 Attention-Guided 模式）
        model.eval()
        eval_model = ema.ema if ema is not None else model
        all_labels = []
        all_probs = []
        
        if USE_ATTENTION_CROP:
            val_desc = f"[{model_name}] Epoch {epoch}/{NUM_EPOCHS} [Val AG-Crop]"
        elif USE_TTA:
            val_desc = f"[{model_name}] Epoch {epoch}/{NUM_EPOCHS} [Val TTA: scales={TTA_SCALES}, hflip={TTA_HFLIP}]"
        else:
            val_desc = f"[{model_name}] Epoch {epoch}/{NUM_EPOCHS} [Val]"
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=val_desc):
                if USE_ATTENTION_CROP:
                    # Attention-Guided 模式
                    imgs_low, imgs_high, labels, _ = batch
                    imgs_low = imgs_low.to(DEVICE)
                    imgs_high = imgs_high.to(DEVICE)
                    logits = eval_model(imgs_low, imgs_high)
                    probs = torch.sigmoid(logits)
                else:
                    # 标准模式（带 TTA）
                    imgs, labels, _ = batch
                    imgs = imgs.to(DEVICE)
                    probs = inference_with_tta(eval_model, imgs)
                
                all_labels.append(labels.numpy())
                all_probs.append(probs.cpu().numpy())
        
        y_true = np.concatenate(all_labels, axis=0)
        y_pred_prob = np.concatenate(all_probs, axis=0)
        
        auc_macro, f1_macro, f1_weighted = compute_metrics(y_true, y_pred_prob, thresholds=None)
        thresholds = search_best_thresholds(y_true, y_pred_prob)
        auc_macro_t, f1_macro_t, f1_weighted_t = compute_metrics(y_true, y_pred_prob, thresholds)
        
        # 打印结果（包含阶段信息）
        # 注意：AUC 是阈值无关的指标
        stage_info = f" [Stage {current_stage}]" if USE_TWO_STAGE else ""
        print(f"\n[{model_name}] Epoch {epoch}{stage_info}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val AUC:                  {auc_macro:.4f}")
        print(f"  Val Macro-F1 (thr=0.5):   {f1_macro:.4f}")
        print(f"  Val Macro-F1 (best thr):  {f1_macro_t:.4f}")
        # print(f"  Val Weighted-F1 (best thr):{f1_weighted_t:.4f}")
        if USE_TWO_STAGE:
            print(f"  Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        val_score = f1_macro_t
        if val_score > best_val_score:
            best_val_score = val_score
            best_thresholds = thresholds
            best_y_true = y_true
            best_y_pred_prob = y_pred_prob
            no_improve_epochs = 0
            
            torch.save({
                "model_state": ema.state_dict() if ema is not None else model.state_dict(),
                "thresholds": best_thresholds,
                "model_type": model_name,
                "epoch": epoch,
                "stage": current_stage if USE_TWO_STAGE else 1,
                "auc": auc_macro_t,
                "f1": f1_macro_t,
                "two_stage_config": {
                    "enabled": USE_TWO_STAGE,
                    "stage1_epochs": STAGE1_EPOCHS,
                    "stage2_loss": STAGE2_LOSS,
                } if USE_TWO_STAGE else None,
            }, best_model_path)
            
            # 保存验证集预测
            np.savez(
                os.path.join(SAVE_DIR, f"val_preds_{model_save_name}.npz"),
                y_true=y_true,
                y_pred_prob=y_pred_prob,
            )
            print(f"  >> 新的最优模型已保存: {best_model_path}")
        else:
            no_improve_epochs += 1
        
        if EARLY_STOP and epoch > WARMUP_EPOCHS and no_improve_epochs >= EARLY_STOP_PATIENCE:
            print(f"\n[{model_name}] 早停: {EARLY_STOP_PATIENCE} 轮无提升 (best F1={best_val_score:.4f})")
            break
    
    print(f"\n[{model_name}] 训练完成，最佳 F1 = {best_val_score:.4f}")
    
    return {
        "model_name": model_name,
        "y_true": best_y_true,
        "y_pred_prob": best_y_pred_prob,
        "thresholds": best_thresholds,
        "best_f1": best_val_score,
    }


def search_ensemble_weights_for_f1(model_results, top_k=50):
    """
    两阶段搜索最优的集成权重（针对 F1 指标）
    
    - 阶段 1: 固定阈值(0.5)快速筛选 Top-K 权重组合
    - 阶段 2: 对 Top-K 进行阈值搜索，找最优
    
    Args:
        model_results: 各模型的预测结果字典
        top_k: 阶段 1 保留的候选数量，默认 50（XRV 模型较少，50 足够）
    """
    print(f"\n{'='*60}")
    print("搜索 F1 最优集成权重（两阶段策略）...")
    print(f"{'='*60}")
    
    model_names = list(model_results.keys())
    n_models = len(model_names)
    y_true = model_results[model_names[0]]["y_true"]
    
    # 收集所有模型的预测
    all_preds = []
    for name in model_names:
        all_preds.append(model_results[name]["y_pred_prob"])
    all_preds = np.stack(all_preds, axis=0)  # (n_models, n_samples, n_classes)
    
    # 生成权重候选（步长 0.1）
    weight_candidates = np.linspace(0.0, 1.0, 11)
    total_combinations = len(weight_candidates) ** n_models
    
    # ========== 阶段 1: 粗筛（固定阈值 0.5）==========
    print(f"\n[阶段 1] 粗筛: 固定阈值搜索 {total_combinations} 个权重组合...")
    
    candidates = []  # 存储 (score, weights) 元组
    
    for weights in itertools.product(weight_candidates, repeat=n_models):
        if sum(weights) == 0:
            continue
        
        # 归一化权重
        w = np.array(weights)
        w = w / w.sum()
        
        # 加权平均
        ensemble_pred = np.tensordot(w, all_preds, axes=1)
        
        # 固定阈值 0.5，快速计算
        _, f1_macro, _ = compute_metrics(y_true, ensemble_pred, thresholds=None)
        
        candidates.append((f1_macro, w.copy()))
    
    # 按得分排序，保留 Top-K
    candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = candidates[:top_k]
    
    print(f"[阶段 1 完成] Top-{top_k} 粗筛得分范围: {top_candidates[-1][0]:.4f} ~ {top_candidates[0][0]:.4f}")
    
    # ========== 阶段 2: 精选（阈值搜索）==========
    print(f"\n[阶段 2] 精选: 对 Top-{top_k} 候选进行阈值搜索...")
    
    best_f1_macro = -1.0
    best_auc = -1.0
    best_f1_weighted = -1.0
    best_weights = None
    best_ensemble_pred = None
    best_thresholds = None
    
    for coarse_score, w in top_candidates:
        ensemble_pred = np.tensordot(w, all_preds, axes=1)
        
        # 完整阈值搜索
        thresholds = search_best_thresholds(y_true, ensemble_pred)
        auc, f1_macro, f1_weighted = compute_metrics(y_true, ensemble_pred, thresholds)
        
        if f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
            best_f1_weighted = f1_weighted
            best_auc = auc
            best_weights = w
            best_ensemble_pred = ensemble_pred
            best_thresholds = thresholds
    
    print(f"[阶段 2 完成] 最佳 F1 = {best_f1_macro:.4f}")
    print(f"  粗筛最佳 (thr=0.5) = {top_candidates[0][0]:.4f}")
    print(f"  精选提升 = +{best_f1_macro - top_candidates[0][0]:.4f}")
    
    # 同时记录阈值=0.5时的指标（用于对比）
    ensemble_pred_best = np.tensordot(best_weights, all_preds, axes=1)
    auc_thr05, f1_macro_thr05, f1_weighted_thr05 = compute_metrics(y_true, ensemble_pred_best, thresholds=None)
    
    return {
        "weights": best_weights,
        "model_names": model_names,
        "y_true": y_true,
        "y_pred_prob": best_ensemble_pred,
        "thresholds": best_thresholds,
        "auc_thr05": auc_thr05,
        "f1_macro_thr05": f1_macro_thr05,
        "f1_weighted_thr05": f1_weighted_thr05,
        "auc_best_thr": best_auc,
        "f1_macro_best_thr": best_f1_macro,
        "f1_weighted_best_thr": best_f1_weighted,
    }


def search_auc_greedy_ensemble(model_results, pc_key="xrv-pc", chex_key="xrv-chex"):
    """
    AUC 贪婪集成：只用 PC 和 CheXpert 两个模型
    搜索 PC 的占比从 0.5 到 1.0，找出最佳 AUC 的比例
    
    返回: dict 包含最佳配比和指标
    """
    print(f"\n{'='*60}")
    print("AUC 贪婪集成：PC + CheXpert 最佳比例搜索")
    print(f"{'='*60}")
    
    # 检查模型是否存在
    if pc_key not in model_results:
        print(f"[警告] 未找到 {pc_key} 模型，跳过 AUC 贪婪集成")
        return None
    if chex_key not in model_results:
        print(f"[警告] 未找到 {chex_key} 模型，跳过 AUC 贪婪集成")
        return None
    
    y_true = model_results[pc_key]["y_true"]
    y_pred_pc = model_results[pc_key]["y_pred_prob"]
    y_pred_chex = model_results[chex_key]["y_pred_prob"]
    
    # 先打印单模型 AUC
    auc_pc, _, _ = compute_metrics(y_true, y_pred_pc, thresholds=None)
    auc_chex, _, _ = compute_metrics(y_true, y_pred_chex, thresholds=None)
    print(f"\n单模型 AUC:")
    print(f"  {pc_key}: AUC = {auc_pc:.4f}")
    print(f"  {chex_key}: AUC = {auc_chex:.4f}")
    
    # 搜索 PC 占比从 0.5 到 1.0（步长 0.01）
    print(f"\n搜索 PC 占比 [0.50, 1.00]，步长 0.01...")
    
    best_ratio = 0.5
    best_auc = -1.0
    best_probs = None
    
    results = []
    
    for pc_ratio in np.arange(0.50, 1.01, 0.01):
        chex_ratio = 1.0 - pc_ratio
        
        # 加权融合
        y_pred_ens = pc_ratio * y_pred_pc + chex_ratio * y_pred_chex
        
        # 计算 AUC
        auc_ens, _, _ = compute_metrics(y_true, y_pred_ens, thresholds=None)
        
        results.append((pc_ratio, auc_ens))
        
        if auc_ens > best_auc:
            best_auc = auc_ens
            best_ratio = pc_ratio
            best_probs = y_pred_ens
    
    # 打印搜索结果
    print(f"\n比例搜索结果 (PC : CheXpert):")
    print("-" * 50)
    for pc_ratio, auc_val in results:
        # 每 0.1 或最优点打印
        if abs(pc_ratio * 100 % 10) < 0.5 or abs(pc_ratio - best_ratio) < 0.005:
            marker = " <-- BEST" if abs(pc_ratio - best_ratio) < 0.005 else ""
            print(f"  PC={pc_ratio:.2f}, CheX={1-pc_ratio:.2f}  =>  AUC = {auc_val:.4f}{marker}")
    print("-" * 50)
    
    print(f"\n>>> AUC 最佳配比: PC = {best_ratio:.2f}, CheXpert = {1-best_ratio:.2f}")
    print(f">>> 最佳 AUC = {best_auc:.4f}")
    print(f">>> 相比单独 PC 提升: {(best_auc - auc_pc) * 100:.2f}%")
    print(f">>> 相比单独 CheXpert 提升: {(best_auc - auc_chex) * 100:.2f}%")
    
    # 对最优集成搜索阈值
    best_thresholds = search_best_thresholds(y_true, best_probs)
    final_auc, final_f1_macro, final_f1_weighted = compute_metrics(y_true, best_probs, best_thresholds)
    
    return {
        "pc_ratio": best_ratio,
        "chex_ratio": 1 - best_ratio,
        "model_names": [pc_key, chex_key],
        "y_true": y_true,
        "y_pred_prob": best_probs,
        "thresholds": best_thresholds,
        "best_auc": best_auc,
        "auc_pc_only": auc_pc,
        "auc_chex_only": auc_chex,
        "auc_best_thr": final_auc,
        "f1_macro_best_thr": final_f1_macro,
        "f1_weighted_best_thr": final_f1_weighted,
    }


def train_ensemble():
    """
    集成学习主函数：训练所有模型并进行加权平均集成
    
    集成策略：
    - F1 计算：使用所有模型的加权平均（排除 mimic_nb 和 mimic_ch 后的效果可能更好）
    - AUC 计算：使用 PC + CheXpert 的贪婪集成，搜索最佳比例
    """
    # 设置日志记录，所有 print 输出同时保存到 logs/ 目录
    setup_logger("train_xrv_ensemble")
    
    print("=" * 60)
    print("ChestX-ray14 多标签分类 - 集成学习")
    print("=" * 60)
    print(f"将训练 {len(ENSEMBLE_MODELS)} 个模型:")
    for name, weights in ENSEMBLE_MODELS.items():
        print(f"  - {name}: {weights}")
    print("=" * 60)
    
    # 准备数据
    df_labels, train_files, val_files, test_files = prepare_data()
    
    # 保存数据集划分信息
    split_info_path = os.path.join(SAVE_DIR, "dataset_split.npz")
    np.savez(
        split_info_path,
        train_files=np.array(train_files),
        val_files=np.array(val_files),
        test_files=np.array(test_files),
    )
    print(f"\n数据集划分信息已保存到: {split_info_path}")
    
    # 训练所有模型（每个模型使用对应的 loss 策略）
    model_results = {}
    for model_name, xrv_weights in ENSEMBLE_MODELS.items():
        # 为每个模型单独构建 loss（PC 用激进策略，其他用温和策略）
        criterion = build_criterion(df_labels, model_name=model_name)
        
        result = train_single_model(
            model_name, xrv_weights,
            df_labels, train_files, val_files,
            criterion
        )
        model_results[model_name] = result
    
    # 打印单模型结果
    print(f"\n{'='*60}")
    print("单模型结果汇总:")
    print(f"{'='*60}")
    for name, result in model_results.items():
        print(f"  {name}: F1 = {result['best_f1']:.4f}")
    
    # ========== Part 1: F1 集成（使用所有模型） ==========
    print(f"\n{'='*60}")
    print("Part 1: F1 集成（使用所有模型加权平均）")
    print(f"{'='*60}")
    f1_ensemble_result = search_ensemble_weights_for_f1(model_results)
    
    # 打印 F1 集成结果
    print(f"\n各模型权重 (F1 集成):")
    for name, w in zip(f1_ensemble_result["model_names"], f1_ensemble_result["weights"]):
        print(f"  {name}: {w:.3f}")
    
    print(f"\nF1 集成预测指标 (阈值=0.5):")
    print(f"  AUC:         {f1_ensemble_result['auc_thr05']:.4f}")
    print(f"  Macro-F1:    {f1_ensemble_result['f1_macro_thr05']:.4f}")
    # print(f"  Weighted-F1: {f1_ensemble_result['f1_weighted_thr05']:.4f}")
    
    print(f"\nF1 集成预测指标 (最优阈值):")
    print(f"  AUC:         {f1_ensemble_result['auc_best_thr']:.4f}")
    print(f"  Macro-F1:    {f1_ensemble_result['f1_macro_best_thr']:.4f}")
    # print(f"  Weighted-F1: {f1_ensemble_result['f1_weighted_best_thr']:.4f}")
    
    # ========== Part 2: AUC 贪婪集成（只用 PC + CheXpert） ==========
    print(f"\n{'='*60}")
    print("Part 2: AUC 贪婪集成（只用 PC + CheXpert）")
    print(f"{'='*60}")
    auc_ensemble_result = search_auc_greedy_ensemble(model_results, pc_key="xrv-pc", chex_key="xrv-chex")
    
    # ========== 保存结果 ==========
    # 保存 F1 集成结果
    np.savez(
        os.path.join(SAVE_DIR, "ensemble_result_f1.npz"),
        weights=f1_ensemble_result["weights"],
        model_names=np.array(f1_ensemble_result["model_names"]),
        y_true=f1_ensemble_result["y_true"],
        y_pred_prob=f1_ensemble_result["y_pred_prob"],
        thresholds=f1_ensemble_result["thresholds"],
    )
    print(f"\nF1 集成结果已保存到: {os.path.join(SAVE_DIR, 'ensemble_result_f1.npz')}")
    
    # 保存 AUC 集成结果
    if auc_ensemble_result is not None:
        np.savez(
            os.path.join(SAVE_DIR, "ensemble_result_auc.npz"),
            pc_ratio=auc_ensemble_result["pc_ratio"],
            chex_ratio=auc_ensemble_result["chex_ratio"],
            model_names=np.array(auc_ensemble_result["model_names"]),
            y_true=auc_ensemble_result["y_true"],
            y_pred_prob=auc_ensemble_result["y_pred_prob"],
            thresholds=auc_ensemble_result["thresholds"],
            best_auc=auc_ensemble_result["best_auc"],
        )
        print(f"AUC 集成结果已保存到: {os.path.join(SAVE_DIR, 'ensemble_result_auc.npz')}")
    
    # 保存权重配置（综合两种策略）
    import json
    weights_info = {
        "f1_ensemble": {
            "weights": {name: float(w) for name, w in zip(f1_ensemble_result["model_names"], f1_ensemble_result["weights"])},
            "auc": float(f1_ensemble_result["auc_best_thr"]),
            "macro_f1": float(f1_ensemble_result["f1_macro_best_thr"]),
            "weighted_f1": float(f1_ensemble_result["f1_weighted_best_thr"]),
            "thresholds": f1_ensemble_result["thresholds"].tolist(),
        },
    }
    
    if auc_ensemble_result is not None:
        weights_info["auc_ensemble"] = {
            "pc_ratio": float(auc_ensemble_result["pc_ratio"]),
            "chex_ratio": float(auc_ensemble_result["chex_ratio"]),
            "best_auc": float(auc_ensemble_result["best_auc"]),
            "auc_pc_only": float(auc_ensemble_result["auc_pc_only"]),
            "auc_chex_only": float(auc_ensemble_result["auc_chex_only"]),
            "thresholds": auc_ensemble_result["thresholds"].tolist(),
        }
    
    with open(os.path.join(SAVE_DIR, "ensemble_weights.json"), "w") as f:
        json.dump(weights_info, f, indent=2, ensure_ascii=False)
    print(f"权重配置已保存到: {os.path.join(SAVE_DIR, 'ensemble_weights.json')}")
    
    # ========== 最终汇总 ==========
    print(f"\n{'='*60}")
    print("集成策略汇总")
    print(f"{'='*60}")
    print("F1 计算: 使用所有模型的加权平均")
    print(f"  -> 最佳 Macro-F1 = {f1_ensemble_result['f1_macro_best_thr']:.4f}")
    
    if auc_ensemble_result is not None:
        print(f"\nAUC 计算: 使用 PC({auc_ensemble_result['pc_ratio']:.2f}) + CheXpert({auc_ensemble_result['chex_ratio']:.2f})")
        print(f"  -> 最佳 AUC = {auc_ensemble_result['best_auc']:.4f}")
    print("=" * 60)
    
    # 关闭日志记录
    close_logger()
    
    return {
        "f1_ensemble": f1_ensemble_result,
        "auc_ensemble": auc_ensemble_result,
    }


if __name__ == "__main__":
    train_ensemble()
