import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  # 新增

def compute_pos_weight(
    df_labels,
    class_names,
    multi_label_col="Finding Labels",
    smoothing: str = "none",
    clamp_max: float = 20.0,
):
    """
    自动计算每类的 pos_weight，并支持温和化处理.

    情况 A：如果 df_labels 中已经有 14 个 one-hot 列（列名和 class_names 一致），
            就直接按列求和统计正例数；
    情况 B：否则，假设多标签信息在一列字符串中（默认叫 multi_label_col），
            例如 'Atelectasis|Effusion'，则通过字符串匹配统计每一类的正例数。

    参数:
        df_labels: 包含标签信息的 DataFrame
        class_names: 类别名称列表
        multi_label_col: 多标签字符串列名
        smoothing: 温和化策略，可选值:
            - "none": 不做处理，直接使用 neg/pos
            - "clamp": 裁剪到 [1.0, clamp_max] 范围
            - "sqrt": 开方，更平滑地压缩极端值
            - "log": 对数变换，压缩幅度更大
        clamp_max: 当 smoothing="clamp" 时的上限值，推荐 10/20/50

    返回:
        pos_weight: shape (C,) 的权重张量
    """
    df_cols = set(df_labels.columns)

    # 情况 A：已有 one-hot 列
    if set(class_names).issubset(df_cols):
        pos = df_labels[class_names].sum(axis=0).values.astype(float)
    else:
        # 情况 B：通过多标签字符串列统计
        if multi_label_col not in df_cols:
            raise ValueError(
                f"既找不到 one-hot 标签列，也找不到多标签列 '{multi_label_col}'，"
                f"请检查 CSV 中的列名。当前列有：{list(df_labels.columns)}"
            )

        counts = []
        # 这里假设标签列是用 '|' 分隔，例如 'Atelectasis|Effusion' 或 'No Finding'
        for cls in class_names:
            # contains(..., na=False) 可以避免 NaN
            c = df_labels[multi_label_col].str.contains(cls, na=False).sum()
            counts.append(c)
        pos = np.array(counts, dtype=float)

    neg = len(df_labels) - pos
    pos_weight = torch.tensor(neg / (pos + 1e-6), dtype=torch.float32)

    # 温和化处理：防止极端权重导致训练不稳定
    if smoothing == "clamp":
        # 裁剪策略：直接限制最大值
        pos_weight = torch.clamp(pos_weight, 1.0, clamp_max)
    elif smoothing == "sqrt":
        # 开方策略：更平滑地压缩，保留类别间相对关系
        pos_weight = torch.sqrt(pos_weight)
    elif smoothing == "log":
        # 对数策略：压缩幅度更大，适合极端不平衡
        pos_weight = torch.log1p(pos_weight)
    elif smoothing != "none":
        raise ValueError(f"不支持的 smoothing 策略: {smoothing}，可选: none/clamp/sqrt/log")

    return pos_weight


def compute_effective_num_weight(
    df_labels,
    class_names,
    beta: float = 0.9999,
    multi_label_col: str = "Finding Labels",
):
    """
    Class-Balanced Loss 权重 (Effective Number of Samples).
    返回 shape=(C,) 的权重，可用于样本 reweight 或 logit adjustment.
    """
    df_cols = set(df_labels.columns)
    if set(class_names).issubset(df_cols):
        pos = df_labels[class_names].sum(axis=0).values.astype(float)
    else:
        counts = []
        for cls in class_names:
            c = df_labels[multi_label_col].str.contains(cls, na=False).sum()
            counts.append(c)
        pos = np.array(counts, dtype=float)

    # 有效样本数
    effective_num = 1.0 - np.power(beta, pos)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def get_logit_adjustment(class_freq: torch.Tensor, tau: float = 1.0):
    """
    计算 logit adjustment 偏置：log(p_c) / tau.
    class_freq: shape (C,)，可以是正例计数或频率。
    """
    probs = class_freq / (class_freq.sum() + 1e-8)
    logit_adj = torch.log(probs + 1e-8) / tau
    return logit_adj


# ============================
# 2. 多标签 Focal Loss
# ============================
class MultiLabelFocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: (B, C) raw logits
        targets: (B, C) 0/1 labels
        """
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        prob = torch.sigmoid(logits)
        pt = torch.where(targets == 1, prob, 1 - prob)
        focal_weight = (1 - pt) ** self.gamma

        loss = self.alpha * focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ============================
# 3. Asymmetric Loss (ASL)
# ============================
class AsymmetricLoss(nn.Module):
    """
    实现 Asymmetric Loss (ASL)，对正/负样本使用不同 gamma，并可对负样本做 clip。
    参考论文: Asymmetric Loss For Multi-Label Classification (CVPR 2021)
    """
    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.float()
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        pt_pos = xs_pos * targets
        pt_neg = xs_neg * (1 - targets)
        pt = pt_pos + pt_neg

        asym_weight = torch.pow(1 - pt, self.gamma_pos * targets + self.gamma_neg * (1 - targets))
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = asym_weight * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ============================
# 3. Label Smoothing Loss（多标签版）
# ============================
class MultiLabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        """
        多标签平滑：1 -> 1-smoothing, 0 -> smoothing
        """
        smooth_targets = targets * (1 - self.smoothing) + self.smoothing * 0.5

        return F.binary_cross_entropy_with_logits(logits, smooth_targets)


# ============================
# 4. 组合 Loss：BCE + Focal（论文常用策略）
# ============================
class HybridLoss(nn.Module):
    """
    组合损失：BCE (主损失) + Focal (辅助) + Smooth (正则)
    
    设计原理：
    - BCE 作为主损失，提供稳定的梯度信号
    - Focal 作为辅助项，聚焦难分样本（权重建议 0.3~0.5）
    - Smooth 作为正则项，防止过拟合（权重建议 0.05~0.2）
    
    避免 1:1 直接相加，因为 BCE 和 Focal 本质同源，
    直接相加会让 Focal 对 hard samples 的惩罚过强，影响概率校准。
    """
    def __init__(
        self,
        pos_weight=None,
        focal_gamma=2.0,
        smooth=0.0,
        bce_weight=1.0,
        focal_weight=0.5,
        smooth_weight=0.1,
    ):
        """
        参数:
            pos_weight: BCE 的正例权重
            focal_gamma: Focal Loss 的 gamma 参数
            smooth: Label Smoothing 系数（0 表示不使用）
            bce_weight: BCE 损失的权重系数（默认 1.0，作为主损失）
            focal_weight: Focal 损失的权重系数（默认 0.5，作为辅助项）
            smooth_weight: Smooth 损失的权重系数（默认 0.1，作为正则项）
        """
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.focal = MultiLabelFocalLoss(gamma=focal_gamma)
        self.smooth = MultiLabelSmoothLoss(smoothing=smooth) if smooth > 0 else None
        
        # 损失权重系数
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.smooth_weight = smooth_weight

    def forward(self, logits, targets):
        # BCE 作为主损失
        loss = self.bce_weight * self.bce(logits, targets)
        # Focal 作为辅助项，聚焦难分样本
        loss = loss + self.focal_weight * self.focal(logits, targets)
        # Smooth 作为正则项（可选）
        if self.smooth:
            loss = loss + self.smooth_weight * self.smooth(logits, targets)
        return loss


class LogitAdjustedLoss(nn.Module):
    """
    BCEWithLogitsLoss + logit adjustment (LA) for class imbalance.
    logits 会在进入 BCE 前加上偏置。
    """
    def __init__(self, logit_bias: torch.Tensor, reduction: str = "mean"):
        super().__init__()
        self.register_buffer("logit_bias", logit_bias)
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, logits, targets):
        logits = logits + self.logit_bias
        return self.bce(logits, targets)
