# metrics_utils.py

from typing import Tuple, Optional

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


def compute_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """
    y_true: (N, C) 0/1
    y_pred_prob: (N, C) 概率值(0~1)
    thresholds: (C,) 每个类别的阈值；如果为 None，则统一用 0.5
    返回：macro AUC, macro F1, weighted F1
    """
    if thresholds is None:
        thresholds = np.full(y_pred_prob.shape[1], 0.5, dtype=np.float32)

    # AUC（可能会因为某个类别只有单一标签而报错，做个保护）
    try:
        auc_macro = roc_auc_score(y_true, y_pred_prob, average="macro")
    except ValueError:
        auc_macro = float("nan")

    # F1 需要先二值化
    y_pred_bin = (y_pred_prob >= thresholds[None, :]).astype(int)
    f1_macro = f1_score(y_true, y_pred_bin, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred_bin, average="weighted", zero_division=0)

    return auc_macro, f1_macro, f1_weighted


def search_best_thresholds(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold_candidates=None,
) -> np.ndarray:
    """
    针对每个类别，遍历一组候选阈值，找到使该类别 F1 最大的阈值。
    返回 shape=(C,) 的最佳阈值数组。
    """
    num_classes = y_true.shape[1]
    if threshold_candidates is None:
        threshold_candidates = np.linspace(0.1, 0.9, 17)  # 0.1,0.15,...,0.9

    best_thresholds = np.full(num_classes, 0.5, dtype=np.float32)

    for c in range(num_classes):
        best_f1 = -1.0
        for t in threshold_candidates:
            y_true_c = y_true[:, c]
            y_pred_c = (y_pred_prob[:, c] >= t).astype(int)
            f1_c = f1_score(y_true_c, y_pred_c, zero_division=0)
            if f1_c > best_f1:
                best_f1 = f1_c
                best_thresholds[c] = t

    return best_thresholds
