# ensemble_timm.py
"""
Timm 模型集成评估脚本：
- 专门用于 train_timm_models.py 训练的模型
- 支持多种集成策略：简单平均、加权平均、分层加权、Logistic Stacking
- 使用 per-class 独立阈值优化
"""
import os
import itertools

import numpy as np
from sklearn.linear_model import LogisticRegression

from cxr_config import SAVE_DIR, CLASS_NAMES
from metrics_utils import compute_metrics, search_best_thresholds
from log_utils import setup_logger, close_logger


# ========== 模型配置 ==========
# 参与集成的 Timm 模型（与 train_timm_models.py 中的 MODELS_TO_TRAIN 对应）
TIMM_MODELS = [
    "convnext_base_in22k",
    "convnext_base_in1k",
    "convnext_base_in22k_384",
    "convnext_base_in22k_512",
    "convnextv2_base_fcmae_384",
]

# 分层集成配置：按分辨率分组
HIGH_RES_MODELS = [
    "convnext_base_in22k_384",
    "convnext_base_in22k_512",
    "convnextv2_base_fcmae_384",
]
LOW_RES_MODELS = [
    "convnext_base_in22k",
    "convnext_base_in1k"
]


def load_single_model_preds(name: str):
    """加载单个模型的验证集预测"""
    npz_path = os.path.join(SAVE_DIR, f"val_preds_{name}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"找不到验证集预测文件: {npz_path}\n请确认该模型已经训练并保存了 .npz")

    data = np.load(npz_path)
    y_true = data["y_true"]
    y_pred_prob = data["y_pred_prob"]
    
    # 尝试加载训练过程中的最佳指标（新版本保存的文件才有）
    best_metrics = {}
    if "best_auc" in data:
        best_metrics["best_auc"] = float(data["best_auc"])
    if "best_f1_macro" in data:
        best_metrics["best_f1_macro"] = float(data["best_f1_macro"])
    if "best_f1_weighted" in data:
        best_metrics["best_f1_weighted"] = float(data["best_f1_weighted"])
    
    return y_true, y_pred_prob, best_metrics


def evaluate_single_model(name: str):
    """评估单个模型的性能"""
    y_true, y_pred_prob, best_metrics = load_single_model_preds(name)

    # 阈值=0.5
    auc_macro, f1_macro, f1_weighted = compute_metrics(y_true, y_pred_prob, thresholds=None)

    # 搜索每类最佳阈值
    thresholds = search_best_thresholds(y_true, y_pred_prob)
    auc_t, f1_t, f1_weighted_t = compute_metrics(y_true, y_pred_prob, thresholds)

    print(f"\n===== 单模型 [{name}] =====")
    
    # 如果有训练过程中的最佳指标，显示它们
    if best_metrics:
        print(f"  [训练过程最佳]")
        print(f"    Best AUC:         {best_metrics.get('best_auc', 'N/A'):.4f}" if 'best_auc' in best_metrics else "    Best AUC:         N/A")
        print(f"    Best Macro-F1:    {best_metrics.get('best_f1_macro', 'N/A'):.4f}" if 'best_f1_macro' in best_metrics else "    Best Macro-F1:    N/A")
        # print(f"    Best Weighted-F1: {best_metrics.get('best_f1_weighted', 'N/A'):.4f}" if 'best_f1_weighted' in best_metrics else "    Best Weighted-F1: N/A")
    else:
        # 旧版本的 npz 文件没有最佳指标，显示保存时的指标
        print(f"  AUC:                {auc_macro:.4f}")
        print(f"  Macro-F1:           {f1_t:.4f}")
        # print(f"  Weighted-F1:        {f1_weighted_t:.4f}")

    return {
        "name": name,
        "y_true": y_true,
        "y_pred_prob": y_pred_prob,
        "auc": auc_t,
        "f1_macro": f1_t,
        "f1_weighted": f1_weighted_t,
        "thresholds": thresholds,
        "best_metrics": best_metrics,
    }


def simple_average_ensemble(preds_list: list) -> np.ndarray:
    """简单平均集成"""
    stacked = np.stack(preds_list, axis=0)  # (M, N, C)
    return stacked.mean(axis=0)  # (N, C)


# ========== 原始版本（每次都搜索阈值，非常慢）==========
# def weighted_average_ensemble_slow(preds_list: list, y_true: np.ndarray, metric="f1"):
#     """
#     加权平均集成：网格搜索最佳权重（原始版本，每次都搜索阈值）
#     metric: 'f1' 或 'auc'，用于优化的指标
#     
#     注意：此版本对于 5 个模型需要约 10 小时，因为每个权重组合都要搜索阈值。
#     已被 weighted_average_ensemble（两阶段快速版本）替代。
#     """
#     from tqdm import tqdm
#     
#     m = len(preds_list)
#     grid = np.linspace(0.0, 1.0, 11)  # 0.0, 0.1, ..., 1.0
#     
#     # 计算总组合数用于进度条
#     total_combinations = len(grid) ** m
#     
#     best_score = -1.0
#     best_weights = None
#     
#     pbar = tqdm(
#         itertools.product(grid, repeat=m),
#         total=total_combinations,
#         desc=f"加权搜索 ({m}模型, {len(grid)}^{m}={total_combinations}组合)",
#     )
#     
#     for weights in pbar:
#         if sum(weights) == 0:
#             continue
#         w = np.array(weights)
#         w = w / w.sum()  # 归一化
#         
#         ens_prob = np.tensordot(w, np.stack(preds_list, axis=0), axes=1)
#         
#         if metric == "f1":
#             thresholds = search_best_thresholds(y_true, ens_prob)
#             _, score, _ = compute_metrics(y_true, ens_prob, thresholds)
#         else:  # auc
#             score, _, _ = compute_metrics(y_true, ens_prob, thresholds=None)
#         
#         if score > best_score:
#             best_score = score
#             best_weights = w
#             pbar.set_postfix({"best": f"{best_score:.4f}"})
#     
#     if best_weights is None:
#         best_weights = np.ones(m) / m
#     
#     ens_prob = np.tensordot(best_weights, np.stack(preds_list, axis=0), axes=1)
#     return ens_prob, best_weights, best_score


def weighted_average_ensemble(preds_list: list, y_true: np.ndarray, metric="f1", top_k=100):
    """
    两阶段加权平均集成（快速版本）：
    - 阶段 1: 固定阈值(0.5)快速筛选 Top-K 权重组合
    - 阶段 2: 对 Top-K 进行阈值搜索，找最优
    
    相比原始版本，速度提升约 200 倍（10小时 -> 25分钟）
    
    Args:
        preds_list: 各模型的预测概率列表
        y_true: 真实标签
        metric: 'f1' 或 'auc'，用于优化的指标
        top_k: 阶段 1 保留的候选数量，默认 100
    
    Returns:
        (集成概率, 最佳权重, 最佳得分)
    """
    from tqdm import tqdm
    
    m = len(preds_list)
    grid = np.linspace(0.0, 1.0, 11)  # 0.0, 0.1, ..., 1.0
    total_combinations = len(grid) ** m
    stacked_preds = np.stack(preds_list, axis=0)  # 预先堆叠，避免重复计算
    
    # ========== 阶段 1: 粗筛（固定阈值 0.5）==========
    print(f"\n[阶段 1] 粗筛: 固定阈值搜索 {total_combinations} 个权重组合...")
    
    candidates = []  # 存储 (score, weights) 元组
    
    pbar = tqdm(
        itertools.product(grid, repeat=m),
        total=total_combinations,
        desc=f"阶段 1: 粗筛 ({m}模型, {total_combinations}组合)",
    )
    
    for weights in pbar:
        if sum(weights) == 0:
            continue
        w = np.array(weights)
        w = w / w.sum()
        
        ens_prob = np.tensordot(w, stacked_preds, axes=1)
        
        # 固定阈值 0.5，快速计算
        if metric == "f1":
            _, score, _ = compute_metrics(y_true, ens_prob, thresholds=None)
        else:  # auc
            score, _, _ = compute_metrics(y_true, ens_prob, thresholds=None)
        
        candidates.append((score, w.copy()))
        
        # 更新进度条显示当前最佳
        if len(candidates) % 5000 == 0:
            current_best = max(c[0] for c in candidates)
            pbar.set_postfix({"best": f"{current_best:.4f}"})
    
    # 按得分排序，保留 Top-K
    candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = candidates[:top_k]
    
    print(f"\n[阶段 1 完成] Top-{top_k} 粗筛得分范围: {top_candidates[-1][0]:.4f} ~ {top_candidates[0][0]:.4f}")
    
    # ========== 阶段 2: 精选（阈值搜索）==========
    # 对于 AUC 指标，阈值搜索不影响结果，直接返回阶段 1 的最佳
    if metric == "auc":
        best_weights = top_candidates[0][1]
        best_score = top_candidates[0][0]
        ens_prob = np.tensordot(best_weights, stacked_preds, axes=1)
        print(f"[AUC 模式] 与阈值无关，无需阶段 2，最佳 AUC = {best_score:.4f}")
        return ens_prob, best_weights, best_score
    
    print(f"\n[阶段 2] 精选: 对 Top-{top_k} 候选进行阈值搜索...")
    
    best_score = -1.0
    best_weights = None
    
    for i, (coarse_score, w) in enumerate(tqdm(top_candidates, desc="阶段 2: 精选")):
        ens_prob = np.tensordot(w, stacked_preds, axes=1)
        
        # 完整阈值搜索
        thresholds = search_best_thresholds(y_true, ens_prob)
        _, score, _ = compute_metrics(y_true, ens_prob, thresholds)
        
        if score > best_score:
            best_score = score
            best_weights = w
    
    print(f"\n[阶段 2 完成] 最佳 F1 = {best_score:.4f}")
    print(f"  粗筛最佳 (thr=0.5) = {top_candidates[0][0]:.4f}")
    print(f"  精选提升 = +{best_score - top_candidates[0][0]:.4f}")
    
    # 返回最终结果
    ens_prob = np.tensordot(best_weights, stacked_preds, axes=1)
    return ens_prob, best_weights, best_score


def hierarchical_ensemble(
    high_res_preds: list,
    low_res_preds: list,
    y_true: np.ndarray,
    alpha_candidates=None,
):
    """
    分层加权集成：高分辨率组 vs 低分辨率组
    
    ensemble = α × mean(高分辨率组) + (1-α) × mean(低分辨率组)
    
    返回: (best_alpha, best_probs, best_f1, search_results)
    """
    if alpha_candidates is None:
        alpha_candidates = np.linspace(0.0, 1.0, 21)  # 0.0, 0.05, ..., 1.0
    
    # 计算每组的平均概率
    high_res_mean = np.stack(high_res_preds, axis=0).mean(axis=0) if high_res_preds else None
    low_res_mean = np.stack(low_res_preds, axis=0).mean(axis=0) if low_res_preds else None
    
    if high_res_mean is None:
        return 0.0, low_res_mean, -1.0, []  # 只有低分辨率组，无搜索结果
    if low_res_mean is None:
        return 1.0, high_res_mean, -1.0, []  # 只有高分辨率组，无搜索结果
    
    best_alpha = 0.5
    best_f1 = -1.0
    best_probs = None
    
    results = []
    
    for alpha in alpha_candidates:
        ens_prob = alpha * high_res_mean + (1 - alpha) * low_res_mean
        
        # 使用 per-class 阈值优化
        thresholds = search_best_thresholds(y_true, ens_prob)
        _, f1_macro, _ = compute_metrics(y_true, ens_prob, thresholds)
        
        results.append((alpha, f1_macro))
        
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_alpha = alpha
            best_probs = ens_prob
    
    return best_alpha, best_probs, best_f1, results


def logistic_stacking(preds_list: list, y_true: np.ndarray, n_splits: int = 5, C: float = 0.1) -> np.ndarray:
    """
    Logistic Stacking：使用交叉验证避免数据泄露
    
    对每个类别训练一个逻辑回归元分类器，使用 K-Fold 交叉验证
    生成 out-of-fold 预测，避免在同一数据上训练和评估。
    
    Args:
        preds_list: 各模型的预测概率列表
        y_true: 真实标签
        n_splits: K-Fold 的折数
        C: 正则化强度的倒数，值越小正则化越强（默认 0.1，比 sklearn 默认的 1.0 更强）
    """
    from sklearn.model_selection import StratifiedKFold
    
    n, c = preds_list[0].shape
    
    feats = np.stack(preds_list, axis=0).transpose(1, 2, 0)  # (N, C, M)
    stacked_probs = np.zeros((n, c), dtype=np.float32)
    
    for cls in range(c):
        X = feats[:, cls, :]  # (N, M)
        y = y_true[:, cls]
        
        # 检查是否有足够的正负样本
        if y.sum() < 5 or (1 - y).sum() < 5:
            # 样本太少，使用简单平均
            stacked_probs[:, cls] = X.mean(axis=1)
        else:
            # 使用 K-Fold 交叉验证生成 out-of-fold 预测
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            oof_preds = np.zeros(n, dtype=np.float32)
            
            for train_idx, val_idx in skf.split(X, y):
                clf = LogisticRegression(
                    C=C,  # 更强的 L2 正则化
                    max_iter=500, 
                    solver='lbfgs',
                    class_weight="balanced"
                )
                clf.fit(X[train_idx], y[train_idx])
                oof_preds[val_idx] = clf.predict_proba(X[val_idx])[:, 1]
            
            stacked_probs[:, cls] = oof_preds
    
    return stacked_probs


def nonneg_stacking(preds_list: list, y_true: np.ndarray) -> tuple:
    """
    非负最小二乘 Stacking：约束权重 >= 0，更接近加权平均的物理意义
    
    对每个类别独立求解: min ||Xw - y||^2  s.t. w >= 0
    然后归一化权重，使其和为 1。
    
    Args:
        preds_list: 各模型的预测概率列表
        y_true: 真实标签
    
    Returns:
        (stacked_probs, weights_per_class): 集成概率和每个类别的权重矩阵
    """
    from scipy.optimize import nnls
    
    n, c = preds_list[0].shape
    m = len(preds_list)
    feats = np.stack(preds_list, axis=0).transpose(1, 2, 0)  # (N, C, M)
    stacked_probs = np.zeros((n, c), dtype=np.float32)
    weights_per_class = np.zeros((c, m), dtype=np.float32)  # (C, M)
    
    for cls in range(c):
        X = feats[:, cls, :]  # (N, M)
        y = y_true[:, cls].astype(np.float64)
        
        # 非负最小二乘: min ||Xw - y||^2  s.t. w >= 0
        weights, _ = nnls(X, y)
        
        # 归一化权重
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(m) / m
        
        weights_per_class[cls] = weights
        stacked_probs[:, cls] = X @ weights
    
    return stacked_probs, weights_per_class


def per_class_weighted_ensemble(preds_list: list, model_aucs: list, temperature=1.0):
    """
    Per-class 自适应权重集成：根据每个模型在每个类别上的 AUC 分配权重
    
    Args:
        preds_list: 各模型的预测概率列表，每个元素 shape=(N, C)
        model_aucs: 各模型的 per-class AUC 列表，每个元素 shape=(C,)
        temperature: softmax 温度参数，越大权重越均匀
    
    Returns:
        (集成概率, 权重矩阵)
    """
    m = len(preds_list)
    n, c = preds_list[0].shape
    
    stacked = np.stack(preds_list, axis=0)  # (M, N, C)
    
    # 计算 per-class 权重：softmax(AUC / temperature)
    auc_matrix = np.stack(model_aucs, axis=0)  # (M, C)
    weights = np.exp(auc_matrix / temperature)
    weights = weights / weights.sum(axis=0, keepdims=True)  # (M, C) 每列和为1
    
    # 加权融合
    ens_prob = np.zeros((n, c), dtype=np.float32)
    for cls in range(c):
        for model_idx in range(m):
            ens_prob[:, cls] += weights[model_idx, cls] * stacked[model_idx, :, cls]
    
    return ens_prob, weights


def report_results(tag: str, y_true: np.ndarray, y_pred_prob: np.ndarray):
    """报告集成结果"""
    # 阈值=0.5
    auc_macro, f1_macro, _ = compute_metrics(y_true, y_pred_prob, thresholds=None)
    
    # Per-class 最佳阈值
    thresholds = search_best_thresholds(y_true, y_pred_prob)
    _, f1_t, f1_weighted_t = compute_metrics(y_true, y_pred_prob, thresholds)
    
    print(f"\n===== 集成结果 [{tag}] =====")
    print(f"  AUC:                      {auc_macro:.4f}")
    print(f"  Macro-F1 (per-class thr): {f1_t:.4f}")
    # print(f"  Weighted-F1 (per-class thr): {f1_weighted_t:.4f}")
    
    return {
        "tag": tag,
        "auc": auc_macro,
        "f1_macro": f1_t,
        "f1_weighted": f1_weighted_t,
        "thresholds": thresholds,
        "y_pred_prob": y_pred_prob,
    }


def main():
    # 设置日志记录，所有 print 输出同时保存到 logs/ 目录
    setup_logger("ensemble_timm")
    
    print("=" * 70)
    print("Timm 模型集成评估")
    print("=" * 70)
    print(f"SAVE_DIR = {SAVE_DIR}")
    print(f"参与集成的模型: {TIMM_MODELS}")
    print(f"高分辨率组: {HIGH_RES_MODELS}")
    print(f"低分辨率组: {LOW_RES_MODELS}")
    print("=" * 70)
    
    # ========== 1. 加载所有模型预测 ==========
    available_models = []
    model_results = {}
    
    for name in TIMM_MODELS:
        try:
            result = evaluate_single_model(name)
            available_models.append(name)
            model_results[name] = result
        except FileNotFoundError as e:
            print(f"\n[跳过] {name}: {e}")
    
    if len(available_models) < 2:
        print(f"\n[错误] 只有 {len(available_models)} 个模型可用，无法进行集成")
        return
    
    print(f"\n可用模型: {available_models}")
    
    # 获取 y_true 和预测列表
    y_true = model_results[available_models[0]]["y_true"]
    preds_list = [model_results[name]["y_pred_prob"] for name in available_models]
    
    # 验证 y_true 一致性
    for name in available_models[1:]:
        if not np.array_equal(y_true, model_results[name]["y_true"]):
            raise ValueError(f"模型 {available_models[0]} 与 {name} 的 y_true 不一致")
    
    all_results = []
    
    # ========== 1. 加权平均集成 ==========
    print("\n" + "=" * 70)
    print("策略 1: 加权平均（网格搜索）")
    print("=" * 70)
    
    ens_weighted, best_weights, best_score = weighted_average_ensemble(preds_list, y_true, metric="f1")
    print(f"最佳权重: {dict(zip(available_models, best_weights))}")
    result_weighted = report_results("加权平均", y_true, ens_weighted)
    all_results.append(result_weighted)
    
    # ========== 2. 分层加权集成 ==========
    print("\n" + "=" * 70)
    print("策略 2: 分层加权（高分辨率 vs 低分辨率）")
    print("=" * 70)
    
    # 分离高分辨率和低分辨率模型的预测
    high_res_preds = [model_results[name]["y_pred_prob"] 
                      for name in HIGH_RES_MODELS if name in available_models]
    low_res_preds = [model_results[name]["y_pred_prob"] 
                     for name in LOW_RES_MODELS if name in available_models]
    
    if high_res_preds and low_res_preds:
        best_alpha, ens_hier, best_f1, search_results = hierarchical_ensemble(
            high_res_preds, low_res_preds, y_true
        )
        
        print(f"高分辨率组模型: {[n for n in HIGH_RES_MODELS if n in available_models]}")
        print(f"低分辨率组模型: {[n for n in LOW_RES_MODELS if n in available_models]}")
        print(f"\n分层权重搜索结果 (α = 高分辨率组权重):")
        print("-" * 45)
        for alpha, f1 in search_results:
            marker = " <-- BEST" if abs(alpha - best_alpha) < 0.01 else ""
            if abs(alpha * 20 % 2) < 0.1 or marker:  # 每 0.1 打印一次
                print(f"  α={alpha:.2f} (高分辨率={alpha:.0%}, 低分辨率={1-alpha:.0%}) => F1={f1:.4f}{marker}")
        
        result_hier = report_results(f"分层加权 (α={best_alpha:.2f})", y_true, ens_hier)
        all_results.append(result_hier)
    else:
        print("[跳过] 高分辨率或低分辨率组模型不足")
    
    # ========== 3. Logistic Stacking (强正则化) ==========
    print("\n" + "=" * 70)
    print("策略 3: Logistic Stacking (C=0.1, 强正则化)")
    print("=" * 70)
    
    ens_stack = logistic_stacking(preds_list, y_true, C=0.1)
    result_stack = report_results("Logistic Stacking (C=0.1)", y_true, ens_stack)
    all_results.append(result_stack)
    
    # ========== 3b. 非负约束 Stacking ==========
    print("\n" + "=" * 70)
    print("策略 3b: 非负约束 Stacking (NNLS)")
    print("=" * 70)
    
    ens_nnls, nnls_weights = nonneg_stacking(preds_list, y_true)
    
    # 显示每个类别学到的平均权重
    mean_weights = nnls_weights.mean(axis=0)
    print(f"NNLS 平均权重: {dict(zip(available_models, mean_weights.round(3)))}")
    
    result_nnls = report_results("非负约束 Stacking (NNLS)", y_true, ens_nnls)
    all_results.append(result_nnls)
    
    # ========== 4. AUC 优化的加权平均 ==========
    print("\n" + "=" * 70)
    print("策略 4: 加权平均（AUC 优化）")
    print("=" * 70)
    
    ens_auc_opt, best_weights_auc, best_auc_score = weighted_average_ensemble(
        preds_list, y_true, metric="auc"
    )
    print(f"最佳权重 (AUC优化): {dict(zip(available_models, best_weights_auc))}")
    result_auc_opt = report_results("加权平均 (AUC优化)", y_true, ens_auc_opt)
    all_results.append(result_auc_opt)
    
    # ========== 5. Per-class 自适应权重集成 ==========
    print("\n" + "=" * 70)
    print("策略 5: Per-class 自适应权重（基于各模型在每类上的 AUC）")
    print("=" * 70)
    
    # 计算每个模型的 per-class AUC
    from sklearn.metrics import roc_auc_score
    model_per_class_aucs = []
    num_classes = y_true.shape[1]
    
    print("\n各模型 Per-class AUC:")
    for name in available_models:
        y_pred = model_results[name]["y_pred_prob"]
        per_class_auc = np.zeros(num_classes, dtype=np.float32)
        for c in range(num_classes):
            try:
                per_class_auc[c] = roc_auc_score(y_true[:, c], y_pred[:, c])
            except ValueError:
                per_class_auc[c] = 0.5  # 如果某类只有单一标签，默认 0.5
        model_per_class_aucs.append(per_class_auc)
        print(f"  {name}: mean={per_class_auc.mean():.4f}, min={per_class_auc.min():.4f}, max={per_class_auc.max():.4f}")
    
    # 尝试不同的 temperature 参数
    best_temp = 1.0
    best_temp_f1 = -1.0
    best_temp_result = None
    
    for temp in [0.5, 1.0, 2.0]:
        ens_perclass, weights_matrix = per_class_weighted_ensemble(
            preds_list, model_per_class_aucs, temperature=temp
        )
        thresholds = search_best_thresholds(y_true, ens_perclass)
        _, f1_macro, _ = compute_metrics(y_true, ens_perclass, thresholds)
        
        if f1_macro > best_temp_f1:
            best_temp_f1 = f1_macro
            best_temp = temp
            best_temp_result = (ens_perclass, weights_matrix)
    
    print(f"\n最佳 temperature = {best_temp}")
    ens_perclass, weights_matrix = best_temp_result
    result_perclass = report_results(f"Per-class 自适应 (T={best_temp})", y_true, ens_perclass)
    all_results.append(result_perclass)
    
    # ========== 6. 汇总与保存结果 ==========
    print("\n" + "=" * 70)
    print("集成策略汇总")
    print("=" * 70)
    
    print(f"\n{'策略':<30} | {'AUC':>8} | {'Macro-F1':>10}")
    print("-" * 55)
    
    for r in all_results:
        print(f"{r['tag']:<30} | {r['auc']:>8.4f} | {r['f1_macro']:>10.4f}")
    
    # 找到 F1 最佳策略
    best_f1_result = max(all_results, key=lambda x: x["f1_macro"])
    # 找到 AUC 最佳策略
    best_auc_result = max(all_results, key=lambda x: x["auc"])
    
    print("-" * 70)
    print(f"\n>>> F1 最佳策略: {best_f1_result['tag']}")
    print(f">>> Macro-F1 = {best_f1_result['f1_macro']:.4f}, AUC = {best_f1_result['auc']:.4f}")
    print(f"\n>>> AUC 最佳策略: {best_auc_result['tag']}")
    print(f">>> AUC = {best_auc_result['auc']:.4f}, Macro-F1 = {best_auc_result['f1_macro']:.4f}")
    
    # 保存 F1 最佳集成结果（保持原有文件名兼容性）
    save_path_f1 = os.path.join(SAVE_DIR, "val_preds_timm_ensemble.npz")
    np.savez(
        save_path_f1,
        y_true=y_true,
        y_pred_prob=best_f1_result["y_pred_prob"],
        thresholds=best_f1_result["thresholds"],
        model_names=np.array(available_models),
        ensemble_strategy=best_f1_result["tag"],
    )
    print(f"\n>>> F1 最佳集成已保存至: {save_path_f1}")
    
    # 保存 AUC 最佳集成结果（新增）
    save_path_auc = os.path.join(SAVE_DIR, "val_preds_timm_ensemble_auc.npz")
    np.savez(
        save_path_auc,
        y_true=y_true,
        y_pred_prob=best_auc_result["y_pred_prob"],
        thresholds=best_auc_result["thresholds"],
        model_names=np.array(available_models),
        ensemble_strategy=best_auc_result["tag"],
    )
    print(f">>> AUC 最佳集成已保存至: {save_path_auc}")
    
    # 单独保存阈值（方便推理使用）
    threshold_path = os.path.join(SAVE_DIR, "timm_ensemble_thresholds.npy")
    np.save(threshold_path, best_f1_result["thresholds"])
    print(f">>> F1 阈值已保存至: {threshold_path}")
    
    threshold_path_auc = os.path.join(SAVE_DIR, "timm_ensemble_thresholds_auc.npy")
    np.save(threshold_path_auc, best_auc_result["thresholds"])
    print(f">>> AUC 阈值已保存至: {threshold_path_auc}")
    
    # 关闭日志记录
    close_logger()


if __name__ == "__main__":
    main()
