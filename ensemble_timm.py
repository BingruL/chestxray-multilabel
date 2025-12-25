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

from src.cxr_config import SAVE_DIR
from src.metrics_utils import compute_metrics, search_best_thresholds
from src.log_utils import setup_logger, close_logger
from sklearn.metrics import f1_score


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


# ========== 优化工具函数 ==========

def search_best_thresholds_refined(y_true: np.ndarray, y_pred_prob: np.ndarray) -> np.ndarray:
    """
    两阶段阈值搜索：先粗搜，再局部细化
    
    相比原始的 search_best_thresholds（步长 0.05），此方法在最佳阈值附近
    进行更精细的搜索（步长 0.01），可提升约 0.5%~1% 的 F1。
    
    Args:
        y_true: 真实标签 shape=(N, C)
        y_pred_prob: 预测概率 shape=(N, C)
    
    Returns:
        最佳阈值数组 shape=(C,)
    """
    from sklearn.metrics import f1_score
    
    num_classes = y_true.shape[1]
    best_thresholds = np.full(num_classes, 0.5, dtype=np.float32)
    
    # 阶段 1: 粗搜 (0.1-0.9, 步长 0.05)
    coarse_grid = np.linspace(0.1, 0.9, 17)
    
    for c in range(num_classes):
        best_f1 = -1.0
        for t in coarse_grid:
            y_pred_c = (y_pred_prob[:, c] >= t).astype(int)
            f1_c = f1_score(y_true[:, c], y_pred_c, zero_division=0)
            if f1_c > best_f1:
                best_f1 = f1_c
                best_thresholds[c] = t
    
    # 阶段 2: 局部细化 (±0.05 范围，步长 0.01)
    for c in range(num_classes):
        coarse_t = best_thresholds[c]
        fine_grid = np.linspace(max(0.05, coarse_t - 0.05), 
                                 min(0.95, coarse_t + 0.05), 11)
        best_f1 = -1.0
        for t in fine_grid:
            y_pred_c = (y_pred_prob[:, c] >= t).astype(int)
            f1_c = f1_score(y_true[:, c], y_pred_c, zero_division=0)
            if f1_c > best_f1:
                best_f1 = f1_c
                best_thresholds[c] = t
    
    return best_thresholds


def calibrate_predictions(preds_list: list, y_true: np.ndarray, use_cv: bool = False, n_splits: int = 5) -> list:
    """
    概率校准：对每个模型的每个类别进行 Isotonic Regression 校准
    
    不同模型的输出概率尺度可能不一致，校准后可以使概率更有意义，
    提升集成效果，特别是对 Stacking 类方法。
    
    Args:
        preds_list: 各模型的预测概率列表
        y_true: 真实标签
        use_cv: 是否使用交叉验证（避免数据泄露，但可能不稳定）
        n_splits: K-Fold 的折数，默认 5（仅 use_cv=True 时有效）
    
    Returns:
        校准后的预测概率列表
    
    Note:
        - use_cv=False: 在全量数据上校准，有轻微乐观偏差，但更稳定（推荐用于验证集评估）
        - use_cv=True: 使用 StratifiedKFold 生成 OOF 校准，严格无泄露，但对稀有类别不稳定
    """
    from sklearn.isotonic import IsotonicRegression
    
    calibrated = []
    
    if use_cv:
        # 严格无泄露模式：使用 StratifiedKFold
        from sklearn.model_selection import StratifiedKFold
        
        for probs in preds_list:
            n, num_classes = probs.shape
            calib_probs = np.zeros_like(probs, dtype=np.float32)
            
            for c in range(num_classes):
                y_c = y_true[:, c]
                pos_count = y_c.sum()
                neg_count = n - pos_count
                
                # 如果正/负样本太少，跳过校准（保持原始概率）
                min_samples_per_fold = 10
                if pos_count < min_samples_per_fold * n_splits or neg_count < min_samples_per_fold * n_splits:
                    calib_probs[:, c] = probs[:, c]
                    continue
                
                # 使用 StratifiedKFold 确保每个 fold 都有正负样本
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                
                for train_idx, val_idx in skf.split(probs, y_c):
                    ir = IsotonicRegression(out_of_bounds='clip')
                    ir.fit(probs[train_idx, c], y_c[train_idx])
                    calib_probs[val_idx, c] = ir.predict(probs[val_idx, c])
            
            calibrated.append(calib_probs)
    else:
        # 稳定模式：在全量数据上校准（有轻微乐观偏差，但结果更稳定）
        for probs in preds_list:
            calib_probs = np.zeros_like(probs, dtype=np.float32)
            for c in range(probs.shape[1]):
                ir = IsotonicRegression(out_of_bounds='clip')
                ir.fit(probs[:, c], y_true[:, c])
                calib_probs[:, c] = ir.predict(probs[:, c])
            calibrated.append(calib_probs)
    
    return calibrated


def compute_model_diversity(preds_list: list) -> np.ndarray:
    """
    计算模型间的多样性分数
    
    多样性 = 1 - 平均相关性（与其他模型）
    多样性高的模型对集成贡献更大。
    
    Args:
        preds_list: 各模型的预测概率列表
    
    Returns:
        多样性分数数组 shape=(M,)
    """
    m = len(preds_list)
    
    # 展平预测用于计算相关性
    flat_preds = [p.flatten() for p in preds_list]
    corr_matrix = np.corrcoef(flat_preds)  # (M, M)
    
    # 多样性分数 = 1 - 平均相关性（排除自身）
    diversity_scores = np.zeros(m, dtype=np.float32)
    for i in range(m):
        other_corrs = [corr_matrix[i, j] for j in range(m) if j != i]
        diversity_scores[i] = 1 - np.mean(other_corrs)
    
    return diversity_scores


def diversity_weighted_ensemble(preds_list: list, model_aucs: list, diversity_weight: float = 0.3):
    """
    多样性感知的加权集成
    
    综合考虑模型性能和模型间的多样性，避免高度相关的模型主导集成。
    
    综合权重 = (1 - diversity_weight) × 性能权重 + diversity_weight × 多样性权重
    
    Args:
        preds_list: 各模型的预测概率列表
        model_aucs: 各模型的 AUC 分数
        diversity_weight: 多样性在综合权重中的占比，默认 0.3
    
    Returns:
        (集成概率, 综合权重, 性能权重, 多样性分数)
    """
    m = len(preds_list)
    stacked = np.stack(preds_list, axis=0)  # (M, N, C)
    
    # 性能权重（归一化）
    performance_scores = np.array(model_aucs, dtype=np.float32)
    perf_weights = performance_scores / performance_scores.sum()
    
    # 多样性分数（归一化）
    diversity_scores = compute_model_diversity(preds_list)
    div_weights = diversity_scores / diversity_scores.sum()
    
    # 综合权重
    combined_weights = (1 - diversity_weight) * perf_weights + diversity_weight * div_weights
    combined_weights = combined_weights / combined_weights.sum()
    
    # 加权集成
    ens_prob = np.tensordot(combined_weights, stacked, axes=1)
    
    return ens_prob, combined_weights, perf_weights, diversity_scores


def joint_optimize_weights_thresholds(preds_list: list, y_true: np.ndarray, model_names: list = None):
    """
    权重与阈值联合优化：同时优化集成权重和 per-class 阈值
    
    相比先搜权重再搜阈值的两阶段方法，联合优化可以找到更优的组合，
    因为权重和阈值是相互影响的。
    
    Args:
        preds_list: 各模型的预测概率列表
        y_true: 真实标签
        model_names: 模型名称列表（用于日志输出）
    
    Returns:
        (集成概率, 最佳权重, 最佳阈值, 最佳F1)
    """
    from scipy.optimize import minimize
    from sklearn.metrics import f1_score
    
    m = len(preds_list)
    n, c = preds_list[0].shape
    stacked = np.stack(preds_list, axis=0)  # (M, N, C)
    
    def objective(params):
        """目标函数：最小化负 Macro-F1"""
        weights = params[:m]
        thresholds = params[m:]
        
        # 归一化权重（确保非负）
        weights = np.abs(weights)
        if weights.sum() == 0:
            weights = np.ones(m) / m
        else:
            weights = weights / weights.sum()
        
        # 裁剪阈值到合理范围
        thresholds = np.clip(thresholds, 0.1, 0.9)
        
        # 计算集成概率
        ens_prob = np.tensordot(weights, stacked, axes=1)
        
        # 计算 Macro-F1
        y_pred_bin = (ens_prob >= thresholds[None, :]).astype(int)
        f1 = f1_score(y_true, y_pred_bin, average='macro', zero_division=0)
        
        return -f1  # 最小化负 F1
    
    # 初始值：均匀权重 + 0.5 阈值
    x0 = np.concatenate([np.ones(m) / m, np.full(c, 0.5)])
    
    # 使用 Nelder-Mead 优化
    result = minimize(
        objective, 
        x0, 
        method='Nelder-Mead',
        options={'maxiter': 3000, 'xatol': 1e-4, 'fatol': 1e-4}
    )
    
    # 提取最优参数
    best_weights = np.abs(result.x[:m])
    best_weights = best_weights / best_weights.sum()
    best_thresholds = np.clip(result.x[m:], 0.1, 0.9)
    best_f1 = -result.fun
    
    # 计算最终集成概率
    ens_prob = np.tensordot(best_weights, stacked, axes=1)
    
    return ens_prob, best_weights, best_thresholds, best_f1


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


def weighted_average_ensemble(preds_list: list, y_true: np.ndarray, metric="f1", top_k=200):
    """
    三阶段加权平均集成：
    - 阶段 1: 固定阈值(0.5)快速筛选 Top-K 权重组合
    - 阶段 2: 对 Top-K 进行阈值搜索，找最优
    - 阶段 3: 在最佳权重附近进行局部精调
    
    Args:
        preds_list: 各模型的预测概率列表
        y_true: 真实标签
        metric: 'f1' 或 'auc'，用于优化的指标
        top_k: 阶段 1 保留的候选数量，默认 200
    
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
    # 对于 AUC 指标，阈值搜索不影响结果，直接跳到阶段 3
    if metric == "auc":
        stage2_best_weights = top_candidates[0][1]
        stage2_best_score = top_candidates[0][0]
        print(f"[AUC 模式] 与阈值无关，跳过阶段 2，当前最佳 AUC = {stage2_best_score:.4f}")
    else:
        print(f"\n[阶段 2] 精选: 对 Top-{top_k} 候选进行阈值搜索...")
        
        stage2_best_score = -1.0
        stage2_best_weights = None
        
        for i, (coarse_score, w) in enumerate(tqdm(top_candidates, desc="阶段 2: 精选")):
            ens_prob = np.tensordot(w, stacked_preds, axes=1)
            
            # 完整阈值搜索
            thresholds = search_best_thresholds(y_true, ens_prob)
            _, score, _ = compute_metrics(y_true, ens_prob, thresholds)
            
            if score > stage2_best_score:
                stage2_best_score = score
                stage2_best_weights = w
        
        print(f"\n[阶段 2 完成] 最佳 F1 = {stage2_best_score:.4f}")
        print(f"  粗筛最佳 (thr=0.5) = {top_candidates[0][0]:.4f}")
        print(f"  精选提升 = +{stage2_best_score - top_candidates[0][0]:.4f}")
    
    # ========== 阶段 3: 局部精调 ==========
    fine_grid = np.linspace(-0.02, 0.02, 5)  # [-0.02, -0.01, 0, 0.01, 0.02]
    fine_combinations = len(fine_grid) ** m  # 5^5 = 3125
    
    print(f"\n[阶段 3] 局部精调: 在最佳权重 ±0.02 范围内搜索 {fine_combinations} 个组合...")
    
    best_score = stage2_best_score
    best_weights = stage2_best_weights.copy()
    
    pbar = tqdm(
        itertools.product(fine_grid, repeat=m),
        total=fine_combinations,
        desc=f"阶段 3: 局部精调",
    )
    
    improved_count = 0
    for offsets in pbar:
        # 在阶段 2 最佳权重基础上加偏移
        w = stage2_best_weights + np.array(offsets)
        
        # 裁剪到 [0, 1] 范围
        w = np.clip(w, 0, 1)
        
        # 跳过全零权重
        if w.sum() == 0:
            continue
        
        # 归一化
        w = w / w.sum()
        
        # 计算集成概率
        ens_prob = np.tensordot(w, stacked_preds, axes=1)
        
        # 计算分数
        if metric == "f1":
            thresholds = search_best_thresholds(y_true, ens_prob)
            _, score, _ = compute_metrics(y_true, ens_prob, thresholds)
        else:  # auc
            score, _, _ = compute_metrics(y_true, ens_prob, thresholds=None)
        
        if score > best_score:
            best_score = score
            best_weights = w.copy()
            improved_count += 1
            pbar.set_postfix({"best": f"{best_score:.4f}", "improved": improved_count})
    
    print(f"\n[阶段 3 完成] 最佳 {metric.upper()} = {best_score:.4f}")
    print(f"  阶段 2 最佳 = {stage2_best_score:.4f}")
    print(f"  局部精调提升 = +{best_score - stage2_best_score:.4f}")
    print(f"  共找到 {improved_count} 个更优权重组合")
    
    # 返回最终结果
    ens_prob = np.tensordot(best_weights, stacked_preds, axes=1)
    return ens_prob, best_weights, best_score


def hierarchical_ensemble(
    high_res_preds: list,
    low_res_preds: list,
    y_true: np.ndarray,
    high_res_scores: list = None,
    low_res_scores: list = None,
    alpha_candidates=None,
):
    """
    分层加权集成：高分辨率组 vs 低分辨率组
    
    改进版：组内按模型性能加权平均，组间搜索最优 alpha
    
    ensemble = α × weighted_mean(高分辨率组) + (1-α) × weighted_mean(低分辨率组)
    
    Args:
        high_res_preds: 高分辨率组模型的预测概率列表
        low_res_preds: 低分辨率组模型的预测概率列表
        y_true: 真实标签
        high_res_scores: 高分辨率组模型的性能分数（如 AUC），用于组内加权
        low_res_scores: 低分辨率组模型的性能分数
        alpha_candidates: alpha 候选值列表
    
    返回: (best_alpha, best_probs, best_f1, search_results, group_weights)
    """
    if alpha_candidates is None:
        alpha_candidates = np.linspace(0.0, 1.0, 21)  # 0.0, 0.05, ..., 1.0
    
    def weighted_mean(preds, scores=None):
        """组内加权平均：按性能分数加权，若无分数则简单平均"""
        if not preds:
            return None, None
        
        stacked = np.stack(preds, axis=0)  # (M, N, C)
        
        if scores is None or len(scores) == 0:
            # 无分数信息，使用简单平均
            weights = np.ones(len(preds)) / len(preds)
        else:
            # 按性能分数加权（归一化）
            weights = np.array(scores, dtype=np.float32)
            weights = weights / weights.sum()
        
        # 加权平均
        result = np.tensordot(weights, stacked, axes=1)  # (N, C)
        return result, weights
    
    # 计算每组的加权平均概率
    high_res_mean, high_weights = weighted_mean(high_res_preds, high_res_scores)
    low_res_mean, low_weights = weighted_mean(low_res_preds, low_res_scores)
    
    # 保存组内权重信息
    group_weights = {
        "high_res": high_weights,
        "low_res": low_weights,
    }
    
    if high_res_mean is None:
        return 0.0, low_res_mean, -1.0, [], group_weights
    if low_res_mean is None:
        return 1.0, high_res_mean, -1.0, [], group_weights
    
    best_alpha = 0.5
    best_f1 = -1.0
    best_probs = None
    
    results = []
    
    for alpha in alpha_candidates:
        ens_prob = alpha * high_res_mean + (1 - alpha) * low_res_mean
        
        # 使用细化阈值搜索（两阶段：粗搜 + 局部细化）
        thresholds = search_best_thresholds_refined(y_true, ens_prob)
        _, f1_macro, _ = compute_metrics(y_true, ens_prob, thresholds)
        
        results.append((alpha, f1_macro))
        
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_alpha = alpha
            best_probs = ens_prob
    
    return best_alpha, best_probs, best_f1, results, group_weights


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


def nonneg_stacking(preds_list: list, y_true: np.ndarray, use_cv: bool = False, n_splits: int = 5) -> tuple:
    """
    非负最小二乘 Stacking：约束权重 >= 0，更接近加权平均的物理意义
    
    对每个类别独立求解: min ||Xw - y||^2  s.t. w >= 0
    然后归一化权重，使其和为 1。
    
    Args:
        preds_list: 各模型的预测概率列表
        y_true: 真实标签
        use_cv: 是否使用交叉验证（避免数据泄露）
        n_splits: K-Fold 的折数，默认 5（仅 use_cv=True 时有效）
    
    Returns:
        (stacked_probs, weights_per_class): 集成概率和每个类别的权重矩阵
    
    Note:
        - use_cv=False: 在全量数据上拟合权重，有轻微乐观偏差，但更稳定
        - use_cv=True: 使用 StratifiedKFold 生成 OOF 预测，严格无泄露
    """
    from scipy.optimize import nnls
    
    n, c = preds_list[0].shape
    m = len(preds_list)
    feats = np.stack(preds_list, axis=0).transpose(1, 2, 0)  # (N, C, M)
    stacked_probs = np.zeros((n, c), dtype=np.float32)
    weights_per_class = np.zeros((c, m), dtype=np.float32)  # (C, M)
    
    if use_cv:
        # 严格无泄露模式：使用 StratifiedKFold
        from sklearn.model_selection import StratifiedKFold
        
        for cls in range(c):
            X = feats[:, cls, :]  # (N, M)
            y = y_true[:, cls].astype(np.float64)
            
            pos_count = y.sum()
            neg_count = n - pos_count
            
            # 如果正/负样本太少，使用简单平均
            min_samples_per_fold = 10
            if pos_count < min_samples_per_fold * n_splits or neg_count < min_samples_per_fold * n_splits:
                weights = np.ones(m) / m
                weights_per_class[cls] = weights
                stacked_probs[:, cls] = X @ weights
                continue
            
            fold_weights = []
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            for train_idx, val_idx in skf.split(X, y_true[:, cls]):
                weights, _ = nnls(X[train_idx], y[train_idx])
                
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                else:
                    weights = np.ones(m) / m
                
                fold_weights.append(weights)
                stacked_probs[val_idx, cls] = X[val_idx] @ weights
            
            weights_per_class[cls] = np.mean(fold_weights, axis=0)
    else:
        # 稳定模式：在全量数据上拟合权重
        for cls in range(c):
            X = feats[:, cls, :]
            y = y_true[:, cls].astype(np.float64)
            
            weights, _ = nnls(X, y)
            
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


def per_class_f1_optimized_ensemble(preds_list: list, y_true: np.ndarray, model_names: list = None):
    """
    Per-class F1 直接优化集成：对每个类别独立搜索使 F1 最大的权重组合
    
    相比基于 AUC 的间接方法，此方法直接优化目标指标（F1），
    能找到真正使 F1 最大的权重组合。
    
    Args:
        preds_list: 各模型的预测概率列表，每个元素 shape=(N, C)
        y_true: 真实标签 shape=(N, C)
        model_names: 模型名称列表（用于日志输出）
    
    Returns:
        (集成概率, 权重矩阵, per_class_info)
    """
    from scipy.optimize import minimize
    from sklearn.metrics import f1_score
    
    m = len(preds_list)
    n, c = preds_list[0].shape
    
    stacked = np.stack(preds_list, axis=0)  # (M, N, C)
    ens_prob = np.zeros((n, c), dtype=np.float32)
    weights_matrix = np.zeros((c, m), dtype=np.float32)  # (C, M)
    per_class_info = []  # 存储每个类别的优化信息
    
    for cls in range(c):
        preds_cls = stacked[:, :, cls].T  # (N, M) - 每列是一个模型的预测
        y_cls = y_true[:, cls]
        
        # 检查是否有足够的正负样本
        if y_cls.sum() < 2 or (1 - y_cls).sum() < 2:
            # 样本太少，使用简单平均
            weights = np.ones(m) / m
            best_f1 = 0.0
        else:
            def objective(w):
                """目标函数：最小化负 F1"""
                # 确保权重非负并归一化
                w_abs = np.abs(w)
                if w_abs.sum() == 0:
                    w_abs = np.ones(m) / m
                else:
                    w_abs = w_abs / w_abs.sum()
                
                # 计算加权集成概率
                prob = preds_cls @ w_abs
                
                # 使用 0.5 作为固定阈值计算 F1（加速优化）
                y_pred = (prob > 0.5).astype(int)
                f1 = f1_score(y_cls, y_pred, zero_division=0)
                
                return -f1  # 最小化负 F1
            
            # 使用 Nelder-Mead 优化（无需梯度，适合小规模问题）
            # 初始值：均匀分布
            x0 = np.ones(m) / m
            
            result = minimize(
                objective, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4}
            )
            
            # 获取最优权重
            weights = np.abs(result.x)
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = np.ones(m) / m
            
            best_f1 = -result.fun
        
        weights_matrix[cls] = weights
        ens_prob[:, cls] = preds_cls @ weights
        
        per_class_info.append({
            'class_idx': cls,
            'weights': weights.copy(),
            'f1': best_f1,
        })
    
    return ens_prob, weights_matrix, per_class_info


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
    
    # ========== 2. 分层加权集成（组内按性能加权）==========
    print("\n" + "=" * 70)
    print("策略 2: 分层加权（高分辨率 vs 低分辨率，组内按AUC加权）")
    print("=" * 70)
    
    # 分离高分辨率和低分辨率模型的预测及其性能分数
    high_res_names = [n for n in HIGH_RES_MODELS if n in available_models]
    low_res_names = [n for n in LOW_RES_MODELS if n in available_models]
    
    high_res_preds = [model_results[name]["y_pred_prob"] for name in high_res_names]
    low_res_preds = [model_results[name]["y_pred_prob"] for name in low_res_names]
    
    # 获取每个模型的 AUC 分数作为组内权重依据
    high_res_scores = [model_results[name]["auc"] for name in high_res_names]
    low_res_scores = [model_results[name]["auc"] for name in low_res_names]
    
    if high_res_preds and low_res_preds:
        best_alpha, ens_hier, best_f1, search_results, group_weights = hierarchical_ensemble(
            high_res_preds, low_res_preds, y_true,
            high_res_scores=high_res_scores,
            low_res_scores=low_res_scores,
        )
        
        print(f"高分辨率组模型: {high_res_names}")
        print(f"  组内 AUC: {[f'{s:.4f}' for s in high_res_scores]}")
        print(f"  组内权重: {dict(zip(high_res_names, group_weights['high_res'].round(3)))}")
        
        print(f"\n低分辨率组模型: {low_res_names}")
        print(f"  组内 AUC: {[f'{s:.4f}' for s in low_res_scores]}")
        print(f"  组内权重: {dict(zip(low_res_names, group_weights['low_res'].round(3)))}")
        
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
    
    # ========== 3a. 概率校准 + Logistic Stacking ==========
    print("\n" + "=" * 70)
    print("策略 3a: 概率校准 + Logistic Stacking (Isotonic + C=0.1)")
    print("=" * 70)
    
    print("正在对各模型预测进行 Isotonic Regression 概率校准...")
    calibrated_preds = calibrate_predictions(preds_list, y_true, use_cv=True)
    
    ens_stack_calib = logistic_stacking(calibrated_preds, y_true, C=0.1)
    result_stack_calib = report_results("概率校准 + Logistic Stacking", y_true, ens_stack_calib)
    all_results.append(result_stack_calib)
    
    # ========== 3b. 非负约束 Stacking ==========
    print("\n" + "=" * 70)
    print("策略 3b: 非负约束 Stacking (NNLS)")
    print("=" * 70)
    
    ens_nnls, nnls_weights = nonneg_stacking(preds_list, y_true, use_cv=True)
    
    # 显示每个类别学到的平均权重
    mean_weights = nnls_weights.mean(axis=0)
    print(f"NNLS 平均权重: {dict(zip(available_models, mean_weights.round(3)))}")
    
    result_nnls = report_results("非负约束 Stacking (NNLS)", y_true, ens_nnls)
    all_results.append(result_nnls)
    
    # ========== 3c. 概率校准 + 非负约束 Stacking ==========
    print("\n" + "=" * 70)
    print("策略 3c: 概率校准 + 非负约束 Stacking (Isotonic + NNLS)")
    print("=" * 70)
    
    ens_nnls_calib, nnls_weights_calib = nonneg_stacking(calibrated_preds, y_true, use_cv=False)
    
    mean_weights_calib = nnls_weights_calib.mean(axis=0)
    print(f"校准后 NNLS 平均权重: {dict(zip(available_models, mean_weights_calib.round(3)))}")
    
    result_nnls_calib = report_results("概率校准 + NNLS", y_true, ens_nnls_calib)
    all_results.append(result_nnls_calib)
    
    # ========== 1b. 多样性加权集成 (F1) ==========
    print("\n" + "=" * 70)
    print("策略 1b: 多样性加权集成（性能 × 多样性）")
    print("=" * 70)
    
    model_aucs = [model_results[name]["auc"] for name in available_models]
    
    ens_diversity, div_weights, perf_weights, diversity_scores = diversity_weighted_ensemble(
        preds_list, model_aucs, diversity_weight=0.3
    )
    
    print(f"模型多样性分析:")
    print(f"  性能分数 (AUC):     {dict(zip(available_models, [f'{s:.4f}' for s in model_aucs]))}")
    print(f"  多样性分数:         {dict(zip(available_models, diversity_scores.round(4)))}")
    print(f"  综合权重 (0.7性能+0.3多样性): {dict(zip(available_models, div_weights.round(4)))}")
    
    result_diversity = report_results("多样性加权 (0.7P+0.3D)", y_true, ens_diversity)
    all_results.append(result_diversity)
    
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
    
    # ========== 4b. 多样性加权 + AUC 优化 ==========
    print("\n" + "=" * 70)
    print("策略 4b: 多样性加权集成（AUC 模式，性能 × 多样性）")
    print("=" * 70)
    
    # 使用更高的多样性权重
    ens_diversity_auc, div_weights_auc, _, _ = diversity_weighted_ensemble(
        preds_list, model_aucs, diversity_weight=0.4
    )
    
    print(f"综合权重 (0.6性能+0.4多样性): {dict(zip(available_models, div_weights_auc.round(4)))}")
    
    result_diversity_auc = report_results("多样性加权 (0.6P+0.4D)", y_true, ens_diversity_auc)
    all_results.append(result_diversity_auc)
    
    # ========== 5. Per-class F1 直接优化集成 ==========
    print("\n" + "=" * 70)
    print("策略 5: Per-class F1 直接优化（仅高分辨率模型: 3,4,5）")
    print("=" * 70)
    
    # 只使用高分辨率模型（剔除 AUC 较低的前两个模型）
    selected_models = [n for n in HIGH_RES_MODELS if n in available_models]
    
    if len(selected_models) >= 2:
        selected_preds = [model_results[name]["y_pred_prob"] for name in selected_models]
        
        print(f"\n选用模型（高分辨率组）:")
        for name in selected_models:
            auc = model_results[name]["auc"]
            f1 = model_results[name]["f1_macro"]
            print(f"  {name}: AUC={auc:.4f}, F1={f1:.4f}")
        
        print(f"\n正在对 {len(selected_models)} 个模型进行 Per-class F1 直接优化...")
        
        # 使用新的 F1 直接优化方法
        ens_perclass, weights_matrix, per_class_info = per_class_f1_optimized_ensemble(
            selected_preds, y_true, model_names=selected_models
        )
        
        # 显示每个类别学到的权重（汇总）
        print(f"\nPer-class 优化权重汇总:")
        print(f"  模型: {selected_models}")
        
        mean_weights = weights_matrix.mean(axis=0)
        std_weights = weights_matrix.std(axis=0)
        print(f"  平均权重: {dict(zip(selected_models, mean_weights.round(3)))}")
        print(f"  权重标准差: {dict(zip(selected_models, std_weights.round(3)))}")
        
        # 显示权重分布极端的类别
        print(f"\n权重分配特殊的类别（某模型权重 > 0.6）:")
        for info in per_class_info:
            max_w = info['weights'].max()
            if max_w > 0.6:
                max_idx = info['weights'].argmax()
                print(f"  类别 {info['class_idx']}: {selected_models[max_idx]} 权重={max_w:.3f}, F1={info['f1']:.4f}")
        
        result_perclass = report_results("Per-class F1优化 (高分辨率)", y_true, ens_perclass)
        all_results.append(result_perclass)
    else:
        print(f"[跳过] 高分辨率模型不足（需要至少 2 个，当前 {len(selected_models)} 个）")
    
    # ========== 5b. 权重与阈值联合优化 ==========
    print("\n" + "=" * 70)
    print("策略 5b: 权重与阈值联合优化（全模型）")
    print("=" * 70)
    
    print("正在对全部模型进行权重与阈值联合优化（Nelder-Mead）...")
    ens_joint, joint_weights, joint_thresholds, joint_f1 = joint_optimize_weights_thresholds(
        preds_list, y_true, model_names=available_models
    )
    
    print(f"\n联合优化结果:")
    print(f"  最佳权重: {dict(zip(available_models, joint_weights.round(4)))}")
    print(f"  优化后 Macro-F1: {joint_f1:.4f}")
    print(f"  各类阈值范围: [{joint_thresholds.min():.3f}, {joint_thresholds.max():.3f}]")
    print(f"  阈值均值: {joint_thresholds.mean():.3f}, 标准差: {joint_thresholds.std():.3f}")
    
    # 使用联合优化的阈值计算指标
    y_pred_bin_joint = (ens_joint >= joint_thresholds[None, :]).astype(int)
    f1_joint_final = f1_score(y_true, y_pred_bin_joint, average='macro', zero_division=0)
    auc_joint, _, _ = compute_metrics(y_true, ens_joint, thresholds=None)
    
    print(f"\n===== 集成结果 [权重与阈值联合优化] =====")
    print(f"  AUC:                      {auc_joint:.4f}")
    print(f"  Macro-F1 (联合优化阈值):  {f1_joint_final:.4f}")
    
    # 为了公平比较，也用 search_best_thresholds 评估
    thresholds_search = search_best_thresholds(y_true, ens_joint)
    _, f1_search, _ = compute_metrics(y_true, ens_joint, thresholds_search)
    print(f"  Macro-F1 (独立搜索阈值):  {f1_search:.4f}")
    
    result_joint = {
        "tag": "权重与阈值联合优化",
        "auc": auc_joint,
        "f1_macro": max(f1_joint_final, f1_search),  # 取两种阈值方式的更优值
        "f1_weighted": 0,
        "thresholds": joint_thresholds if f1_joint_final >= f1_search else thresholds_search,
        "y_pred_prob": ens_joint,
    }
    all_results.append(result_joint)
    
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
