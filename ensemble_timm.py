# ensemble_timm.py
"""
Timm 模型集成评估脚本（简化版）：
- F1 优化：使用 2 个高 F1 模型，只输出 F1
- AUC 优化：使用 3 个高 AUC 模型，只输出 AUC
- 多架构 AUC 优化：使用 4 种不同架构模型，只输出 AUC
"""
import os
import itertools

import numpy as np
from sklearn.metrics import f1_score
from scipy.optimize import minimize
from tqdm import tqdm

from src.cxr_config import SAVE_DIR
from src.metrics_utils import compute_metrics, search_best_thresholds
from src.log_utils import setup_logger, close_logger


# ========== 模型配置 ==========

# F1 优化集成（2个高F1模型）
F1_MODELS = [
    "convnext_base_in22k_512",      # F1=0.3454 ★
    "convnext_small_in22k_384",     # F1=0.3420
]

# AUC 优化集成（3个高AUC模型）
AUC_MODELS = [
    "convnext_small_in22k_512",     # AUC=0.8186
    "convnext_base_in22k_512",      # AUC=0.8180
    "convnextv2_tiny_in22k_512",    # AUC=0.8176
]

# 多架构 AUC 优化集成（4种不同架构）
DIVERSE_AUC_MODELS = [
    "convnext_small_in22k_512",     # ConvNeXt, AUC=0.8186
    "convnextv2_tiny_in22k_512",    # ConvNeXtV2, AUC=0.8176
    "swin_base_in22k_384",          # Swin Transformer, AUC≈0.81
    "coat_lite_medium_384",         # CoaT, AUC=0.8199 ★
]

# 所有需要加载的模型（去重）
ALL_MODELS = list(set(F1_MODELS + AUC_MODELS + DIVERSE_AUC_MODELS))


def load_single_model_preds(name: str):
    """加载单个模型的验证集预测"""
    npz_path = os.path.join(SAVE_DIR, f"val_preds_{name}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"找不到验证集预测文件: {npz_path}\n请确认该模型已经训练并保存了 .npz")

    data = np.load(npz_path)
    y_true = data["y_true"]
    y_pred_prob = data["y_pred_prob"]
    
    # 尝试加载训练过程中的最佳指标
    best_metrics = {}
    if "best_auc" in data:
        best_metrics["best_auc"] = float(data["best_auc"])
    if "best_f1_macro" in data:
        best_metrics["best_f1_macro"] = float(data["best_f1_macro"])
    
    return y_true, y_pred_prob, best_metrics


def evaluate_single_model(name: str, model_results: dict):
    """评估单个模型的性能并存入 model_results"""
    y_true, y_pred_prob, best_metrics = load_single_model_preds(name)

    # 搜索每类最佳阈值
    thresholds = search_best_thresholds(y_true, y_pred_prob)
    auc_t, f1_t, f1_weighted_t = compute_metrics(y_true, y_pred_prob, thresholds)

    print(f"\n===== 单模型 [{name}] =====")
    if best_metrics:
        print(f"  [训练过程最佳]")
        if 'best_auc' in best_metrics:
            print(f"    Best AUC:      {best_metrics['best_auc']:.4f}")
        if 'best_f1_macro' in best_metrics:
            print(f"    Best Macro-F1: {best_metrics['best_f1_macro']:.4f}")

    model_results[name] = {
        "name": name,
        "y_true": y_true,
        "y_pred_prob": y_pred_prob,
        "auc": auc_t,
        "f1_macro": f1_t,
        "thresholds": thresholds,
        "best_metrics": best_metrics,
    }


# ========== 阈值搜索工具 ==========

def search_best_thresholds_refined(y_true: np.ndarray, y_pred_prob: np.ndarray) -> np.ndarray:
    """两阶段阈值搜索：先粗搜，再局部细化"""
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


# ========== F1 优化集成策略 ==========

def weighted_average_f1(preds_list: list, y_true: np.ndarray, model_names: list, top_k=100):
    """
    F1 优化的加权平均集成
    - 只输出 F1，不计算 AUC
    """
    m = len(preds_list)
    grid = np.linspace(0.0, 1.0, 21)  # 21点网格
    total_combinations = len(grid) ** m
    stacked_preds = np.stack(preds_list, axis=0)
    
    # ========== 阶段 1: 粗筛（固定阈值 0.5）==========
    print(f"\n[阶段 1] 粗筛: {total_combinations} 个权重组合...")
    
    candidates = []
    
    pbar = tqdm(
        itertools.product(grid, repeat=m),
        total=total_combinations,
        desc=f"F1粗筛 ({m}模型)",
    )
    
    for weights in pbar:
        if sum(weights) == 0:
            continue
        w = np.array(weights)
        w = w / w.sum()
        
        ens_prob = np.tensordot(w, stacked_preds, axes=1)
        _, score, _ = compute_metrics(y_true, ens_prob, thresholds=None)
        
        candidates.append((score, w.copy()))
    
    candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = candidates[:top_k]
    
    print(f"[阶段 1 完成] Top-{top_k} 粗筛 F1: {top_candidates[-1][0]:.4f} ~ {top_candidates[0][0]:.4f}")
    
    # ========== 阶段 2: 精选（精细阈值搜索）==========
    print(f"\n[阶段 2] 精选: 对 Top-{top_k} 进行精细阈值搜索...")
    
    best_f1 = -1.0
    best_weights = None
    best_thresholds = None
    best_prob = None
    
    for _, w in tqdm(top_candidates, desc="F1精选"):
        ens_prob = np.tensordot(w, stacked_preds, axes=1)
        thresholds = search_best_thresholds_refined(y_true, ens_prob)
        _, f1_macro, _ = compute_metrics(y_true, ens_prob, thresholds)
        
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_weights = w
            best_thresholds = thresholds
            best_prob = ens_prob
    
    print(f"\n[F1 集成完成]")
    print(f"  最佳 Macro-F1: {best_f1:.4f}")
    print(f"  权重: {dict(zip(model_names, best_weights.round(4)))}")
    
    return best_prob, best_weights, best_thresholds, best_f1


def per_class_f1_ensemble(preds_list: list, y_true: np.ndarray, model_names: list):
    """
    Per-class F1 直接优化集成
    - 对每个类别独立搜索最优权重
    - 只输出 F1，不计算 AUC
    """
    m = len(preds_list)
    n, c = preds_list[0].shape
    
    stacked = np.stack(preds_list, axis=0)  # (M, N, C)
    ens_prob = np.zeros((n, c), dtype=np.float32)
    weights_matrix = np.zeros((c, m), dtype=np.float32)
    
    for cls in range(c):
        preds_cls = stacked[:, :, cls].T  # (N, M)
        y_cls = y_true[:, cls]
        
        if y_cls.sum() < 2 or (1 - y_cls).sum() < 2:
            weights = np.ones(m) / m
        else:
            def objective(w):
                w_abs = np.abs(w)
                if w_abs.sum() == 0:
                    w_abs = np.ones(m) / m
                else:
                    w_abs = w_abs / w_abs.sum()
                prob = preds_cls @ w_abs
                y_pred = (prob > 0.5).astype(int)
                f1 = f1_score(y_cls, y_pred, zero_division=0)
                return -f1
            
            result = minimize(objective, np.ones(m) / m, method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            
            weights = np.abs(result.x)
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones(m) / m
        
        weights_matrix[cls] = weights
        ens_prob[:, cls] = preds_cls @ weights
    
    # 计算最终 F1
    thresholds = search_best_thresholds_refined(y_true, ens_prob)
    _, f1_macro, _ = compute_metrics(y_true, ens_prob, thresholds)
    
    mean_weights = weights_matrix.mean(axis=0)
    print(f"\n[Per-class F1 集成完成]")
    print(f"  Macro-F1: {f1_macro:.4f}")
    print(f"  平均权重: {dict(zip(model_names, mean_weights.round(3)))}")
    
    return ens_prob, weights_matrix, thresholds, f1_macro


# ========== AUC 优化集成策略 ==========

def weighted_average_auc(preds_list: list, y_true: np.ndarray, model_names: list):
    """
    AUC 优化的加权平均集成
    - 只输出 AUC，不计算 F1
    """
    m = len(preds_list)
    grid = np.linspace(0.0, 1.0, 21)  # 21点网格
    total_combinations = len(grid) ** m
    stacked_preds = np.stack(preds_list, axis=0)
    
    print(f"\n[AUC 网格搜索] {total_combinations} 个权重组合...")
    
    best_auc = -1.0
    best_weights = None
    best_prob = None
    
    pbar = tqdm(
        itertools.product(grid, repeat=m),
        total=total_combinations,
        desc=f"AUC搜索 ({m}模型)",
    )
    
    for weights in pbar:
        if sum(weights) == 0:
            continue
        w = np.array(weights)
        w = w / w.sum()
        
        ens_prob = np.tensordot(w, stacked_preds, axes=1)
        auc, _, _ = compute_metrics(y_true, ens_prob, thresholds=None)
        
        if auc > best_auc:
            best_auc = auc
            best_weights = w.copy()
            best_prob = ens_prob
            pbar.set_postfix({"best_auc": f"{best_auc:.4f}"})
    
    print(f"\n[AUC 集成完成]")
    print(f"  最佳 AUC: {best_auc:.4f}")
    print(f"  权重: {dict(zip(model_names, best_weights.round(4)))}")
    
    return best_prob, best_weights, best_auc


def compute_model_diversity(preds_list: list) -> np.ndarray:
    """计算模型间的多样性分数（1 - 平均相关性）"""
    m = len(preds_list)
    flat_preds = [p.flatten() for p in preds_list]
    corr_matrix = np.corrcoef(flat_preds)
    
    diversity_scores = np.zeros(m, dtype=np.float32)
    for i in range(m):
        other_corrs = [corr_matrix[i, j] for j in range(m) if j != i]
        diversity_scores[i] = 1 - np.mean(other_corrs)
    
    return diversity_scores


def diversity_weighted_auc(preds_list: list, y_true: np.ndarray, model_aucs: list, model_names: list, diversity_weight: float = 0.3):
    """
    多样性感知的 AUC 集成
    - 综合考虑性能和多样性
    - 只输出 AUC，不计算 F1
    """
    m = len(preds_list)
    stacked = np.stack(preds_list, axis=0)
    
    # 性能权重
    performance_scores = np.array(model_aucs, dtype=np.float32)
    perf_weights = performance_scores / performance_scores.sum()
    
    # 多样性权重
    diversity_scores = compute_model_diversity(preds_list)
    div_weights = diversity_scores / diversity_scores.sum()
    
    # 综合权重
    combined_weights = (1 - diversity_weight) * perf_weights + diversity_weight * div_weights
    combined_weights = combined_weights / combined_weights.sum()
    
    # 集成
    ens_prob = np.tensordot(combined_weights, stacked, axes=1)
    auc, _, _ = compute_metrics(y_true, ens_prob, thresholds=None)
    
    print(f"\n[多样性加权 AUC 集成]")
    print(f"  多样性分数: {dict(zip(model_names, diversity_scores.round(4)))}")
    print(f"  综合权重: {dict(zip(model_names, combined_weights.round(4)))}")
    print(f"  AUC: {auc:.4f}")
    
    return ens_prob, combined_weights, auc


# ========== 主函数 ==========

def main():
    setup_logger("ensemble_timm")
    
    print("=" * 70)
    print("Timm 模型集成评估（简化版）")
    print("=" * 70)
    print(f"SAVE_DIR = {SAVE_DIR}")
    print(f"\nF1 优化模型组（2模型）: {F1_MODELS}")
    print(f"AUC 优化模型组（3模型）: {AUC_MODELS}")
    print(f"多架构 AUC 模型组（4模型）: {DIVERSE_AUC_MODELS}")
    print("=" * 70)
    
    # ========== 1. 加载所有模型预测 ==========
    model_results = {}
    
    for name in ALL_MODELS:
        try:
            evaluate_single_model(name, model_results)
        except FileNotFoundError as e:
            print(f"\n[跳过] {name}: {e}")
    
    print(f"\n已加载模型: {list(model_results.keys())}")
    
    # 获取 y_true（从任一模型）
    if not model_results:
        print("[错误] 没有可用的模型")
        return
    
    y_true = list(model_results.values())[0]["y_true"]
    
    # 验证 y_true 一致性
    for name, result in model_results.items():
        if not np.array_equal(y_true, result["y_true"]):
            raise ValueError(f"模型 {name} 的 y_true 不一致")
    
    f1_results = []
    auc_results = []
    
    # ========== 2. F1 优化集成（2模型）==========
    print("\n" + "=" * 70)
    print("【F1 优化集成】使用 2 个高 F1 模型")
    print("=" * 70)
    
    f1_model_names = [n for n in F1_MODELS if n in model_results]
    
    if len(f1_model_names) >= 2:
        f1_preds = [model_results[name]["y_pred_prob"] for name in f1_model_names]
        
        print(f"\n使用模型:")
        for name in f1_model_names:
            f1 = model_results[name]["f1_macro"]
            print(f"  {name}: F1={f1:.4f}")
        
        # 策略 F1-1: 加权平均
        print("\n--- 策略 F1-1: 加权平均 ---")
        prob_f1_1, weights_f1_1, thr_f1_1, score_f1_1 = weighted_average_f1(
            f1_preds, y_true, f1_model_names
        )
        f1_results.append(("加权平均", score_f1_1, prob_f1_1, thr_f1_1))
        
        # 策略 F1-2: Per-class 优化
        print("\n--- 策略 F1-2: Per-class 优化 ---")
        prob_f1_2, _, thr_f1_2, score_f1_2 = per_class_f1_ensemble(
            f1_preds, y_true, f1_model_names
        )
        f1_results.append(("Per-class优化", score_f1_2, prob_f1_2, thr_f1_2))
        
        # 策略 F1-3: 简单平均
        print("\n--- 策略 F1-3: 简单平均 ---")
        prob_f1_3 = np.stack(f1_preds, axis=0).mean(axis=0)
        thr_f1_3 = search_best_thresholds_refined(y_true, prob_f1_3)
        _, score_f1_3, _ = compute_metrics(y_true, prob_f1_3, thr_f1_3)
        print(f"  Macro-F1: {score_f1_3:.4f}")
        f1_results.append(("简单平均", score_f1_3, prob_f1_3, thr_f1_3))
        
    else:
        print(f"[跳过] F1 模型不足（需要 2 个，当前 {len(f1_model_names)} 个）")
    
    # ========== 3. AUC 优化集成（3模型）==========
    print("\n" + "=" * 70)
    print("【AUC 优化集成】使用 3 个高 AUC 模型")
    print("=" * 70)
    
    auc_model_names = [n for n in AUC_MODELS if n in model_results]
    
    if len(auc_model_names) >= 2:
        auc_preds = [model_results[name]["y_pred_prob"] for name in auc_model_names]
        
        print(f"\n使用模型:")
        for name in auc_model_names:
            auc = model_results[name]["auc"]
            print(f"  {name}: AUC={auc:.4f}")
        
        # 策略 AUC-1: 加权平均
        print("\n--- 策略 AUC-1: 加权平均 ---")
        prob_auc_1, weights_auc_1, score_auc_1 = weighted_average_auc(
            auc_preds, y_true, auc_model_names
        )
        auc_results.append(("3模型加权平均", score_auc_1, prob_auc_1))
        
        # 策略 AUC-2: 简单平均
        print("\n--- 策略 AUC-2: 简单平均 ---")
        prob_auc_2 = np.stack(auc_preds, axis=0).mean(axis=0)
        score_auc_2, _, _ = compute_metrics(y_true, prob_auc_2, thresholds=None)
        print(f"  AUC: {score_auc_2:.4f}")
        auc_results.append(("3模型简单平均", score_auc_2, prob_auc_2))
        
    else:
        print(f"[跳过] AUC 模型不足（需要 2 个，当前 {len(auc_model_names)} 个）")
    
    # ========== 4. 多架构 AUC 优化集成（4模型）==========
    print("\n" + "=" * 70)
    print("【多架构 AUC 集成】使用 4 种不同架构模型")
    print("=" * 70)
    
    diverse_model_names = [n for n in DIVERSE_AUC_MODELS if n in model_results]
    
    if len(diverse_model_names) >= 2:
        diverse_preds = [model_results[name]["y_pred_prob"] for name in diverse_model_names]
        diverse_aucs = [model_results[name]["auc"] for name in diverse_model_names]
        
        print(f"\n使用模型:")
        for name in diverse_model_names:
            auc = model_results[name]["auc"]
            # 架构类型
            if "convnextv2" in name:
                arch = "ConvNeXtV2"
            elif "convnext" in name:
                arch = "ConvNeXt"
            elif "swin" in name:
                arch = "Swin"
            elif "coat" in name:
                arch = "CoaT"
            else:
                arch = "Unknown"
            print(f"  {name} [{arch}]: AUC={auc:.4f}")
        
        # 策略 Div-1: 加权平均
        print("\n--- 策略 Div-1: 加权平均 ---")
        prob_div_1, weights_div_1, score_div_1 = weighted_average_auc(
            diverse_preds, y_true, diverse_model_names
        )
        auc_results.append(("4架构加权平均", score_div_1, prob_div_1))
        
        # 策略 Div-2: 多样性加权
        print("\n--- 策略 Div-2: 多样性加权 ---")
        prob_div_2, weights_div_2, score_div_2 = diversity_weighted_auc(
            diverse_preds, y_true, diverse_aucs, diverse_model_names, diversity_weight=0.3
        )
        auc_results.append(("4架构多样性加权", score_div_2, prob_div_2))
        
        # 策略 Div-3: 简单平均
        print("\n--- 策略 Div-3: 简单平均 ---")
        prob_div_3 = np.stack(diverse_preds, axis=0).mean(axis=0)
        score_div_3, _, _ = compute_metrics(y_true, prob_div_3, thresholds=None)
        print(f"  AUC: {score_div_3:.4f}")
        auc_results.append(("4架构简单平均", score_div_3, prob_div_3))
        
    else:
        print(f"[跳过] 多架构模型不足（需要 2 个，当前 {len(diverse_model_names)} 个）")
    
    # ========== 5. 汇总结果 ==========
    print("\n" + "=" * 70)
    print("集成结果汇总")
    print("=" * 70)
    
    # F1 结果
    if f1_results:
        print(f"\n【F1 优化结果】（2模型: {', '.join(f1_model_names)}）")
        print("-" * 40)
        print(f"{'策略':<20} | {'Macro-F1':>10}")
        print("-" * 40)
        for name, score, _, _ in f1_results:
            print(f"{name:<20} | {score:>10.4f}")
        
        # 找到最佳 F1
        best_f1 = max(f1_results, key=lambda x: x[1])
        print("-" * 40)
        print(f">>> 最佳 F1 策略: {best_f1[0]}, Macro-F1 = {best_f1[1]:.4f}")
    
    # AUC 结果
    if auc_results:
        print(f"\n【AUC 优化结果】")
        print("-" * 40)
        print(f"{'策略':<20} | {'AUC':>10}")
        print("-" * 40)
        for name, score, _ in auc_results:
            print(f"{name:<20} | {score:>10.4f}")
        
        # 找到最佳 AUC
        best_auc = max(auc_results, key=lambda x: x[1])
        print("-" * 40)
        print(f">>> 最佳 AUC 策略: {best_auc[0]}, AUC = {best_auc[1]:.4f}")
    
    # ========== 6. 保存结果 ==========
    print("\n" + "=" * 70)
    print("保存结果")
    print("=" * 70)
    
    # 保存 F1 最佳结果
    if f1_results:
        best_f1_name, best_f1_score, best_f1_prob, best_f1_thr = max(f1_results, key=lambda x: x[1])
        save_path_f1 = os.path.join(SAVE_DIR, "val_preds_timm_ensemble_f1.npz")
        np.savez(
            save_path_f1,
            y_true=y_true,
            y_pred_prob=best_f1_prob,
            thresholds=best_f1_thr,
            model_names=np.array(f1_model_names),
            ensemble_strategy=best_f1_name,
            best_f1=best_f1_score,
        )
        print(f">>> F1 最佳集成已保存至: {save_path_f1}")
        
        # 单独保存阈值
        np.save(os.path.join(SAVE_DIR, "timm_ensemble_thresholds_f1.npy"), best_f1_thr)
    
    # 保存 AUC 最佳结果
    if auc_results:
        best_auc_name, best_auc_score, best_auc_prob = max(auc_results, key=lambda x: x[1])
        save_path_auc = os.path.join(SAVE_DIR, "val_preds_timm_ensemble_auc.npz")
        np.savez(
            save_path_auc,
            y_true=y_true,
            y_pred_prob=best_auc_prob,
            model_names=np.array(diverse_model_names if "4架构" in best_auc_name else auc_model_names),
            ensemble_strategy=best_auc_name,
            best_auc=best_auc_score,
        )
        print(f">>> AUC 最佳集成已保存至: {save_path_auc}")
    
    print("\n>>> 集成评估完成！")
    
    close_logger()


if __name__ == "__main__":
    main()
