# ensemble_val.py
"""
集成评估脚本：
- F1 计算：使用所有模型的集成策略（平均/加权/stacking）
- AUC 计算：使用 pc + chex 的贪婪集成策略，搜索最佳比例
"""
import os
import itertools

import numpy as np
from sklearn.linear_model import LogisticRegression

from cxr_config import SAVE_DIR, CLASS_NAMES
from metrics_utils import compute_metrics, search_best_thresholds


# ========== 模型配置 ==========
# F1 集成：参与所有集成策略的模型（对应 train.py 中的 ENSEMBLE_MODELS）
MODEL_NAMES_FOR_F1 = [
    "densenet121_xrv_chex",
    "densenet121_xrv_pc",
    "densenet121_xrv_mimic_nb",
    "densenet121_xrv_mimic_ch",
]

# AUC 贪婪集成：只用 pc 和 chex
MODEL_NAMES_FOR_AUC = [
    "densenet121_xrv_pc",
    "densenet121_xrv_chex",
]

# 保持向后兼容
MODEL_NAMES = MODEL_NAMES_FOR_F1


def load_single_model_preds(name: str):
    npz_path = os.path.join(SAVE_DIR, f"val_preds_{name}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"找不到验证集预测文件: {npz_path}\n请确认该模型已经训练并保存了 .npz")

    data = np.load(npz_path)
    y_true = data["y_true"]
    y_pred_prob = data["y_pred_prob"]
    return y_true, y_pred_prob


def evaluate_single_model(name: str):
    y_true, y_pred_prob = load_single_model_preds(name)

    # 阈值=0.5
    auc_macro, f1_macro, f1_weighted = compute_metrics(y_true, y_pred_prob, thresholds=None)

    # 搜索每类最佳阈值
    thresholds = search_best_thresholds(y_true, y_pred_prob)
    auc_t, f1_t, f1_weighted_t = compute_metrics(y_true, y_pred_prob, thresholds)

    print(f"\n===== 单模型 [{name}] 评估结果 =====")
    print(f"Val AUC:                 {auc_t:.4f}")
    print(f"Val Macro-F1 (best thr):  {f1_t:.4f}")
    # print(f"Val Weighted-F1 (best thr):{f1_weighted_t:.4f}")

    return y_true, y_pred_prob


def evaluate_ensemble(model_names):
    assert len(model_names) >= 2, "集成至少需要两个模型"

    print("\n======================================")
    print("集成模型名单:", model_names)
    print("======================================")

    ys = []
    preds = []

    for name in model_names:
        y_true, y_pred_prob = load_single_model_preds(name)
        ys.append(y_true)
        preds.append(y_pred_prob)

    # 检查所有模型的 y_true 是否一致
    for i in range(1, len(ys)):
        if not np.array_equal(ys[0], ys[i]):
            raise ValueError(f"模型 {model_names[0]} 与 {model_names[i]} 的 y_true 不一致，请检查 .npz")

    y_true = ys[0]

    # 概率平均（baseline）
    stacked = np.stack(preds, axis=0)     # (M, N, C)
    y_pred_mean = stacked.mean(axis=0)     # (N, C)

    # 加权线性融合：简单网格搜索（步长0.1）
    def search_weighted_average(pred_list, y_true):
        m = len(pred_list)
        grid = np.linspace(0.0, 1.0, 11)
        best_f1 = -1.0
        best_w = None
        for weights in itertools.product(grid, repeat=m):
            if sum(weights) == 0:
                continue
            w = np.array(weights)
            w = w / w.sum()
            ens = np.tensordot(w, np.stack(pred_list, axis=0), axes=1)
            f1 = compute_metrics(y_true, ens, thresholds=None)[1]
            if f1 > best_f1:
                best_f1 = f1
                best_w = w
        return best_w, best_f1

    best_w, best_f1 = search_weighted_average(preds, y_true)
    if best_w is not None:
        print(f"\n>>> 加权平均最佳权重: {best_w}, F1={best_f1:.4f}")
        y_pred_weighted = np.tensordot(best_w, stacked, axes=1)
    else:
        y_pred_weighted = y_pred_mean

    # Logistic stacking：对每个类别训练一对多逻辑回归
    def logistic_stack(pred_list, y_true):
        m, n, c = np.stack(pred_list, axis=0).shape
        feats = np.stack(pred_list, axis=0).transpose(1, 2, 0)  # (N, C, M)
        stacked_probs = np.zeros((n, c), dtype=np.float32)
        for cls in range(c):
            X = feats[:, cls, :]  # (N, M)
            y = y_true[:, cls]
            clf = LogisticRegression(max_iter=200, n_jobs=1, class_weight="balanced")
            clf.fit(X, y)
            stacked_probs[:, cls] = clf.predict_proba(X)[:, 1]
        return stacked_probs

    y_pred_stack = logistic_stack(preds, y_true)

    def report(tag, probs):
        auc_macro, f1_macro, f1_weighted = compute_metrics(y_true, probs, thresholds=None)
        thresholds = search_best_thresholds(y_true, probs)
        auc_t, f1_t, f1_weighted_t = compute_metrics(y_true, probs, thresholds)
        print(f"\n===== 集成结果（{tag}） =====")
        print(f"Val AUC:                 {auc_t:.4f}")
        print(f"Val Macro-F1 (best thr):  {f1_t:.4f}")
        # print(f"Val Weighted-F1 (best thr):{f1_weighted_t:.4f}")
        return auc_t, f1_t, f1_weighted_t, thresholds, probs

    print("\n===== 基线：简单平均 =====")
    report("simple mean", y_pred_mean)
    print("\n===== 加权平均（搜索得到的权重） =====")
    weighted_results = report("weighted mean", y_pred_weighted)
    print("\n===== Logistic Stacking =====")
    stack_results = report("logistic stacking", y_pred_stack)

    # 返回表现最好的方案（按 Macro-F1 排序）
    candidates = [weighted_results, stack_results]
    best = max(candidates, key=lambda x: x[1])  # x[1] 是 macro-f1
    auc_best, f1_best, f1_weighted_best, thr_best, prob_best = best
    # 保存最佳集成的预测与阈值，方便推理/提交
    np.savez(
        os.path.join(SAVE_DIR, "val_preds_best_ensemble.npz"),
        y_true=y_true,
        y_pred_prob=prob_best,
        thresholds=thr_best,
        model_names=np.array(model_names),
    )
    print(f"\n>>> 最佳集成已保存至 saved_models/val_preds_best_ensemble.npz (Macro-F1={f1_best:.4f})")
    return best


def search_auc_greedy_ensemble(pc_name="densenet121_xrv_pc", chex_name="densenet121_xrv_chex"):
    """
    AUC 贪婪集成：只用 PC 和 CheXpert 两个模型
    搜索 PC 的占比从 0.5 到 1.0，找出最佳 AUC 的比例
    
    返回: (best_ratio, best_auc, best_probs, y_true)
    """
    print("\n" + "=" * 60)
    print("AUC 贪婪集成：PC + CheXpert 最佳比例搜索")
    print("=" * 60)
    
    # 加载两个模型的预测
    y_true_pc, y_pred_pc = load_single_model_preds(pc_name)
    y_true_chex, y_pred_chex = load_single_model_preds(chex_name)
    
    # 验证 y_true 一致
    if not np.array_equal(y_true_pc, y_true_chex):
        raise ValueError("PC 和 CheXpert 模型的 y_true 不一致，请检查数据划分")
    
    y_true = y_true_pc
    
    # 先打印单模型 AUC
    auc_pc, _, _ = compute_metrics(y_true, y_pred_pc, thresholds=None)
    auc_chex, _, _ = compute_metrics(y_true, y_pred_chex, thresholds=None)
    print(f"\n单模型 AUC:")
    print(f"  {pc_name}: AUC = {auc_pc:.4f}")
    print(f"  {chex_name}: AUC = {auc_chex:.4f}")
    
    # 搜索 PC 占比从 0.5 到 1.0（步长 0.01，更精细）
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
    
    # 打印搜索结果（每 0.1 打印一次 + 最优点）
    print(f"\n比例搜索结果 (PC : CheXpert):")
    print("-" * 45)
    for pc_ratio, auc_val in results:
        # 每 0.1 或最优点打印
        if abs(pc_ratio * 100 % 10) < 0.5 or abs(pc_ratio - best_ratio) < 0.005:
            marker = " <-- BEST" if abs(pc_ratio - best_ratio) < 0.005 else ""
            print(f"  PC={pc_ratio:.2f}, CheX={1-pc_ratio:.2f}  =>  AUC = {auc_val:.4f}{marker}")
    
    print("-" * 45)
    print(f"\n>>> AUC 最佳配比: PC = {best_ratio:.2f}, CheXpert = {1-best_ratio:.2f}")
    print(f">>> 最佳 AUC = {best_auc:.4f}")
    print(f">>> 相比单独 PC 提升: {(best_auc - auc_pc) * 100:.2f}%")
    print(f">>> 相比单独 CheXpert 提升: {(best_auc - auc_chex) * 100:.2f}%")
    
    # 保存 AUC 最佳集成结果
    np.savez(
        os.path.join(SAVE_DIR, "val_preds_auc_best_ensemble.npz"),
        y_true=y_true,
        y_pred_prob=best_probs,
        pc_ratio=best_ratio,
        chex_ratio=1-best_ratio,
        best_auc=best_auc,
        models=np.array([pc_name, chex_name]),
    )
    print(f"\n>>> AUC 最佳集成已保存至 saved_models/val_preds_auc_best_ensemble.npz")
    
    return best_ratio, best_auc, best_probs, y_true


def main():
    print("SAVE_DIR =", SAVE_DIR)
    
    # ========== Part 1: F1 集成（使用所有模型） ==========
    print("\n" + "=" * 60)
    print("Part 1: F1 集成评估（使用所有模型）")
    print("=" * 60)
    print("参与 F1 集成的模型:", MODEL_NAMES_FOR_F1)
    
    # 先打印每个单模型的指标
    for name in MODEL_NAMES_FOR_F1:
        try:
            evaluate_single_model(name)
        except FileNotFoundError as e:
            print(f"[跳过] {name}: {e}")
    
    # 评估所有模型一起的集成（用于 F1）
    available_models = []
    for name in MODEL_NAMES_FOR_F1:
        npz_path = os.path.join(SAVE_DIR, f"val_preds_{name}.npz")
        if os.path.exists(npz_path):
            available_models.append(name)
    
    if len(available_models) >= 2:
        print(f"\n可用于 F1 集成的模型: {available_models}")
        evaluate_ensemble(available_models)
    else:
        print(f"\n[警告] 只有 {len(available_models)} 个模型可用，无法进行集成")
    
    # ========== Part 2: AUC 贪婪集成（只用 PC + CheXpert） ==========
    print("\n" + "=" * 60)
    print("Part 2: AUC 贪婪集成（PC + CheXpert）")
    print("=" * 60)
    print("参与 AUC 集成的模型:", MODEL_NAMES_FOR_AUC)
    
    try:
        best_ratio, best_auc, _, _ = search_auc_greedy_ensemble(
            pc_name="densenet121_xrv_pc",
            chex_name="densenet121_xrv_chex"
        )
    except FileNotFoundError as e:
        print(f"[跳过 AUC 集成] {e}")
    
    # ========== 汇总 ==========
    print("\n" + "=" * 60)
    print("集成策略汇总")
    print("=" * 60)
    print("F1 计算: 使用所有模型的加权平均/Logistic Stacking")
    print("AUC 计算: 使用 PC + CheXpert 的最佳比例融合")
    print("=" * 60)


if __name__ == "__main__":
    main()
