# ensemble_search.py
"""
在验证集上对多模型进行集成搜索：
- 比较单模型 performance
- 所有两两组合的集成
- 所有三模型组合的集成
"""

import itertools
import os
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.cxr_config import SAVE_DIR
from src.metrics_utils import compute_metrics
from src.log_utils import setup_logger, close_logger


# 这里写上你已经训练过、并且在 saved_models 下能找到 val_preds_*.npz 的模型名字
MODEL_NAMES = [
    #"convnext_base_in22k",
    #"vit_base_in21k",
    #"efficientnet_b0_in1k",
    # 根据实际训练情况加减
]


def load_val_preds(name):
    path = os.path.join(SAVE_DIR, f"val_preds_{name}.npz")
    data = np.load(path, allow_pickle=True)
    return data["y_true"], data["y_pred_prob"]


def main():
    # 设置日志记录，所有 print 输出同时保存到 logs/ 目录
    setup_logger("ensemble_search")
    
    preds = {}
    y_true_ref = None

    for name in MODEL_NAMES:
        y_true, y_pred = load_val_preds(name)
        preds[name] = y_pred
        if y_true_ref is None:
            y_true_ref = y_true
        else:
            assert np.allclose(y_true_ref, y_true), "不同模型的 y_true 不一致，请检查划分是否相同"

    print("===== 单模型表现 =====")
    for name in MODEL_NAMES:
        auc, f1_macro, f1_weighted = compute_metrics(y_true_ref, preds[name], thresholds=None)
        print(f"{name:25s}  AUC={auc:.4f}  Macro-F1={f1_macro:.4f}")

    print("\n===== 两模型平均集成 =====")
    best_score = -1.0
    best_combo = None
    best_auc = None
    best_f1_weighted = None
    best_probs = None

    for combo in itertools.combinations(MODEL_NAMES, 2):
        prob_ens = (preds[combo[0]] + preds[combo[1]]) / 2.0
        auc, f1_macro, f1_weighted = compute_metrics(y_true_ref, prob_ens, thresholds=None)
        print(f"{'+'.join(combo):50s}  AUC={auc:.4f}  Macro-F1={f1_macro:.4f}")
        if f1_macro > best_score:
            best_score = f1_macro
            best_f1_weighted = f1_weighted
            best_auc = auc
            best_combo = combo
            best_probs = prob_ens

    print("\n===== 三模型平均集成 =====")
    for combo in itertools.combinations(MODEL_NAMES, 3):
        prob_ens = sum(preds[n] for n in combo) / len(combo)
        auc, f1_macro, f1_weighted = compute_metrics(y_true_ref, prob_ens, thresholds=None)
        print(f"{'+'.join(combo):50s}  AUC={auc:.4f}  Macro-F1={f1_macro:.4f}")
        if f1_macro > best_score:
            best_score = f1_macro
            best_f1_weighted = f1_weighted
            best_auc = auc
            best_combo = combo
            best_probs = prob_ens

    print("\n===== 验证集上最优组合（按 Macro-F1 排序） =====")
    print(f"最优组合: {best_combo}")
    print(f" AUC         = {best_auc:.4f}")
    print(f" Macro-F1    = {best_score:.4f}")
    # print(f" Weighted-F1 = {best_f1_weighted:.4f}")

    # 进一步：对最优组合做加权搜索 & logistic stacking
    def search_weighted_average(selected):
        pred_list = [preds[n] for n in selected]
        grid = np.linspace(0.0, 1.0, 11)
        best_f1 = -1.0
        best_w = None
        for weights in itertools.product(grid, repeat=len(selected)):
            if sum(weights) == 0:
                continue
            w = np.array(weights)
            w = w / w.sum()
            ens = np.tensordot(w, np.stack(pred_list, axis=0), axes=1)
            f1 = compute_metrics(y_true_ref, ens, thresholds=None)[1]
            if f1 > best_f1:
                best_f1 = f1
                best_w = w
        return best_w, best_f1

    if best_combo:
        best_w, best_f1_w = search_weighted_average(best_combo)
        print(f"\n>>> 最优组合 {best_combo} 加权搜索最佳 F1={best_f1_w:.4f}, 权重={best_w}")
        if best_w is not None:
            best_probs = np.tensordot(best_w, np.stack([preds[n] for n in best_combo], axis=0), axes=1)

        # logistic stacking
        pred_list = [preds[n] for n in best_combo]
        m, n, c = np.stack(pred_list, axis=0).shape
        feats = np.stack(pred_list, axis=0).transpose(1, 2, 0)  # (N, C, M)
        stacked_probs = np.zeros((n, c), dtype=np.float32)
        for cls in range(c):
            X = feats[:, cls, :]
            y = y_true_ref[:, cls]
            clf = LogisticRegression(max_iter=200, n_jobs=1, class_weight="balanced")
            clf.fit(X, y)
            stacked_probs[:, cls] = clf.predict_proba(X)[:, 1]
        auc_stack, f1_stack_macro, f1_stack_weighted = compute_metrics(y_true_ref, stacked_probs, thresholds=None)
        print(f">>> Logistic stacking on best combo: AUC={auc_stack:.4f}, Macro-F1={f1_stack_macro:.4f}")
        if f1_stack_macro > best_score:
            best_score = f1_stack_macro
            best_f1_weighted = f1_stack_weighted
            best_auc = auc_stack
            best_probs = stacked_probs
            best_combo = ("stacking",) + best_combo

    # 保存最佳集成预测
    if best_probs is not None:
        np.savez(os.path.join(SAVE_DIR, "val_preds_best_ensemble.npz"),
                 y_true=y_true_ref,
                 y_pred_prob=best_probs)
        print(f"已将最优集成的验证集预测保存到 saved_models/val_preds_best_ensemble.npz")
    
    # 关闭日志记录
    close_logger()


if __name__ == "__main__":
    main()
