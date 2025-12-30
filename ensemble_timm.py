# ensemble_timm.py
"""
Timm 模型集成评估脚本（优化版）：
- F1 优化：使用高 F1 模型，只输出 F1
- AUC 优化：使用高 AUC 模型，只输出 AUC
- 多架构 AUC 优化：使用多种不同架构模型

优化特性：
- 使用差分进化算法替代网格搜索（效率提升 10-100x）
- 使用 Brent 方法进行精细阈值优化
- 支持交叉验证选择集成权重（避免过拟合验证集）
- 概率校准 (Probability Calibration)
- Stacking 集成（元学习器）
- 联合阈值优化
"""
import os
import itertools
import warnings

import numpy as np
from sklearn.metrics import f1_score as sklearn_f1_score
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize, differential_evolution, brent
# from scipy.special import logit, expit  # 预留用于温度缩放
from tqdm import tqdm

# 屏蔽 sklearn 在 CV 中因单类别导致的 AUC 警告
warnings.filterwarnings("ignore", message="Only one class is present in y_true")

from src.cxr_config import SAVE_DIR
from src.metrics_utils import compute_metrics, search_best_thresholds
from src.log_utils import setup_logger, close_logger


# ========== 优化配置 ==========
USE_CROSS_VALIDATION = False   # 是否使用交叉验证选择权重
CV_FOLDS = 5                  # 交叉验证折数
USE_FAST_OPTIMIZATION = True  # 使用差分进化替代网格搜索

# ========== 高级优化配置 ==========
USE_PROBABILITY_CALIBRATION = True  # 是否使用概率校准
CALIBRATION_METHOD = "isotonic"     # 校准方法: "isotonic" 或 "platt"
USE_JOINT_THRESHOLD = True          # 是否使用联合阈值优化


# ========== 模型配置 ==========

# F1 优化集成（2个高F1模型）- 基准组
F1_MODELS_2 = [
    "convnext_base_in22k_512",      # F1=0.3454 ★
    "convnext_small_in22k_384",     # F1=0.3420
]

# F1 优化集成（3个模型）- 对照组
F1_MODELS_3 = [
    "convnext_base_in22k_512",      # F1=0.3454 ★
    "convnext_small_in22k_384",     # F1=0.3420
    "convnext_small_in22k_512",     # F1=0.3406
]

# 兼容旧代码
F1_MODELS = F1_MODELS_2

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
    "swin_base_in22k_384",          # Swin Transformer, AUC=0.8015
    "coat_lite_medium_384",         # CoaT, AUC=0.8199 ★
]

# 所有需要加载的模型（去重）
ALL_MODELS = list(set(F1_MODELS_2 + F1_MODELS_3 + AUC_MODELS + DIVERSE_AUC_MODELS))


def load_npz_file(npz_path: str):
    """
    加载单个 npz 文件并提取指标
    
    Returns:
        y_true, y_pred_prob, best_metrics, source_path
    """
    data = np.load(npz_path)
    y_true = data["y_true"]
    y_pred_prob = data["y_pred_prob"]
    
    # 提取最佳指标
    best_metrics = {}
    # 新格式
    if "best_f1" in data:
        best_metrics["best_f1"] = float(data["best_f1"])
    if "auc_at_best_f1" in data:
        best_metrics["auc_at_best_f1"] = float(data["auc_at_best_f1"])
    if "best_auc" in data:
        best_metrics["best_auc"] = float(data["best_auc"])
    if "f1_at_best_auc" in data:
        best_metrics["f1_at_best_auc"] = float(data["f1_at_best_auc"])
    # 旧格式兼容
    if "best_f1_macro" in data:
        best_metrics["best_f1_macro"] = float(data["best_f1_macro"])
    if "thresholds" in data:
        best_metrics["thresholds"] = data["thresholds"]
    
    return y_true, y_pred_prob, best_metrics


def load_all_versions(name: str):
    """
    加载单个模型的所有可用 npz 版本
    
    Args:
        name: 模型名称
    
    Returns:
        versions: dict, 键为版本名称 ("old", "f1", "auc")，值为 (y_true, y_pred_prob, metrics, path)
    """
    versions = {}
    
    # 旧格式: val_preds_{name}.npz
    npz_path_old = os.path.join(SAVE_DIR, f"val_preds_{name}.npz")
    if os.path.exists(npz_path_old):
        try:
            y_true, y_pred_prob, metrics = load_npz_file(npz_path_old)
            versions["old"] = (y_true, y_pred_prob, metrics, npz_path_old)
        except Exception as e:
            print(f"  [警告] 加载旧格式文件失败: {e}")
    
    # 新格式 F1: val_preds_{name}_f1.npz
    npz_path_f1 = os.path.join(SAVE_DIR, f"val_preds_{name}_f1.npz")
    if os.path.exists(npz_path_f1):
        try:
            y_true, y_pred_prob, metrics = load_npz_file(npz_path_f1)
            versions["f1"] = (y_true, y_pred_prob, metrics, npz_path_f1)
        except Exception as e:
            print(f"  [警告] 加载 F1 格式文件失败: {e}")
    
    # 新格式 AUC: val_preds_{name}_auc.npz
    npz_path_auc = os.path.join(SAVE_DIR, f"val_preds_{name}_auc.npz")
    if os.path.exists(npz_path_auc):
        try:
            y_true, y_pred_prob, metrics = load_npz_file(npz_path_auc)
            versions["auc"] = (y_true, y_pred_prob, metrics, npz_path_auc)
        except Exception as e:
            print(f"  [警告] 加载 AUC 格式文件失败: {e}")
    
    return versions


def evaluate_single_model(name: str, model_results_f1: dict, model_results_auc: dict,
                          for_f1: bool = True, for_auc: bool = True):
    """
    评估单个模型的性能，从所有可用版本中选择最优的 F1 和 AUC 版本
    
    对于 F1 优化集成：从所有版本中选择 F1 最高的
    对于 AUC 优化集成：从所有版本中选择 AUC 最高的
    
    Args:
        name: 模型名称
        model_results_f1: 存储 F1 最优版本结果的字典
        model_results_auc: 存储 AUC 最优版本结果的字典
        for_f1: 是否参与 F1 集成
        for_auc: 是否参与 AUC 集成
    """
    # 构建用途标签
    usage = []
    if for_f1:
        usage.append("F1")
    if for_auc:
        usage.append("AUC")
    usage_str = "+".join(usage) if usage else "未使用"
    
    print(f"\n===== 单模型 [{name}] ({usage_str}集成) =====")
    
    # 加载所有可用版本
    versions = load_all_versions(name)
    
    if not versions:
        print(f"  [错误] 找不到任何可用的 npz 文件")
        return
    
    print(f"  找到 {len(versions)} 个版本: {list(versions.keys())}")
    
    # 计算每个版本的 AUC 和 F1
    version_scores = {}
    for ver_name, (y_true, y_pred_prob, metrics, path) in versions.items():
        thresholds = search_best_thresholds(y_true, y_pred_prob)
        auc_val, f1_val, _ = compute_metrics(y_true, y_pred_prob, thresholds)
        
        # 优先使用 npz 中保存的指标（训练时记录的），否则用重新计算的
        if ver_name == "f1":
            saved_f1 = metrics.get("best_f1", f1_val)
            saved_auc = metrics.get("auc_at_best_f1", auc_val)
        elif ver_name == "auc":
            saved_auc = metrics.get("best_auc", auc_val)
            saved_f1 = metrics.get("f1_at_best_auc", f1_val)
        else:  # old format
            saved_f1 = metrics.get("best_f1_macro", metrics.get("best_f1", f1_val))
            saved_auc = auc_val  # 旧格式可能没有保存 AUC
        
        version_scores[ver_name] = {
            "y_true": y_true,
            "y_pred_prob": y_pred_prob,
            "thresholds": thresholds,
            "auc": saved_auc,
            "f1": saved_f1,
            "computed_auc": auc_val,
            "computed_f1": f1_val,
            "metrics": metrics,
            "path": path,
        }
        
        # 根据用途显示相关指标
        if for_f1 and for_auc:
            print(f"    [{ver_name:>3}] AUC={saved_auc:.4f}, F1={saved_f1:.4f} (来自 {os.path.basename(path)})")
        elif for_f1:
            print(f"    [{ver_name:>3}] F1={saved_f1:.4f} (来自 {os.path.basename(path)})")
        elif for_auc:
            print(f"    [{ver_name:>3}] AUC={saved_auc:.4f} (来自 {os.path.basename(path)})")
    
    # 选择 F1 最优版本（仅当参与 F1 集成时）
    if for_f1:
        best_f1_ver = max(version_scores.items(), key=lambda x: x[1]["f1"])
        best_f1_name, best_f1_data = best_f1_ver
        
        model_results_f1[name] = {
            "name": name,
            "y_true": best_f1_data["y_true"],
            "y_pred_prob": best_f1_data["y_pred_prob"],
            "auc": best_f1_data["auc"],
            "f1_macro": best_f1_data["f1"],
            "thresholds": best_f1_data["thresholds"],
            "best_metrics": best_f1_data["metrics"],
            "source_version": best_f1_name,
            "source_path": best_f1_data["path"],
        }
        print(f"  >>> 选择 F1 最优: [{best_f1_name}] F1={best_f1_data['f1']:.4f}")
    
    # 选择 AUC 最优版本（仅当参与 AUC 集成时）
    if for_auc:
        best_auc_ver = max(version_scores.items(), key=lambda x: x[1]["auc"])
        best_auc_name, best_auc_data = best_auc_ver
        
        model_results_auc[name] = {
            "name": name,
            "y_true": best_auc_data["y_true"],
            "y_pred_prob": best_auc_data["y_pred_prob"],
            "auc": best_auc_data["auc"],
            "f1_macro": best_auc_data["f1"],
            "thresholds": best_auc_data["thresholds"],
            "best_metrics": best_auc_data["metrics"],
            "source_version": best_auc_name,
            "source_path": best_auc_data["path"],
        }
        print(f"  >>> 选择 AUC 最优: [{best_auc_name}] AUC={best_auc_data['auc']:.4f}")


# ========== 阈值搜索工具 ==========

def search_best_thresholds_refined(y_true: np.ndarray, y_pred_prob: np.ndarray) -> np.ndarray:
    """
    两阶段阈值搜索：先粗搜，再局部细化
    注意：此函数保留作为备选方案，当前主要使用 search_best_thresholds_brent()
    """
    num_classes = y_true.shape[1]
    best_thresholds = np.full(num_classes, 0.5, dtype=np.float32)
    
    # 阶段 1: 粗搜 (0.1-0.9, 步长 0.05)
    coarse_grid = np.linspace(0.1, 0.9, 17)
    
    for c in range(num_classes):
        best_f1 = -1.0
        for t in coarse_grid:
            y_pred_c = (y_pred_prob[:, c] >= t).astype(int)
            f1_c = sklearn_f1_score(y_true[:, c], y_pred_c, zero_division=0)
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
            f1_c = sklearn_f1_score(y_true[:, c], y_pred_c, zero_division=0)
            if f1_c > best_f1:
                best_f1 = f1_c
                best_thresholds[c] = t
    
    return best_thresholds


def search_best_thresholds_fast(y_true: np.ndarray, y_pred_prob: np.ndarray) -> np.ndarray:
    """
    快速阈值搜索（用于优化目标函数中的快速评估）
    - 使用 9 点网格搜索，速度快
    - 精度略低于 Brent，但足够用于权重优化
    """
    num_classes = y_true.shape[1]
    best_thresholds = np.zeros(num_classes, dtype=np.float32)
    grid = np.linspace(0.1, 0.9, 9)
    
    for c in range(num_classes):
        y_c = y_true[:, c]
        p_c = y_pred_prob[:, c]
        
        if y_c.sum() < 2 or (1 - y_c).sum() < 2:
            best_thresholds[c] = 0.5
            continue
        
        best_f1 = -1.0
        best_t = 0.5
        for t in grid:
            y_pred_c = (p_c >= t).astype(int)
            f1 = sklearn_f1_score(y_c, y_pred_c, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thresholds[c] = best_t
    
    return best_thresholds


def compute_f1_with_optimal_threshold(y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
    """
    计算使用最优阈值的 Macro-F1（用于优化目标函数）
    """
    thresholds = search_best_thresholds_fast(y_true, y_pred_prob)
    _, f1_macro, _ = compute_metrics(y_true, y_pred_prob, thresholds)
    return f1_macro


def search_best_thresholds_brent(y_true: np.ndarray, y_pred_prob: np.ndarray) -> np.ndarray:
    """
    使用 Brent 方法进行精细阈值优化
    - 连续优化，精度更高
    - 速度与网格搜索相当，但结果更优
    """
    num_classes = y_true.shape[1]
    best_thresholds = np.zeros(num_classes, dtype=np.float32)
    
    for c in range(num_classes):
        y_c = y_true[:, c]
        p_c = y_pred_prob[:, c]
        
        # 如果类别样本太少，使用默认阈值
        if y_c.sum() < 2 or (1 - y_c).sum() < 2:
            best_thresholds[c] = 0.5
            continue
        
        def neg_f1(t):
            """负 F1（用于最小化）"""
            y_pred_c = (p_c >= t).astype(int)
            return -sklearn_f1_score(y_c, y_pred_c, zero_division=0)
        
        # 先粗搜找到大致范围
        coarse_grid = np.linspace(0.1, 0.9, 9)
        best_coarse_t = 0.5
        best_coarse_f1 = -1.0
        for t in coarse_grid:
            f1 = -neg_f1(t)
            if f1 > best_coarse_f1:
                best_coarse_f1 = f1
                best_coarse_t = t
        
        # 使用 Brent 方法在粗搜结果附近精细优化
        try:
            left = max(0.05, best_coarse_t - 0.15)
            right = min(0.95, best_coarse_t + 0.15)
            result = brent(neg_f1, brack=(left, best_coarse_t, right), tol=1e-4)
            best_thresholds[c] = np.clip(result, 0.05, 0.95)
        except Exception:
            # 如果 Brent 失败，回退到粗搜结果
            best_thresholds[c] = best_coarse_t
    
    return best_thresholds


# ========== 高级优化方法 ==========

def calibrate_probabilities(y_true: np.ndarray, y_pred_prob: np.ndarray, 
                            method: str = "isotonic", n_folds: int = 5) -> np.ndarray:
    """
    对模型概率进行校准（使用交叉验证避免数据泄露）
    
    Args:
        y_true: 真实标签 (n_samples, n_classes)
        y_pred_prob: 预测概率 (n_samples, n_classes)
        method: 校准方法
            - "isotonic": 非参数方法，使用保序回归
            - "platt": 参数方法，使用 Sigmoid/Platt scaling
        n_folds: 交叉验证折数，用于避免在同一数据上 fit 和 evaluate
    
    Returns:
        calibrated_prob: 校准后的概率
    """
    n_samples, n_classes = y_pred_prob.shape
    calibrated_prob = np.zeros_like(y_pred_prob)
    
    # 使用交叉验证避免数据泄露
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for c in range(n_classes):
        y_c = y_true[:, c]
        p_c = y_pred_prob[:, c]
        
        # 样本太少则跳过校准
        if y_c.sum() < 10 or (1 - y_c).sum() < 10:
            calibrated_prob[:, c] = p_c
            continue
        
        if method == "isotonic":
            # 使用 K-Fold 交叉验证进行校准
            for train_idx, val_idx in kf.split(y_c):
                ir = IsotonicRegression(out_of_bounds='clip')
                # 在训练折上拟合
                ir.fit(p_c[train_idx], y_c[train_idx])
                # 在验证折上预测（避免数据泄露）
                calibrated_prob[val_idx, c] = ir.transform(p_c[val_idx])
        elif method == "platt":
            # Platt scaling 也使用交叉验证
            for train_idx, val_idx in kf.split(y_c):
                p_train, y_train = p_c[train_idx], y_c[train_idx]
                p_val = p_c[val_idx]
                
                def nll(params):
                    a, b = params
                    p_cal = 1 / (1 + np.exp(-(a * p_train + b)))
                    p_cal = np.clip(p_cal, 1e-7, 1 - 1e-7)
                    return -np.mean(y_train * np.log(p_cal) + (1 - y_train) * np.log(1 - p_cal))
                
                result = minimize(nll, [1.0, 0.0], method='L-BFGS-B')
                a, b = result.x
                calibrated_prob[val_idx, c] = 1 / (1 + np.exp(-(a * p_val + b)))
        else:
            calibrated_prob[:, c] = p_c
    
    return calibrated_prob


def calibrate_all_models(preds_list: list, y_true: np.ndarray, 
                         method: str = "isotonic") -> list:
    """
    对所有模型的预测进行概率校准（使用交叉验证避免数据泄露）
    
    Args:
        preds_list: 模型预测列表
        y_true: 真实标签
        method: 校准方法
    
    Returns:
        calibrated_preds: 校准后的预测列表
    
    Note:
        使用 5-Fold 交叉验证进行校准，在每个 fold 上用训练折拟合校准器，
        在验证折上预测，避免在同一数据上 fit 和 evaluate 导致的数据泄露。
    """
    print(f"[概率校准] 使用 {method} 方法 (5-Fold CV) 校准 {len(preds_list)} 个模型...")
    calibrated_preds = []
    for i, prob in tqdm(enumerate(preds_list), total=len(preds_list), 
                        desc="  概率校准", leave=False):
        cal_prob = calibrate_probabilities(y_true, prob, method=method)
        calibrated_preds.append(cal_prob)
    print(f"  ✓ 完成! 已校准 {len(preds_list)} 个模型")
    return calibrated_preds


def search_thresholds_joint(y_true: np.ndarray, y_pred_prob: np.ndarray,
                            maxiter: int = 200) -> np.ndarray:
    """
    联合优化所有类别的阈值
    
    优点：
    - 考虑类别间的依赖关系
    - 全局最优而非局部最优
    
    Args:
        y_true: 真实标签
        y_pred_prob: 预测概率
        maxiter: 最大迭代次数
    
    Returns:
        optimal_thresholds: 优化后的阈值
    """
    n_classes = y_true.shape[1]
    
    # 定义搜索边界
    THRESH_MIN, THRESH_MAX = 0.05, 0.95
    bounds = [(THRESH_MIN, THRESH_MAX)] * n_classes
    
    # 先用 Brent 方法获取初始阈值，并确保在边界内
    print("  获取初始阈值...")
    init_thresholds = search_best_thresholds_brent(y_true, y_pred_prob)
    init_thresholds = np.clip(init_thresholds, THRESH_MIN, THRESH_MAX)
    _, init_f1, _ = compute_metrics(y_true, y_pred_prob, init_thresholds)
    
    def neg_macro_f1(thresholds):
        """目标函数: 负 Macro-F1"""
        thresholds = np.clip(thresholds, THRESH_MIN, THRESH_MAX)
        y_pred = (y_pred_prob >= thresholds).astype(int)
        f1s = []
        for c in range(n_classes):
            f1_c = sklearn_f1_score(y_true[:, c], y_pred[:, c], zero_division=0)
            f1s.append(f1_c)
        return -np.mean(f1s)
    
    pbar = tqdm(total=maxiter, desc="  联合阈值优化", unit="iter", leave=False)
    
    def callback(xk, convergence):
        current_f1 = -neg_macro_f1(xk)
        pbar.set_postfix({"F1": f"{current_f1:.4f}"})
        pbar.update(1)
    
    # 使用差分进化联合优化
    result = differential_evolution(
        neg_macro_f1, bounds, 
        x0=init_thresholds,  # 使用 Brent 结果作为初始值（已 clip 到边界内）
        maxiter=maxiter, 
        seed=42,
        disp=False, 
        workers=1,
        tol=1e-5,
        mutation=(0.5, 1.0),
        recombination=0.7,
        callback=callback,
    )
    pbar.close()
    
    optimal_thresholds = np.clip(result.x, THRESH_MIN, THRESH_MAX)
    _, final_f1, _ = compute_metrics(y_true, y_pred_prob, optimal_thresholds)
    
    improvement = final_f1 - init_f1
    print(f"  ✓ 完成! 初始F1: {init_f1:.4f} → 优化F1: {final_f1:.4f} ({improvement:+.4f})")
    
    return optimal_thresholds


def optimized_ensemble_pipeline(preds_list: list, y_true: np.ndarray, 
                                 model_names: list, target_metric: str = "auc"):
    """
    优化的集成流程：整合概率校准、Stacking 和联合阈值优化
    
    Args:
        preds_list: 模型预测列表
        y_true: 真实标签
        model_names: 模型名称列表
        target_metric: 目标指标 "auc" 或 "f1"
    
    Returns:
        final_prob: 最终预测概率
        final_thresholds: 最终阈值
        metrics: 性能指标字典
        calibrated_simple_avg: 校准后简单平均结果 (prob, auc, f1) 或 None
    """
    print("\n" + "=" * 60)
    print("【优化集成流程】")
    print("=" * 60)
    
    current_preds = preds_list
    calibrated_simple_avg = None
    
    # Step 1: 概率校准
    if USE_PROBABILITY_CALIBRATION:
        print("\n>>> Step 1: 概率校准")
        current_preds = calibrate_all_models(
            current_preds, y_true, method=CALIBRATION_METHOD
        )
        # 评估校准后的简单平均
        cal_avg_prob = np.stack(current_preds, axis=0).mean(axis=0)
        cal_avg_thr = search_best_thresholds_brent(y_true, cal_avg_prob)
        cal_auc, cal_f1, _ = compute_metrics(y_true, cal_avg_prob, cal_avg_thr)
        print(f"  校准后简单平均: AUC={cal_auc:.4f}, F1={cal_f1:.4f}")
        calibrated_simple_avg = (cal_avg_prob, cal_avg_thr, cal_auc, cal_f1)
    else:
        print("\n>>> Step 1: 跳过概率校准")
    
    # Step 2: 加权平均集成
    print("\n>>> Step 2: 加权平均集成")
    if target_metric == "f1":
        final_prob, _, avg_thr, avg_f1 = weighted_average_f1(
            current_preds, y_true, model_names
        )
        avg_auc, _, _ = compute_metrics(y_true, final_prob, thresholds=None)
    else:
        final_prob, _, avg_auc = weighted_average_auc(
            current_preds, y_true, model_names
        )
        avg_thr = search_best_thresholds_brent(y_true, final_prob)
        _, avg_f1, _ = compute_metrics(y_true, final_prob, avg_thr)
    
    # Step 3: 联合阈值优化 (仅对 F1 优化有效)
    if USE_JOINT_THRESHOLD and target_metric == "f1":
        print("\n>>> Step 3: 联合阈值优化")
        final_thresholds = search_thresholds_joint(y_true, final_prob)
    else:
        print("\n>>> Step 3: 使用 Brent 阈值")
        final_thresholds = avg_thr
    
    # 最终评估
    final_auc, final_f1, _ = compute_metrics(y_true, final_prob, final_thresholds)
    
    print("\n" + "-" * 40)
    print(f"【优化集成最终结果】")
    print(f"  AUC: {final_auc:.4f}")
    print(f"  Macro-F1: {final_f1:.4f}")
    print("-" * 40)
    
    metrics = {
        "auc": final_auc,
        "f1": final_f1,
    }
    
    return final_prob, final_thresholds, metrics, calibrated_simple_avg


# ========== 额外集成策略（从老版本迁移）==========

def nonneg_stacking(preds_list: list, y_true: np.ndarray, model_names: list = None):
    """
    非负最小二乘 Stacking：约束权重 >= 0，更接近加权平均的物理意义
    
    对每个类别独立求解: min ||Xw - y||^2  s.t. w >= 0
    然后归一化权重，使其和为 1。
    
    Args:
        preds_list: 各模型的预测概率列表
        y_true: 真实标签
        model_names: 模型名称列表（用于日志输出）
    
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
    
    # 显示平均权重
    mean_weights = weights_per_class.mean(axis=0)
    if model_names:
        print(f"  NNLS 平均权重: {dict(zip(model_names, mean_weights.round(3)))}")
    
    return stacked_probs, weights_per_class


def logistic_stacking(preds_list: list, y_true: np.ndarray, model_names: list = None,
                      n_splits: int = 5, C: float = 0.1):
    """
    Logistic Stacking：使用交叉验证避免数据泄露
    
    对每个类别训练一个逻辑回归元分类器，使用 K-Fold 交叉验证
    生成 out-of-fold 预测，避免在同一数据上训练和评估。
    
    Args:
        preds_list: 各模型的预测概率列表
        y_true: 真实标签
        model_names: 模型名称列表
        n_splits: K-Fold 的折数
        C: 正则化强度的倒数，值越小正则化越强（默认 0.1）
    
    Returns:
        stacked_probs: 集成后的概率
    """
    from sklearn.linear_model import LogisticRegression
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
                    C=C,  # L2 正则化
                    max_iter=500,
                    solver='lbfgs',
                    class_weight="balanced"
                )
                clf.fit(X[train_idx], y[train_idx])
                oof_preds[val_idx] = clf.predict_proba(X[val_idx])[:, 1]
            
            stacked_probs[:, cls] = oof_preds
    
    return stacked_probs


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


def diversity_weighted_ensemble(preds_list: list, model_aucs: list, model_names: list = None,
                                 diversity_weight: float = 0.3):
    """
    多样性感知的加权集成
    
    综合考虑模型性能和模型间的多样性，避免高度相关的模型主导集成。
    
    综合权重 = (1 - diversity_weight) × 性能权重 + diversity_weight × 多样性权重
    
    Args:
        preds_list: 各模型的预测概率列表
        model_aucs: 各模型的 AUC 分数
        model_names: 模型名称列表
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
    
    # 显示权重信息
    if model_names:
        print(f"  性能权重: {dict(zip(model_names, perf_weights.round(4)))}")
        print(f"  多样性分数: {dict(zip(model_names, diversity_scores.round(4)))}")
        print(f"  综合权重: {dict(zip(model_names, combined_weights.round(4)))}")
    
    return ens_prob, combined_weights, perf_weights, diversity_scores


# ========== F1 优化集成策略 ==========

def weighted_average_f1_grid(preds_list: list, y_true: np.ndarray, model_names: list, top_k=200):
    """
    三阶段加权平均集成（来自老版本，更可靠）：
    - 阶段 1: 固定阈值(0.5)快速筛选 Top-K 权重组合
    - 阶段 2: 对 Top-K 进行阈值搜索，找最优
    - 阶段 3: 在最佳权重附近进行局部精调
    """
    m = len(preds_list)
    grid = np.linspace(0.0, 1.0, 11)  # 0.0, 0.1, ..., 1.0
    total_combinations = len(grid) ** m
    stacked_preds = np.stack(preds_list, axis=0)
    
    # ========== 阶段 1: 粗筛（固定阈值 0.5）==========
    print(f"[阶段 1] 粗筛: {total_combinations} 个权重组合...")
    
    candidates = []
    
    pbar = tqdm(
        itertools.product(grid, repeat=m),
        total=total_combinations,
        desc=f"  粗筛 ({m}模型)",
        leave=False,
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
    
    print(f"  Top-{min(top_k, len(candidates))} 粗筛得分: {top_candidates[-1][0]:.4f} ~ {top_candidates[0][0]:.4f}")
    
    # ========== 阶段 2: 精选（阈值搜索）==========
    print(f"[阶段 2] 精选: 对 Top-{min(top_k, len(candidates))} 进行阈值搜索...")
    
    stage2_best_score = -1.0
    stage2_best_weights = None
    
    for _, w in tqdm(top_candidates, desc="  阈值搜索", leave=False):
        ens_prob = np.tensordot(w, stacked_preds, axes=1)
        thresholds = search_best_thresholds(y_true, ens_prob)
        _, score, _ = compute_metrics(y_true, ens_prob, thresholds)
        
        if score > stage2_best_score:
            stage2_best_score = score
            stage2_best_weights = w
    
    print(f"  精选最佳 F1: {stage2_best_score:.4f} (粗筛提升 +{stage2_best_score - top_candidates[0][0]:.4f})")
    
    # ========== 阶段 3: 局部精调 ==========
    fine_grid = np.linspace(-0.03, 0.03, 7)  # 更细的精调
    fine_combinations = len(fine_grid) ** m
    
    print(f"[阶段 3] 局部精调: {fine_combinations} 个组合...")
    
    best_score = stage2_best_score
    best_weights = stage2_best_weights.copy()
    
    pbar = tqdm(
        itertools.product(fine_grid, repeat=m),
        total=fine_combinations,
        desc="  局部精调",
        leave=False,
    )
    
    for offsets in pbar:
        w = stage2_best_weights + np.array(offsets)
        w = np.clip(w, 0, 1)
        
        if w.sum() == 0:
            continue
        w = w / w.sum()
        
        ens_prob = np.tensordot(w, stacked_preds, axes=1)
        thresholds = search_best_thresholds(y_true, ens_prob)
        _, score, _ = compute_metrics(y_true, ens_prob, thresholds)
        
        if score > best_score:
            best_score = score
            best_weights = w.copy()
            pbar.set_postfix({"best": f"{best_score:.4f}"})
    
    # 最终结果
    best_prob = np.tensordot(best_weights, stacked_preds, axes=1)
    best_thresholds = search_best_thresholds(y_true, best_prob)
    _, best_f1, _ = compute_metrics(y_true, best_prob, best_thresholds)
    
    print(f"  ✓ 最终 F1: {best_f1:.4f} (精调提升 +{best_f1 - stage2_best_score:.4f})")
    print(f"    权重: {dict(zip(model_names, best_weights.round(4)))}")
    
    return best_prob, best_weights, best_thresholds, best_f1


def weighted_average_f1_cv(preds_list: list, y_true: np.ndarray, model_names: list, n_folds: int = 5):
    """
    F1 优化的加权平均集成（交叉验证版本）
    - 使用 K-Fold 交叉验证选择权重，避免过拟合验证集
    - 在每个 fold 上独立优化权重，最终取平均
    - 使用动态阈值搜索
    """
    m = len(preds_list)
    stacked_preds = np.stack(preds_list, axis=0)
    
    print(f"\n[{n_folds}-Fold 交叉验证] 搜索 {m} 模型的最优权重...")
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_weights = []
    fold_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(y_true)):
        y_train = y_true[train_idx]
        preds_train = stacked_preds[:, train_idx, :]
        
        def objective(w):
            w = np.abs(w)
            if w.sum() == 0:
                return 1.0
            w = w / w.sum()
            ens_prob = np.tensordot(w, preds_train, axes=1)
            # 使用动态阈值搜索
            f1_macro = compute_f1_with_optimal_threshold(y_train, ens_prob)
            return -f1_macro
        
        bounds = [(0, 1)] * m
        result = differential_evolution(
            objective, bounds, maxiter=50, seed=42+fold_idx, 
            disp=False, workers=1, tol=1e-3
        )
        
        w = np.abs(result.x)
        w = w / w.sum()
        fold_weights.append(w)
        
        # 在验证折上评估（使用动态阈值）
        y_val = y_true[val_idx]
        preds_val = stacked_preds[:, val_idx, :]
        ens_prob_val = np.tensordot(w, preds_val, axes=1)
        f1_val = compute_f1_with_optimal_threshold(y_val, ens_prob_val)
        fold_scores.append(f1_val)
        
        print(f"  Fold {fold_idx+1}: F1={f1_val:.4f}, weights={w.round(3)}")
    
    # 平均权重
    avg_weights = np.mean(fold_weights, axis=0)
    avg_weights = avg_weights / avg_weights.sum()
    
    # 使用平均权重在整个数据集上计算
    best_prob = np.tensordot(avg_weights, stacked_preds, axes=1)
    best_thresholds = search_best_thresholds_brent(y_true, best_prob)
    _, best_f1, _ = compute_metrics(y_true, best_prob, best_thresholds)
    
    print(f"\n[CV F1 集成完成]")
    print(f"  CV 平均 F1: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"  最终 Macro-F1: {best_f1:.4f}")
    print(f"  平均权重: {dict(zip(model_names, avg_weights.round(4)))}")
    
    return best_prob, avg_weights, best_thresholds, best_f1


def weighted_average_f1(preds_list: list, y_true: np.ndarray, model_names: list, top_k=200):
    """
    F1 优化的加权平均集成
    使用三阶段网格搜索（更可靠）
    """
    return weighted_average_f1_grid(preds_list, y_true, model_names, top_k)


# ========== AUC 优化集成策略 ==========

def weighted_average_auc_fast(preds_list: list, y_true: np.ndarray, model_names: list):
    """
    AUC 优化的加权平均集成（快速版本）
    - 使用差分进化算法替代网格搜索
    """
    m = len(preds_list)
    stacked_preds = np.stack(preds_list, axis=0)
    
    print(f"[差分进化优化] 搜索 {m} 模型的最优 AUC 权重...")
    
    pbar = tqdm(total=100, desc="  AUC权重优化", unit="iter", leave=False)
    
    def objective(w):
        w = np.abs(w)
        if w.sum() == 0:
            return 1.0
        w = w / w.sum()
        ens_prob = np.tensordot(w, stacked_preds, axes=1)
        auc, _, _ = compute_metrics(y_true, ens_prob, thresholds=None)
        return -auc
    
    def callback(xk, convergence):
        current_auc = -objective(xk)
        pbar.set_postfix({"best_AUC": f"{current_auc:.4f}"})
        pbar.update(1)
    
    bounds = [(0, 1)] * m
    result = differential_evolution(
        objective, bounds, maxiter=100, seed=42,
        disp=False, workers=1, updating='deferred', tol=1e-4,
        callback=callback
    )
    pbar.close()
    
    best_weights = np.abs(result.x)
    best_weights = best_weights / best_weights.sum()
    best_prob = np.tensordot(best_weights, stacked_preds, axes=1)
    best_auc, _, _ = compute_metrics(y_true, best_prob, thresholds=None)
    
    print(f"  ✓ 完成! AUC: {best_auc:.4f}")
    print(f"    权重: {dict(zip(model_names, best_weights.round(4)))}")
    
    return best_prob, best_weights, best_auc


def weighted_average_auc_cv(preds_list: list, y_true: np.ndarray, model_names: list, n_folds: int = 5):
    """
    AUC 优化的加权平均集成（交叉验证版本）
    """
    m = len(preds_list)
    stacked_preds = np.stack(preds_list, axis=0)
    
    print(f"\n[{n_folds}-Fold 交叉验证] 搜索 {m} 模型的最优 AUC 权重...")
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_weights = []
    fold_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(y_true)):
        y_train = y_true[train_idx]
        preds_train = stacked_preds[:, train_idx, :]
        
        def objective(w):
            w = np.abs(w)
            if w.sum() == 0:
                return 1.0
            w = w / w.sum()
            ens_prob = np.tensordot(w, preds_train, axes=1)
            auc, _, _ = compute_metrics(y_train, ens_prob, thresholds=None)
            return -auc
        
        bounds = [(0, 1)] * m
        result = differential_evolution(
            objective, bounds, maxiter=50, seed=42+fold_idx,
            disp=False, workers=1, tol=1e-3
        )
        
        w = np.abs(result.x)
        w = w / w.sum()
        fold_weights.append(w)
        
        # 在验证折上评估
        y_val = y_true[val_idx]
        preds_val = stacked_preds[:, val_idx, :]
        ens_prob_val = np.tensordot(w, preds_val, axes=1)
        auc_val, _, _ = compute_metrics(y_val, ens_prob_val, thresholds=None)
        fold_scores.append(auc_val)
        
        print(f"  Fold {fold_idx+1}: AUC={auc_val:.4f}, weights={w.round(3)}")
    
    avg_weights = np.mean(fold_weights, axis=0)
    avg_weights = avg_weights / avg_weights.sum()
    
    best_prob = np.tensordot(avg_weights, stacked_preds, axes=1)
    best_auc, _, _ = compute_metrics(y_true, best_prob, thresholds=None)
    
    print(f"\n[CV AUC 集成完成]")
    print(f"  CV 平均 AUC: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"  最终 AUC: {best_auc:.4f}")
    print(f"  平均权重: {dict(zip(model_names, avg_weights.round(4)))}")
    
    return best_prob, avg_weights, best_auc


def weighted_average_auc(preds_list: list, y_true: np.ndarray, model_names: list):
    """
    AUC 优化的加权平均集成（兼容旧版本接口）
    """
    if USE_CROSS_VALIDATION:
        return weighted_average_auc_cv(preds_list, y_true, model_names, n_folds=CV_FOLDS)
    elif USE_FAST_OPTIMIZATION:
        return weighted_average_auc_fast(preds_list, y_true, model_names)
    else:
        return weighted_average_auc_grid(preds_list, y_true, model_names)


def weighted_average_auc_grid(preds_list: list, y_true: np.ndarray, model_names: list):
    """
    AUC 优化的加权平均集成（原始网格搜索版本）
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


# ========== 主函数 ==========

def main():
    setup_logger("ensemble_timm")
    
    print("=" * 70)
    print("Timm 模型集成评估（优化版）")
    print("=" * 70)
    print(f"SAVE_DIR = {SAVE_DIR}")
    print(f"\n【优化配置】")
    print(f"  交叉验证: {'开启 ({}-Fold)'.format(CV_FOLDS) if USE_CROSS_VALIDATION else '关闭'}")
    print(f"  快速优化: {'开启 (差分进化)' if USE_FAST_OPTIMIZATION else '关闭 (网格搜索)'}")
    print(f"\nF1 优化模型组（{len(F1_MODELS)}模型）: {F1_MODELS}")
    print(f"AUC 优化模型组（{len(AUC_MODELS)}模型）: {AUC_MODELS}")
    print(f"多架构 AUC 模型组（{len(DIVERSE_AUC_MODELS)}模型）: {DIVERSE_AUC_MODELS}")
    print("=" * 70)
    
    # ========== 1. 加载所有模型预测（分别加载 F1 和 AUC 最优版本）==========
    model_results_f1 = {}  # F1 最优版本
    model_results_auc = {}  # AUC 最优版本
    
    # 计算每个模型参与的集成类型
    f1_models_set = set(F1_MODELS_2 + F1_MODELS_3)
    auc_models_set = set(AUC_MODELS + DIVERSE_AUC_MODELS)
    
    print("\n【步骤 1/5】加载单模型预测")
    print("-" * 50)
    
    for name in ALL_MODELS:
        for_f1 = name in f1_models_set
        for_auc = name in auc_models_set
        evaluate_single_model(name, model_results_f1, model_results_auc, 
                              for_f1=for_f1, for_auc=for_auc)
    
    print(f"\n已加载 F1 集成模型 ({len(model_results_f1)}个): {list(model_results_f1.keys())}")
    print(f"已加载 AUC 集成模型 ({len(model_results_auc)}个): {list(model_results_auc.keys())}")
    
    # 获取 y_true（从任一模型）
    all_results = {**model_results_f1, **model_results_auc}
    if not all_results:
        print("[错误] 没有可用的模型")
        return
    
    y_true = list(all_results.values())[0]["y_true"]
    
    f1_results = []
    auc_results = []
    
    # ========== 2. F1 优化集成 - 2模型基准组 ==========
    print("\n" + "=" * 70)
    print("【步骤 2/5】F1 优化集成 - 2模型基准组")
    print("=" * 70)
    
    f1_model_names_2 = [n for n in F1_MODELS_2 if n in model_results_f1]
    
    if len(f1_model_names_2) >= 2:
        f1_preds_2 = [model_results_f1[name]["y_pred_prob"] for name in f1_model_names_2]
        
        print(f"\n使用模型 (F1最优版本):")
        for name in f1_model_names_2:
            f1 = model_results_f1[name]["f1_macro"]
            print(f"  • {name}: F1={f1:.4f}")
        
        # 三阶段加权平均
        print("\n[加权平均集成]")
        prob_2m_1, weights_2m_1, thr_2m_1, score_2m_1 = weighted_average_f1(
            f1_preds_2, y_true, f1_model_names_2
        )
        f1_results.append(("2模型-加权平均", score_2m_1, prob_2m_1, thr_2m_1))
        
    else:
        print(f"[跳过] 2模型组模型不足（需要 2 个，当前 {len(f1_model_names_2)} 个）")
    
    # ========== 2.5 F1 优化集成 - 3模型对照组 ==========
    print("\n" + "=" * 70)
    print("【步骤 2.5/5】F1 优化集成 - 3模型对照组")
    print("=" * 70)
    
    f1_model_names_3 = [n for n in F1_MODELS_3 if n in model_results_f1]
    
    if len(f1_model_names_3) >= 2:
        f1_preds_3 = [model_results_f1[name]["y_pred_prob"] for name in f1_model_names_3]
        
        print(f"\n使用模型 (F1最优版本):")
        for name in f1_model_names_3:
            f1 = model_results_f1[name]["f1_macro"]
            print(f"  • {name}: F1={f1:.4f}")
        
        # 三阶段加权平均
        print("\n[加权平均集成]")
        prob_3m_1, weights_3m_1, thr_3m_1, score_3m_1 = weighted_average_f1(
            f1_preds_3, y_true, f1_model_names_3
        )
        f1_results.append(("3模型-加权平均", score_3m_1, prob_3m_1, thr_3m_1))
        
    else:
        print(f"[跳过] 3模型组模型不足（需要 2 个，当前 {len(f1_model_names_3)} 个）")
    
    # 兼容后续代码
    f1_model_names = f1_model_names_2
    f1_preds = f1_preds_2 if len(f1_model_names_2) >= 2 else []
    
    # ========== 2.6 额外 F1 集成策略（从老版本迁移）==========
    print("\n" + "=" * 70)
    print("【步骤 2.6/5】额外 F1 集成策略")
    print("=" * 70)
    
    # 使用 3 模型组进行额外策略测试（更多模型效果更好）
    extra_model_names = f1_model_names_3 if len(f1_model_names_3) >= 2 else f1_model_names_2
    extra_preds = f1_preds_3 if len(f1_model_names_3) >= 2 else f1_preds_2
    
    if len(extra_model_names) >= 2:
        print(f"\n使用模型: {extra_model_names}")
        
        # 策略 2.6.1: NNLS (非负约束 Stacking)
        print("\n[策略 1/4] NNLS (非负约束 Stacking)")
        nnls_prob, nnls_weights = nonneg_stacking(extra_preds, y_true, extra_model_names)
        nnls_thr = search_best_thresholds_brent(y_true, nnls_prob)
        _, nnls_f1, _ = compute_metrics(y_true, nnls_prob, nnls_thr)
        print(f"  ✓ 完成! F1: {nnls_f1:.4f}")
        f1_results.append(("NNLS", nnls_f1, nnls_prob, nnls_thr))
        
        # 策略 2.6.2: 概率校准 + NNLS（老版本最佳策略）
        print("\n[策略 2/4] 概率校准 + NNLS (老版本最佳)")
        calibrated_extra = calibrate_all_models(extra_preds, y_true, method=CALIBRATION_METHOD)
        cal_nnls_prob, cal_nnls_weights = nonneg_stacking(calibrated_extra, y_true, extra_model_names)
        cal_nnls_thr = search_best_thresholds_brent(y_true, cal_nnls_prob)
        _, cal_nnls_f1, _ = compute_metrics(y_true, cal_nnls_prob, cal_nnls_thr)
        print(f"  ✓ 完成! F1: {cal_nnls_f1:.4f}")
        f1_results.append(("概率校准+NNLS", cal_nnls_f1, cal_nnls_prob, cal_nnls_thr))
        
        # 策略 2.6.3: Logistic Stacking
        print("\n[策略 3/4] Logistic Stacking (5-Fold CV, C=0.1)")
        lr_prob = logistic_stacking(extra_preds, y_true, extra_model_names, n_splits=5, C=0.1)
        lr_thr = search_best_thresholds_brent(y_true, lr_prob)
        _, lr_f1, _ = compute_metrics(y_true, lr_prob, lr_thr)
        print(f"  ✓ 完成! F1: {lr_f1:.4f}")
        f1_results.append(("Logistic Stacking", lr_f1, lr_prob, lr_thr))
        
        # 策略 2.6.4: 多样性加权集成
        print("\n[策略 4/4] 多样性加权集成 (0.7性能 + 0.3多样性)")
        model_f1s = [model_results_f1[n]["f1_macro"] for n in extra_model_names]
        div_prob, div_weights, _, _ = diversity_weighted_ensemble(
            extra_preds, model_f1s, extra_model_names, diversity_weight=0.3
        )
        div_thr = search_best_thresholds_brent(y_true, div_prob)
        _, div_f1, _ = compute_metrics(y_true, div_prob, div_thr)
        print(f"  ✓ 完成! F1: {div_f1:.4f}")
        f1_results.append(("多样性加权", div_f1, div_prob, div_thr))
    else:
        print("[跳过] 模型不足")
    
    # ========== 3. AUC 优化集成（使用 AUC 最优版本）==========
    print("\n" + "=" * 70)
    print("【步骤 3/5】AUC 优化集成")
    print("=" * 70)
    
    auc_model_names = [n for n in AUC_MODELS if n in model_results_auc]
    
    if len(auc_model_names) >= 2:
        auc_preds = [model_results_auc[name]["y_pred_prob"] for name in auc_model_names]
        
        print(f"\n使用模型 (AUC最优版本):")
        for name in auc_model_names:
            auc = model_results_auc[name]["auc"]
            print(f"  • {name}: AUC={auc:.4f}")
        
        # 策略 AUC-1: 加权平均
        print("\n[策略 1/3] 加权平均")
        prob_auc_1, weights_auc_1, score_auc_1 = weighted_average_auc(
            auc_preds, y_true, auc_model_names
        )
        auc_results.append(("3模型加权平均", score_auc_1, prob_auc_1))
        
        # 策略 AUC-2: 简单平均
        print("\n[策略 2/3] 简单平均")
        prob_auc_2 = np.stack(auc_preds, axis=0).mean(axis=0)
        score_auc_2, _, _ = compute_metrics(y_true, prob_auc_2, thresholds=None)
        print(f"  ✓ 完成! AUC: {score_auc_2:.4f}")
        auc_results.append(("3模型简单平均", score_auc_2, prob_auc_2))
        
        # 策略 AUC-3: 多样性加权
        print("\n[策略 3/3] 多样性加权 (0.6性能 + 0.4多样性)")
        model_aucs_3 = [model_results_auc[n]["auc"] for n in auc_model_names]
        prob_auc_3, _, _, _ = diversity_weighted_ensemble(
            auc_preds, model_aucs_3, auc_model_names, diversity_weight=0.4
        )
        score_auc_3, _, _ = compute_metrics(y_true, prob_auc_3, thresholds=None)
        print(f"  ✓ 完成! AUC: {score_auc_3:.4f}")
        auc_results.append(("3模型多样性加权", score_auc_3, prob_auc_3))
        
    else:
        print(f"[跳过] AUC 模型不足（需要 2 个，当前 {len(auc_model_names)} 个）")
    
    # ========== 4. 多架构 AUC 优化集成（使用 AUC 最优版本）==========
    print("\n" + "=" * 70)
    print("【步骤 4/5】多架构 AUC 集成")
    print("=" * 70)
    
    diverse_model_names = [n for n in DIVERSE_AUC_MODELS if n in model_results_auc]
    
    if len(diverse_model_names) >= 2:
        diverse_preds = [model_results_auc[name]["y_pred_prob"] for name in diverse_model_names]
        
        print(f"\n使用模型 (AUC最优版本):")
        for name in diverse_model_names:
            auc = model_results_auc[name]["auc"]
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
            print(f"  • {name} [{arch}]: AUC={auc:.4f}")
        
        # 策略 Div-1: 差分进化优化权重
        print("\n[策略 1/3] 差分进化优化权重")
        prob_div_1, weights_div_1, score_div_1 = weighted_average_auc(
            diverse_preds, y_true, diverse_model_names
        )
        auc_results.append(("4架构差分进化", score_div_1, prob_div_1))
        
        # 策略 Div-2: 简单平均
        print("\n[策略 2/3] 简单平均")
        prob_div_2 = np.stack(diverse_preds, axis=0).mean(axis=0)
        score_div_2, _, _ = compute_metrics(y_true, prob_div_2, thresholds=None)
        print(f"  ✓ 完成! AUC: {score_div_2:.4f}")
        auc_results.append(("4架构简单平均", score_div_2, prob_div_2))
        
        # 策略 Div-3: 多样性加权
        print("\n[策略 3/3] 多样性加权 (0.6性能 + 0.4多样性)")
        model_aucs_4 = [model_results_auc[n]["auc"] for n in diverse_model_names]
        prob_div_3, _, _, _ = diversity_weighted_ensemble(
            diverse_preds, model_aucs_4, diverse_model_names, diversity_weight=0.4
        )
        score_div_3, _, _ = compute_metrics(y_true, prob_div_3, thresholds=None)
        print(f"  ✓ 完成! AUC: {score_div_3:.4f}")
        auc_results.append(("4架构多样性加权", score_div_3, prob_div_3))
        
    else:
        print(f"[跳过] 多架构模型不足（需要 2 个，当前 {len(diverse_model_names)} 个）")
    
    # ========== 5. 高级优化集成（概率校准 + Stacking + 联合阈值）==========
    print("\n" + "=" * 70)
    print("【步骤 5/5】高级优化集成")
    print("=" * 70)
    print(f"  • 概率校准: {'开启 (' + CALIBRATION_METHOD + ')' if USE_PROBABILITY_CALIBRATION else '关闭'}")
    print(f"  • 联合阈值: {'开启' if USE_JOINT_THRESHOLD else '关闭'}")
    
    # F1 优化的高级集成
    if len(f1_model_names) >= 2:
        print("\n[F1 优化流程]")
        opt_f1_prob, opt_f1_thr, opt_f1_metrics, cal_f1_result = optimized_ensemble_pipeline(
            f1_preds, y_true, f1_model_names, target_metric="f1"
        )
        f1_results.append(("高级优化集成", opt_f1_metrics["f1"], opt_f1_prob, opt_f1_thr))
        # 如果有校准后简单平均的结果，也加入
        if cal_f1_result is not None:
            cal_prob, cal_thr, cal_auc, cal_f1 = cal_f1_result
            f1_results.append(("校准后简单平均", cal_f1, cal_prob, cal_thr))
    
    # AUC 优化的高级集成（使用多架构模型）
    if len(diverse_model_names) >= 2:
        print("\n[AUC 优化流程]")
        opt_auc_prob, opt_auc_thr, opt_auc_metrics, cal_auc_result = optimized_ensemble_pipeline(
            diverse_preds, y_true, diverse_model_names, target_metric="auc"
        )
        auc_results.append(("高级优化集成", opt_auc_metrics["auc"], opt_auc_prob))
        # 如果有校准后简单平均的结果，也加入
        if cal_auc_result is not None:
            cal_prob, cal_thr, cal_auc, cal_f1 = cal_auc_result
            auc_results.append(("校准后简单平均", cal_auc, cal_prob))
    
    # ========== 6. 汇总结果 ==========
    print("\n")
    print("=" * 70)
    print("                        集成结果汇总")
    print("=" * 70)
    
    # 首先展示单模型 baseline
    print("\n【单模型 Baseline】(用于对比集成效果)")
    print("-" * 60)
    print(f"{'模型名称':<35} | {'版本':>5} | {'F1':>8} | {'AUC':>8}")
    print("-" * 60)
    
    # 展示 F1 集成使用的单模型
    for name in f1_model_names:
        if name in model_results_f1:
            info = model_results_f1[name]
            print(f"  {name:<33} | {info['source_version']:>5} | {info['f1_macro']:>8.4f} | {info['auc']:>8.4f}")
    print("-" * 60)
    
    # F1 结果 - 分组显示
    if f1_results:
        # 计算单模型最佳 F1 作为 baseline
        single_best_f1 = max(model_results_f1[n]["f1_macro"] for n in f1_model_names if n in model_results_f1)
        
        # 分离 2模型和 3模型结果
        results_2m = [(n, s, p, t) for n, s, p, t in f1_results if n.startswith("2模型")]
        results_3m = [(n, s, p, t) for n, s, p, t in f1_results if n.startswith("3模型")]
        results_other = [(n, s, p, t) for n, s, p, t in f1_results if not n.startswith("2模型") and not n.startswith("3模型")]
        
        print(f"\n【F1 集成结果】(单模型最佳 F1 = {single_best_f1:.4f})")
        print("-" * 60)
        print(f"{'策略':<30} | {'Macro-F1':>10} | {'vs 单模型':>12}")
        print("-" * 60)
        
        def print_result_row(name, score):
            diff = score - single_best_f1
            diff_str = f"{diff:+.4f}" if diff != 0 else "  持平"
            marker = " ★" if score > single_best_f1 else " ✗" if score < single_best_f1 * 0.95 else ""
            print(f"  {name:<28} | {score:>10.4f} | {diff_str:>10}{marker}")
        
        if results_2m:
            print("【2模型基准组】")
            for name, score, _, _ in sorted(results_2m, key=lambda x: -x[1]):
                print_result_row(name, score)
            best_2m = max(results_2m, key=lambda x: x[1])
        
        if results_3m:
            print("【3模型对照组】")
            for name, score, _, _ in sorted(results_3m, key=lambda x: -x[1]):
                print_result_row(name, score)
            best_3m = max(results_3m, key=lambda x: x[1])
        
        if results_other:
            print("【高级优化】")
            for name, score, _, _ in sorted(results_other, key=lambda x: -x[1]):
                print_result_row(name, score)
        
        print("-" * 60)
        
        # 总体最佳
        best_f1 = max(f1_results, key=lambda x: x[1])
        improvement = best_f1[1] - single_best_f1
        print(f"\n>>> 最佳 F1 集成: {best_f1[0]}")
        print(f"    Macro-F1 = {best_f1[1]:.4f} (相比单模型 {improvement:+.4f})")
    
    # AUC 结果
    if auc_results:
        # 计算单模型最佳 AUC 作为 baseline
        single_best_auc = max(model_results_auc[n]["auc"] for n in diverse_model_names if n in model_results_auc)
        
        print(f"\n【AUC 集成结果】(单模型最佳 AUC = {single_best_auc:.4f})")
        print("-" * 60)
        print(f"{'策略':<30} | {'AUC':>10} | {'vs 单模型':>12}")
        print("-" * 60)
        for name, score, _ in sorted(auc_results, key=lambda x: -x[1]):
            diff = score - single_best_auc
            diff_str = f"{diff:+.4f}" if diff != 0 else "  持平"
            marker = " ★" if score > single_best_auc else ""
            print(f"  {name:<28} | {score:>10.4f} | {diff_str:>10}{marker}")
        
        print("-" * 60)
        
        # 找到最佳 AUC
        best_auc = max(auc_results, key=lambda x: x[1])
        improvement = best_auc[1] - single_best_auc
        print(f"\n>>> 最佳 AUC 集成: {best_auc[0]}")
        print(f"    AUC = {best_auc[1]:.4f} (相比单模型 {improvement:+.4f})")
    
    # ========== 7. 保存结果 ==========
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
        # 判断最佳策略使用的模型组
        # "3模型" 开头的策略使用 auc_model_names，其他（4架构、高级优化等）使用 diverse_model_names
        if best_auc_name.startswith("3模型"):
            best_model_names = auc_model_names
        else:
            best_model_names = diverse_model_names
        np.savez(
            save_path_auc,
            y_true=y_true,
            y_pred_prob=best_auc_prob,
            model_names=np.array(best_model_names),
            ensemble_strategy=best_auc_name,
            best_auc=best_auc_score,
        )
        print(f">>> AUC 最佳集成已保存至: {save_path_auc}")
    
    print("\n>>> 集成评估完成！")
    
    close_logger()


if __name__ == "__main__":
    main()

