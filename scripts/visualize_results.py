# visualize_results.py
"""
可视化脚本：生成 ROC 曲线、AUC 柱状图、混淆矩阵统计等可视化内容
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, f1_score
)

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cxr_config import CLASS_NAMES, SAVE_DIR
from src.metrics_utils import search_best_thresholds

# 输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配色方案
COLORS = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
    '#008080', '#e6beff', '#9a6324', '#800000'
]


def load_ensemble_predictions(version="auc"):
    """
    加载集成模型的预测结果
    
    Args:
        version: "auc" 或 "f1"，选择加载哪个版本的集成结果
    
    Returns:
        y_true, y_pred_prob, metadata
    """
    if version == "auc":
        path = os.path.join(SAVE_DIR, "val_preds_timm_ensemble_auc.npz")
    else:
        path = os.path.join(SAVE_DIR, "val_preds_timm_ensemble_f1.npz")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到集成结果文件: {path}")
    
    data = np.load(path, allow_pickle=True)
    y_true = data['y_true']
    y_pred_prob = data['y_pred_prob']
    
    metadata = {
        'strategy': str(data['ensemble_strategy']) if 'ensemble_strategy' in data else 'unknown',
        'model_names': list(data['model_names']) if 'model_names' in data else [],
    }
    
    if 'best_auc' in data:
        metadata['best_auc'] = float(data['best_auc'])
    
    print(f"已加载集成结果: {path}")
    print(f"  策略: {metadata['strategy']}")
    print(f"  数据形状: y_true={y_true.shape}, y_pred_prob={y_pred_prob.shape}")
    
    return y_true, y_pred_prob, metadata


def compute_per_class_metrics(y_true, y_pred_prob, thresholds=None):
    """
    计算每个类别的详细指标
    
    Returns:
        dict: 包含每个类别的 AUC, F1, Precision, Recall 等
    """
    if thresholds is None:
        thresholds = search_best_thresholds(y_true, y_pred_prob)
    
    y_pred_bin = (y_pred_prob >= thresholds[None, :]).astype(int)
    
    metrics = {}
    for i, name in enumerate(CLASS_NAMES):
        # AUC
        try:
            auc_score = roc_auc_score(y_true[:, i], y_pred_prob[:, i])
        except ValueError:
            auc_score = np.nan
        
        # 混淆矩阵
        cm = confusion_matrix(y_true[:, i], y_pred_bin[:, i])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # F1, Precision, Recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # 样本数
        n_positive = int(y_true[:, i].sum())
        n_negative = len(y_true) - n_positive
        
        metrics[name] = {
            'auc': auc_score,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'threshold': thresholds[i],
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'n_positive': n_positive,
            'n_negative': n_negative,
        }
    
    return metrics, thresholds


def plot_roc_curves_all(y_true, y_pred_prob, save_path=None):
    """
    绘制所有类别的 ROC 曲线（单图）
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    aucs = []
    for i, name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_prob[:, i])
        auc_score = auc(fpr, tpr)
        aucs.append(auc_score)
        ax.plot(fpr, tpr, color=COLORS[i], lw=2, 
                label=f'{name} (AUC={auc_score:.3f})')
    
    # 对角线
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random (AUC=0.500)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curves for 14 Chest X-ray Diseases\n(Macro-AUC = {np.mean(aucs):.4f})', fontsize=14)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {save_path}")
    plt.close()
    
    return aucs


def plot_roc_curves_grid(y_true, y_pred_prob, save_path=None):
    """
    绘制每个类别单独的 ROC 曲线（网格布局）
    """
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    axes = axes.flatten()
    
    aucs = []
    for i, name in enumerate(CLASS_NAMES):
        ax = axes[i]
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_prob[:, i])
        auc_score = auc(fpr, tpr)
        aucs.append(auc_score)
        
        ax.plot(fpr, tpr, color=COLORS[i], lw=2)
        ax.fill_between(fpr, tpr, alpha=0.3, color=COLORS[i])
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_title(f'{name}\nAUC = {auc_score:.3f}', fontsize=10)
        ax.set_xlabel('FPR', fontsize=9)
        ax.set_ylabel('TPR', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for j in range(len(CLASS_NAMES), len(axes)):
        axes[j].axis('off')
    
    # 在最后一个位置添加汇总信息
    ax_summary = axes[-1]
    ax_summary.axis('off')
    summary_text = f"Macro-AUC: {np.mean(aucs):.4f}\n\n"
    summary_text += "Top 3:\n"
    sorted_idx = np.argsort(aucs)[::-1]
    for k in range(3):
        idx = sorted_idx[k]
        summary_text += f"  {CLASS_NAMES[idx]}: {aucs[idx]:.3f}\n"
    summary_text += "\nBottom 3:\n"
    for k in range(3):
        idx = sorted_idx[-(k+1)]
        summary_text += f"  {CLASS_NAMES[idx]}: {aucs[idx]:.3f}\n"
    ax_summary.text(0.1, 0.5, summary_text, fontsize=11, 
                    verticalalignment='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.suptitle('ROC Curves by Disease Category', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {save_path}")
    plt.close()
    
    return aucs


def plot_auc_bar_chart(metrics, save_path=None):
    """
    绘制每个类别 AUC 的柱状图
    """
    # 按 AUC 降序排列
    sorted_items = sorted(metrics.items(), key=lambda x: x[1]['auc'], reverse=True)
    names = [item[0] for item in sorted_items]
    aucs = [item[1]['auc'] for item in sorted_items]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    bars = ax.bar(range(len(names)), aucs, color=COLORS[:len(names)], edgecolor='black', linewidth=0.5)
    
    # 添加数值标签
    for bar, auc_val in zip(bars, aucs):
        height = bar.get_height()
        ax.annotate(f'{auc_val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('AUC Score', fontsize=12)
    ax.set_title(f'Per-Class AUC Scores (Macro-AUC = {np.mean(aucs):.4f})', fontsize=14)
    ax.set_ylim(0.5, 1.0)
    ax.axhline(y=np.mean(aucs), color='red', linestyle='--', linewidth=2, label=f'Macro-AUC = {np.mean(aucs):.4f}')
    ax.axhline(y=0.8, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label='AUC = 0.8')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {save_path}")
    plt.close()


def plot_f1_bar_chart(metrics, save_path=None):
    """
    绘制每个类别 F1 Score 的柱状图
    """
    sorted_items = sorted(metrics.items(), key=lambda x: x[1]['f1'], reverse=True)
    names = [item[0] for item in sorted_items]
    f1s = [item[1]['f1'] for item in sorted_items]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    bars = ax.bar(range(len(names)), f1s, color=COLORS[:len(names)], edgecolor='black', linewidth=0.5)
    
    for bar, f1_val in zip(bars, f1s):
        height = bar.get_height()
        ax.annotate(f'{f1_val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title(f'Per-Class F1 Scores (Macro-F1 = {np.mean(f1s):.4f})', fontsize=14)
    ax.set_ylim(0, 0.8)
    ax.axhline(y=np.mean(f1s), color='red', linestyle='--', linewidth=2, label=f'Macro-F1 = {np.mean(f1s):.4f}')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {save_path}")
    plt.close()


def plot_confusion_matrix_summary(metrics, save_path=None):
    """
    绘制混淆矩阵统计汇总（堆叠柱状图）
    """
    names = CLASS_NAMES
    tp = [metrics[n]['tp'] for n in names]
    fp = [metrics[n]['fp'] for n in names]
    fn = [metrics[n]['fn'] for n in names]
    tn = [metrics[n]['tn'] for n in names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：TP, FP, FN 堆叠柱状图
    x = np.arange(len(names))
    width = 0.6
    
    ax1.bar(x, tp, width, label='True Positive (TP)', color='#2ecc71')
    ax1.bar(x, fp, width, bottom=tp, label='False Positive (FP)', color='#e74c3c')
    ax1.bar(x, fn, width, bottom=np.array(tp)+np.array(fp), label='False Negative (FN)', color='#f39c12')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Confusion Matrix Breakdown by Class\n(TP, FP, FN)', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    
    # 右图：Precision 和 Recall 对比
    precision = [metrics[n]['precision'] for n in names]
    recall = [metrics[n]['recall'] for n in names]
    
    width = 0.35
    ax2.bar(x - width/2, precision, width, label='Precision', color='#3498db')
    ax2.bar(x + width/2, recall, width, label='Recall', color='#9b59b6')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Score', fontsize=11)
    ax2.set_title('Precision vs Recall by Class', fontsize=12)
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {save_path}")
    plt.close()


def plot_pr_curves(y_true, y_pred_prob, save_path=None):
    """
    绘制 Precision-Recall 曲线
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    aps = []
    for i, name in enumerate(CLASS_NAMES):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred_prob[:, i])
        ap = average_precision_score(y_true[:, i], y_pred_prob[:, i])
        aps.append(ap)
        ax.plot(recall, precision, color=COLORS[i], lw=2, 
                label=f'{name} (AP={ap:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision-Recall Curves\n(Mean AP = {np.mean(aps):.4f})', fontsize=14)
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {save_path}")
    plt.close()
    
    return aps


def plot_class_distribution(y_true, save_path=None):
    """
    绘制类别样本分布图
    """
    counts = y_true.sum(axis=0).astype(int)
    sorted_idx = np.argsort(counts)[::-1]
    sorted_names = [CLASS_NAMES[i] for i in sorted_idx]
    sorted_counts = counts[sorted_idx]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    bars = ax.bar(range(len(sorted_names)), sorted_counts, 
                  color=COLORS[:len(sorted_names)], edgecolor='black', linewidth=0.5)
    
    for bar, count in zip(bars, sorted_counts):
        height = bar.get_height()
        ax.annotate(f'{count}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Number of Positive Samples', fontsize=12)
    ax.set_title(f'Class Distribution in Validation Set\n(Total samples: {len(y_true)}, Total positive labels: {counts.sum()})', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {save_path}")
    plt.close()


def plot_metrics_heatmap(metrics, save_path=None):
    """
    绘制各类别指标热力图
    """
    metric_names = ['AUC', 'F1', 'Precision', 'Recall']
    data = np.zeros((len(CLASS_NAMES), len(metric_names)))
    
    for i, name in enumerate(CLASS_NAMES):
        data[i, 0] = metrics[name]['auc']
        data[i, 1] = metrics[name]['f1']
        data[i, 2] = metrics[name]['precision']
        data[i, 3] = metrics[name]['recall']
    
    fig, ax = plt.subplots(figsize=(10, 12))
    
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_yticklabels(CLASS_NAMES, fontsize=10)
    
    # 添加数值标注
    for i in range(len(CLASS_NAMES)):
        for j in range(len(metric_names)):
            text = ax.text(j, i, f'{data[i, j]:.3f}',
                          ha='center', va='center', color='black', fontsize=9)
    
    ax.set_title('Performance Metrics Heatmap by Disease', fontsize=14)
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {save_path}")
    plt.close()


def generate_summary_table(metrics):
    """
    生成指标汇总表格并打印
    """
    print("\n" + "="*80)
    print("各类别详细指标汇总")
    print("="*80)
    print(f"{'疾病类别':<20} {'AUC':>8} {'F1':>8} {'Prec':>8} {'Recall':>8} {'阈值':>8} {'正样本':>8}")
    print("-"*80)
    
    # 按 AUC 排序
    sorted_items = sorted(metrics.items(), key=lambda x: x[1]['auc'], reverse=True)
    
    for name, m in sorted_items:
        print(f"{name:<20} {m['auc']:>8.4f} {m['f1']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['threshold']:>8.2f} {m['n_positive']:>8d}")
    
    print("-"*80)
    
    # 计算平均值
    aucs = [m['auc'] for m in metrics.values()]
    f1s = [m['f1'] for m in metrics.values()]
    precs = [m['precision'] for m in metrics.values()]
    recalls = [m['recall'] for m in metrics.values()]
    
    print(f"{'Macro Average':<20} {np.mean(aucs):>8.4f} {np.mean(f1s):>8.4f} {np.mean(precs):>8.4f} {np.mean(recalls):>8.4f}")
    print("="*80)


def main():
    """
    主函数：生成所有可视化内容
    """
    print("="*60)
    print("ChestX-ray14 多标签分类结果可视化")
    print("="*60)
    print(f"输出目录: {OUTPUT_DIR}")
    print()
    
    # 1. 加载数据
    print("\n[1/8] 加载集成模型预测结果...")
    try:
        y_true, y_pred_prob, metadata = load_ensemble_predictions(version="auc")
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请先运行 ensemble_timm.py 生成集成结果")
        return
    
    # 2. 计算指标
    print("\n[2/8] 计算各类别指标...")
    metrics, thresholds = compute_per_class_metrics(y_true, y_pred_prob)
    generate_summary_table(metrics)
    
    # 3. 绘制 ROC 曲线（所有类别）
    print("\n[3/8] 绘制 ROC 曲线（合并图）...")
    plot_roc_curves_all(y_true, y_pred_prob, 
                        save_path=os.path.join(OUTPUT_DIR, "roc_curves_all.png"))
    
    # 4. 绘制 ROC 曲线（网格布局）
    print("\n[4/8] 绘制 ROC 曲线（网格图）...")
    plot_roc_curves_grid(y_true, y_pred_prob,
                         save_path=os.path.join(OUTPUT_DIR, "roc_curves_grid.png"))
    
    # 5. 绘制 AUC 柱状图
    print("\n[5/8] 绘制 AUC 柱状图...")
    plot_auc_bar_chart(metrics,
                       save_path=os.path.join(OUTPUT_DIR, "auc_bar_chart.png"))
    
    # 6. 绘制 F1 柱状图
    print("\n[6/8] 绘制 F1 柱状图...")
    plot_f1_bar_chart(metrics,
                      save_path=os.path.join(OUTPUT_DIR, "f1_bar_chart.png"))
    
    # 7. 绘制混淆矩阵统计
    print("\n[7/8] 绘制混淆矩阵统计...")
    plot_confusion_matrix_summary(metrics,
                                  save_path=os.path.join(OUTPUT_DIR, "confusion_matrix_summary.png"))
    
    # 8. 绘制 PR 曲线
    print("\n[8/8] 绘制 Precision-Recall 曲线...")
    plot_pr_curves(y_true, y_pred_prob,
                   save_path=os.path.join(OUTPUT_DIR, "pr_curves.png"))
    
    # 额外：类别分布图
    print("\n[额外] 绘制类别分布图...")
    plot_class_distribution(y_true,
                           save_path=os.path.join(OUTPUT_DIR, "class_distribution.png"))
    
    # 额外：指标热力图
    print("\n[额外] 绘制指标热力图...")
    plot_metrics_heatmap(metrics,
                         save_path=os.path.join(OUTPUT_DIR, "metrics_heatmap.png"))
    
    print("\n" + "="*60)
    print("可视化完成！")
    print(f"所有图片已保存至: {OUTPUT_DIR}")
    print("="*60)
    
    # 列出生成的文件
    print("\n生成的文件:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.png'):
            fpath = os.path.join(OUTPUT_DIR, f)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  - {f} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()

