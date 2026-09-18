# test_evaluation.py
"""
测试集评估脚本

使用训练时保存的数据集划分信息，在测试集上评估模型性能。
支持单模型评估和集成模型评估。
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

import torch
from torch.utils.data import DataLoader

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cxr_config import (
    IMAGES_DIR, LABEL_CSV, SAVE_DIR, DEVICE, CLASS_NAMES, NUM_CLASSES
)
from src.dataset import ChestXrayDataset, get_transforms
from src.models_timm import TimmMultiLabel
from src.metrics_utils import compute_metrics, search_best_thresholds


def load_dataset_split():
    """加载数据集划分信息"""
    split_path = os.path.join(SAVE_DIR, "dataset_split.npz")
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"未找到数据集划分文件: {split_path}\n"
            "请先运行 train_timm_models.py 进行训练，它会自动保存划分信息。"
        )
    
    data = np.load(split_path, allow_pickle=True)
    return {
        "train_files": data["train_files"].tolist(),
        "val_files": data["val_files"].tolist(),
        "test_files": data["test_files"].tolist(),
    }


def find_model_checkpoints():
    """查找所有已保存的模型权重"""
    patterns = [
        os.path.join(SAVE_DIR, "*_best_auc.pt"),
        os.path.join(SAVE_DIR, "*_best_f1.pt"),
    ]
    
    checkpoints = []
    for pattern in patterns:
        checkpoints.extend(glob(pattern))
    
    return sorted(set(checkpoints))


def load_model(checkpoint_path):
    """加载模型"""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # 从检查点中获取模型配置
    model_name = checkpoint.get("model_name", None)
    input_size = checkpoint.get("input_size", 224)
    
    if model_name is None:
        # 尝试从文件名推断
        basename = os.path.basename(checkpoint_path)
        # 移除 _best_auc.pt 或 _best_f1.pt 后缀
        model_name = basename.replace("_best_auc.pt", "").replace("_best_f1.pt", "")
    
    # 创建模型
    model = TimmMultiLabel(
        model_name=model_name,
        num_classes=NUM_CLASSES,
        pretrained=False,
        input_size=input_size,
    )
    
    # 加载权重
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()
    
    thresholds = checkpoint.get("thresholds", None)
    
    return model, thresholds, input_size


def evaluate_on_test(model, test_loader, thresholds=None):
    """在测试集上评估模型"""
    all_labels = []
    all_probs = []
    all_names = []
    
    with torch.no_grad():
        for imgs, labels, names in tqdm(test_loader, desc="Testing"):
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            
            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())
            all_names.extend(names)
    
    y_true = np.concatenate(all_labels, axis=0)
    y_pred_prob = np.concatenate(all_probs, axis=0)
    
    # 如果没有提供阈值，使用默认 0.5
    if thresholds is None:
        thresholds = [0.5] * NUM_CLASSES
    
    auc_macro, f1_macro = compute_metrics(y_true, y_pred_prob, thresholds)
    
    return {
        "auc": auc_macro,
        "f1": f1_macro,
        "y_true": y_true,
        "y_pred_prob": y_pred_prob,
        "file_names": all_names,
        "thresholds": thresholds,
    }


def print_per_class_metrics(y_true, y_pred_prob, thresholds):
    """打印每个类别的指标"""
    from sklearn.metrics import roc_auc_score, f1_score
    
    print("\n" + "=" * 70)
    print("各类别测试集指标:")
    print("=" * 70)
    print(f"{'类别':<20} {'AUC':<10} {'F1':<10} {'样本数':<10}")
    print("-" * 50)
    
    for i, cls_name in enumerate(CLASS_NAMES):
        try:
            auc = roc_auc_score(y_true[:, i], y_pred_prob[:, i])
        except:
            auc = float('nan')
        
        y_pred = (y_pred_prob[:, i] >= thresholds[i]).astype(int)
        f1 = f1_score(y_true[:, i], y_pred, zero_division=0)
        n_positive = int(y_true[:, i].sum())
        
        print(f"{cls_name:<20} {auc:<10.4f} {f1:<10.4f} {n_positive:<10}")
    
    print("-" * 50)


def main():
    print("=" * 70)
    print("测试集评估")
    print("=" * 70)
    
    # 加载数据集划分
    try:
        split_info = load_dataset_split()
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return
    
    test_files = split_info["test_files"]
    val_files = split_info["val_files"]
    
    print(f"测试集样本数: {len(test_files)}")
    
    # 加载标签数据
    df_labels = pd.read_csv(LABEL_CSV)
    
    # 查找可用的模型
    checkpoints = find_model_checkpoints()
    
    if not checkpoints:
        print("未找到任何模型权重文件。请先运行训练。")
        return
    
    print(f"\n找到 {len(checkpoints)} 个模型权重文件")
    
    # 评估每个模型
    results = []
    
    for ckpt_path in checkpoints:
        ckpt_name = os.path.basename(ckpt_path)
        print(f"\n{'=' * 70}")
        print(f"评估模型: {ckpt_name}")
        print("=" * 70)
        
        try:
            model, thresholds, input_size = load_model(ckpt_path)
        except Exception as e:
            print(f"加载模型失败: {e}")
            continue
        
        # 创建测试数据加载器
        test_transform = get_transforms(train=False)
        test_dataset = ChestXrayDataset(
            file_list=test_files,
            df_labels=df_labels,
            transform=test_transform,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        
        # 评估
        result = evaluate_on_test(model, test_loader, thresholds)
        result["model_name"] = ckpt_name
        results.append(result)
        
        print(f"Test AUC: {result['auc']:.4f}")
        print(f"Test F1:  {result['f1']:.4f}")
        
        # 打印各类别指标
        print_per_class_metrics(result["y_true"], result["y_pred_prob"], result["thresholds"])
        
        # 保存预测结果
        pred_save_path = os.path.join(SAVE_DIR, f"{ckpt_name.replace('.pt', '')}_test_preds.npz")
        np.savez(
            pred_save_path,
            file_names=np.array(result["file_names"]),
            y_true=result["y_true"],
            y_pred_prob=result["y_pred_prob"],
        )
        print(f"预测结果已保存到: {pred_save_path}")
        
        # 释放显存
        del model
        torch.cuda.empty_cache()
    
    # 汇总结果
    if results:
        print("\n" + "=" * 70)
        print("测试集评估汇总")
        print("=" * 70)
        print(f"{'模型':<60} {'AUC':<10} {'F1':<10}")
        print("-" * 80)
        
        for r in sorted(results, key=lambda x: x['auc'], reverse=True):
            print(f"{r['model_name']:<60} {r['auc']:<10.4f} {r['f1']:<10.4f}")
        
        # 找出最佳模型
        best_auc = max(results, key=lambda x: x['auc'])
        best_f1 = max(results, key=lambda x: x['f1'])
        
        print("\n最佳模型:")
        print(f"  AUC 最佳: {best_auc['model_name']} (AUC={best_auc['auc']:.4f})")
        print(f"  F1 最佳:  {best_f1['model_name']} (F1={best_f1['f1']:.4f})")


if __name__ == "__main__":
    main()
