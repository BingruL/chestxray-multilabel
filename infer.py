# infer.py
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from config import (
    NIH_DATA_ROOT, TEST_LIST, DEVICE, BATCH_SIZE,
)
from dataset import ChestXrayDataset, get_transforms
from models import DenseNetMultiLabel
from metrics_utils import compute_metrics


def load_file_list(list_path: str):
    with open(list_path, "r") as f:
        files = [line.strip() for line in f if line.strip()]
    return files


def main():
    model_ckpt = os.path.join("saved_models", "densenet121_best.pth")
    assert os.path.exists(model_ckpt), "请先运行 train.py 训练并保存模型！"

    checkpoint = torch.load(model_ckpt, map_location=DEVICE)
    thresholds = checkpoint.get("thresholds", None)

    model = DenseNetMultiLabel(pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()

    # 标签表 & test 文件列表
    df_labels = pd.read_csv(os.path.join(NIH_DATA_ROOT, "Data_Entry_2017.csv"))
    test_files = load_file_list(TEST_LIST)

    test_dataset = ChestXrayDataset(
        file_list=test_files,
        df_labels=df_labels,
        transform=get_transforms(train=False),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    all_labels = []
    all_probs = []
    all_names = []

    with torch.no_grad():
        for imgs, labels, names in tqdm(test_loader, desc="Inference on test set"):
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs = torch.sigmoid(logits)

            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())
            all_names.extend(names)

    y_true = np.concatenate(all_labels, axis=0)
    y_pred_prob = np.concatenate(all_probs, axis=0)

    auc_macro, f1_macro = compute_metrics(y_true, y_pred_prob, thresholds)
    print(f"\nTest AUC (using best thresholds from val): {auc_macro:.4f}")
    print(f"Test F1  (using best thresholds from val): {f1_macro:.4f}")

    # 把预测结果保存下来（方便老师或你自己检查）
    np.savez(
        "test_predictions.npz",
        file_names=np.array(all_names),
        y_true=y_true,
        y_pred_prob=y_pred_prob,
    )
    print("Saved test predictions to test_predictions.npz")


if __name__ == "__main__":
    main()
"""