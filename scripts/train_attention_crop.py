# train_attention_crop.py
"""
Attention-Guided Crop 训练便捷入口

这个脚本只是一个便捷入口，它会：
1. 设置 USE_ATTENTION_CROP = True
2. 调用 train.py 中的 train_ensemble() 函数

等效于在 train.py 中手动设置 USE_ATTENTION_CROP = True 后运行。

使用方式：
    python train_attention_crop.py

或者直接修改 train.py 中的配置：
    USE_ATTENTION_CROP = True
    然后运行 python train.py
"""

from scripts import train

if __name__ == "__main__":
    # 启用 Attention-Guided Crop 模式
    train.USE_ATTENTION_CROP = True
    
    # 可选：调整其他相关配置
    # train.AG_HIGH_RES = 512
    # train.AG_NUM_CROPS = 2
    # train.AG_BATCH_SIZE = 12
    
    print("=" * 70)
    print("Attention-Guided Crop 模式已启用")
    print("=" * 70)
    print(f"  高分辨率尺寸: {train.AG_HIGH_RES}")
    print(f"  裁剪区域数量: {train.AG_NUM_CROPS}")
    print(f"  裁剪尺寸: {train.AG_CROP_SIZE}")
    print(f"  融合方式: {train.AG_FUSION_TYPE}")
    print(f"  Batch Size: {train.AG_BATCH_SIZE}")
    print("=" * 70)
    
    # 运行训练
    train.train_ensemble()
