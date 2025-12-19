# dataset.py

import os
from typing import List, Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from cxr_config import CLASS_NAMES, NUM_CLASSES, IMAGES_DIR, LABEL_CSV


def get_transforms(train: bool = True, use_jitter: bool = False, sharpness: float = 0.0):
    """图像预处理与数据增强（用于标准 torchvision 模型）"""
    color_blocks = []
    if use_jitter and train:
        color_blocks.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02))
        color_blocks.append(T.RandomAutocontrast())
    if sharpness > 0 and train:
        color_blocks.append(T.RandomAdjustSharpness(sharpness_factor=1.0 + sharpness, p=0.5))

    if train:
        return T.Compose([
            T.Resize(256),
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            *color_blocks,
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.25, 0.25, 0.25]),
        ])
    else:
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.25, 0.25, 0.25]),
        ])


class XRVNormalize:
    """
    torchxrayvision 专用的归一化变换。
    将像素值从 [0, 1] 映射到 [-1024, 1024] 范围。
    
    torchxrayvision 模型期望的输入格式：
    - 像素值范围: [-1024, 1024]
    - 可以是单通道或三通道（模型内部会转换）
    """
    def __call__(self, img):
        """
        Args:
            img: torch.Tensor, 像素值在 [0, 1] 范围
        Returns:
            torch.Tensor, 像素值在 [-1024, 1024] 范围
        """
        # [0, 1] -> [0, 2048] -> [-1024, 1024]
        return img * 2048 - 1024


def get_xrv_transforms(train: bool = True, use_jitter: bool = False, sharpness: float = 0.0):
    """
    torchxrayvision 专用的图像预处理与数据增强。
    
    主要区别：
    1. 使用 XRVNormalize 替代标准 Normalize
    2. 像素值范围映射到 [-1024, 1024]
    """
    color_blocks = []
    if use_jitter and train:
        # 对于 X 光图像，色彩增强要保守一些
        color_blocks.append(T.ColorJitter(brightness=0.1, contrast=0.1))
        color_blocks.append(T.RandomAutocontrast())
    if sharpness > 0 and train:
        color_blocks.append(T.RandomAdjustSharpness(sharpness_factor=1.0 + sharpness, p=0.5))

    if train:
        return T.Compose([
            T.Resize(256),
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            *color_blocks,
            T.ToTensor(),
            XRVNormalize(),
        ])
    else:
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            XRVNormalize(),
        ])


def get_xrv_transforms_with_hist_eq(train: bool = True):
    """
    带直方图均衡化的 torchxrayvision 预处理。
    torchxrayvision 官方推荐对 X 光图像做直方图均衡化。
    """
    try:
        import torchxrayvision as xrv  # type: ignore
        has_xrv = True
    except ImportError:
        has_xrv = False
    
    base_transforms = []
    
    if train:
        base_transforms = [
            T.Resize(256),
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
        ]
    else:
        base_transforms = [
            T.Resize(256),
            T.CenterCrop(224),
        ]
    
    # ToTensor + XRV 归一化
    base_transforms.extend([
        T.ToTensor(),
        XRVNormalize(),
    ])
    
    return T.Compose(base_transforms)


class ChestXrayDataset(Dataset):
    def __init__(
        self,
        file_list: List[str],
        df_labels: Optional[pd.DataFrame] = None,
        images_dir: str = IMAGES_DIR,
        transform=None,
    ):
        """
        file_list: 图像文件名列表（例如：'00000001_000.png'）
        df_labels: filtered_labels.csv 读出来的 DataFrame
        """
        self.file_list = file_list
        self.images_dir = images_dir
        self.transform = transform if transform is not None else get_transforms(train=True)

        if df_labels is None:
            df_labels = pd.read_csv(LABEL_CSV)
            df_labels = df_labels[["Image Index", "Finding Labels"]]
        else:
            # train.py 已经过滤过，这里只取需要的两列
            df_labels = df_labels[["Image Index", "Finding Labels"]]

        self.df_labels = df_labels.set_index("Image Index")


    def __len__(self):
        return len(self.file_list)

    def _get_label_vector(self, findings: str) -> torch.Tensor:
        """
        把 'Atelectasis|Effusion' 这种字符串转成 14 维 multi-hot 向量
        """
        label_vec = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        findings = str(findings).strip()
        if findings == "No Finding":
            return label_vec

        for disease in findings.split("|"):
            disease = disease.strip()
            if disease in CLASS_NAMES:
                idx = CLASS_NAMES.index(disease)
                label_vec[idx] = 1.0
        return label_vec

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.images_dir, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        findings = self.df_labels.loc[img_name]["Finding Labels"]
        label_vec = self._get_label_vector(findings)

        return img, label_vec, img_name


# ==================== 多分辨率数据集（用于 Attention-Guided Crop） ====================

def get_xrv_multi_res_transforms(
    train: bool = True,
    low_res: int = 224,
    high_res: int = 512,
    use_jitter: bool = False,
):
    """
    多分辨率预处理变换。
    返回两个 transform：一个用于低分辨率，一个用于高分辨率。
    
    注意：为了保证裁剪位置对应，训练时的随机增强需要在原图上做一次，
    然后分别 resize 到不同分辨率。
    """
    # 基础增强（在原图上做）
    if train:
        base_augment = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
        ])
        if use_jitter:
            base_augment = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomRotation(10),
                T.ColorJitter(brightness=0.1, contrast=0.1),
            ])
    else:
        base_augment = None
    
    # 低分辨率变换
    low_res_transform = T.Compose([
        T.Resize((low_res, low_res)),
        T.ToTensor(),
        XRVNormalize(),
    ])
    
    # 高分辨率变换
    high_res_transform = T.Compose([
        T.Resize((high_res, high_res)),
        T.ToTensor(),
        XRVNormalize(),
    ])
    
    return base_augment, low_res_transform, high_res_transform


class ChestXrayMultiResDataset(Dataset):
    """
    多分辨率胸部X光数据集
    
    同时返回低分辨率(224)和高分辨率(512)的图像，
    用于 Attention-Guided Crop 模型。
    
    关键点：
    - 训练时的随机增强（翻转、旋转）在原图上做一次
    - 然后分别 resize 到低/高分辨率
    - 这样保证两个分辨率的图像内容一致，裁剪位置能正确对应
    """
    
    def __init__(
        self,
        file_list: List[str],
        df_labels: Optional[pd.DataFrame] = None,
        images_dir: str = IMAGES_DIR,
        train: bool = True,
        low_res: int = 224,
        high_res: int = 512,
        use_jitter: bool = False,
    ):
        """
        Args:
            file_list: 图像文件名列表
            df_labels: 标签 DataFrame
            images_dir: 图像目录
            train: 是否为训练模式
            low_res: 低分辨率尺寸（全局分支输入）
            high_res: 高分辨率尺寸（用于裁剪局部区域）
            use_jitter: 是否使用颜色增强
        """
        self.file_list = file_list
        self.images_dir = images_dir
        self.train = train
        self.low_res = low_res
        self.high_res = high_res
        
        # 获取变换
        self.base_augment, self.low_res_transform, self.high_res_transform = \
            get_xrv_multi_res_transforms(train, low_res, high_res, use_jitter)
        
        # 标签处理
        if df_labels is None:
            df_labels = pd.read_csv(LABEL_CSV)
            df_labels = df_labels[["Image Index", "Finding Labels"]]
        else:
            df_labels = df_labels[["Image Index", "Finding Labels"]]
        
        self.df_labels = df_labels.set_index("Image Index")
    
    def __len__(self):
        return len(self.file_list)
    
    def _get_label_vector(self, findings: str) -> torch.Tensor:
        """把标签字符串转成 multi-hot 向量"""
        label_vec = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        findings = str(findings).strip()
        if findings == "No Finding":
            return label_vec
        
        for disease in findings.split("|"):
            disease = disease.strip()
            if disease in CLASS_NAMES:
                idx = CLASS_NAMES.index(disease)
                label_vec[idx] = 1.0
        return label_vec
    
    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        # 加载原图
        img = Image.open(img_path).convert("RGB")
        
        # 训练时先做随机增强（在原图上）
        if self.train and self.base_augment is not None:
            img = self.base_augment(img)
        
        # 分别生成低分辨率和高分辨率图像
        img_low = self.low_res_transform(img)   # (C, low_res, low_res)
        img_high = self.high_res_transform(img)  # (C, high_res, high_res)
        
        # 获取标签
        findings = self.df_labels.loc[img_name]["Finding Labels"]
        label_vec = self._get_label_vector(findings)
        
        return img_low, img_high, label_vec, img_name


class ChestXrayMultiResDatasetV2(Dataset):
    """
    多分辨率数据集 V2 版本
    
    改进：
    - 支持 RandomResizedCrop 风格的增强
    - 高分辨率图像上做 crop，低分辨率做对应的 resize
    """
    
    def __init__(
        self,
        file_list: List[str],
        df_labels: Optional[pd.DataFrame] = None,
        images_dir: str = IMAGES_DIR,
        train: bool = True,
        low_res: int = 224,
        high_res: int = 512,
        crop_scale: tuple = (0.85, 1.0),
    ):
        self.file_list = file_list
        self.images_dir = images_dir
        self.train = train
        self.low_res = low_res
        self.high_res = high_res
        self.crop_scale = crop_scale
        
        # 标签处理
        if df_labels is None:
            df_labels = pd.read_csv(LABEL_CSV)
            df_labels = df_labels[["Image Index", "Finding Labels"]]
        else:
            df_labels = df_labels[["Image Index", "Finding Labels"]]
        
        self.df_labels = df_labels.set_index("Image Index")
        
        # 基础变换
        self.to_tensor = T.ToTensor()
        self.normalize = XRVNormalize()
    
    def __len__(self):
        return len(self.file_list)
    
    def _get_label_vector(self, findings: str) -> torch.Tensor:
        label_vec = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        findings = str(findings).strip()
        if findings == "No Finding":
            return label_vec
        
        for disease in findings.split("|"):
            disease = disease.strip()
            if disease in CLASS_NAMES:
                idx = CLASS_NAMES.index(disease)
                label_vec[idx] = 1.0
        return label_vec
    
    def _get_random_crop_params(self, img_size, scale):
        """获取随机裁剪参数"""
        import random
        import math
        
        area = img_size[0] * img_size[1]
        target_area = random.uniform(scale[0], scale[1]) * area
        aspect_ratio = 1.0  # 保持正方形
        
        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))
        
        w = min(w, img_size[0])
        h = min(h, img_size[1])
        
        i = random.randint(0, img_size[1] - h)
        j = random.randint(0, img_size[0] - w)
        
        return i, j, h, w
    
    def __getitem__(self, idx):
        import torchvision.transforms.functional as TF
        import random
        
        img_name = self.file_list[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        img = Image.open(img_path).convert("RGB")
        
        if self.train:
            # 随机水平翻转
            if random.random() > 0.5:
                img = TF.hflip(img)
            
            # 随机旋转
            angle = random.uniform(-10, 10)
            img = TF.rotate(img, angle)
            
            # 随机裁剪（可选）
            if random.random() > 0.3:
                i, j, h, w = self._get_random_crop_params(img.size, self.crop_scale)
                img = TF.crop(img, i, j, h, w)
        
        # Resize 到高分辨率
        img_high = img.resize((self.high_res, self.high_res), Image.BILINEAR)
        
        # Resize 到低分辨率
        img_low = img.resize((self.low_res, self.low_res), Image.BILINEAR)
        
        # 转 tensor 并归一化
        img_low = self.normalize(self.to_tensor(img_low))
        img_high = self.normalize(self.to_tensor(img_high))
        
        # 标签
        findings = self.df_labels.loc[img_name]["Finding Labels"]
        label_vec = self._get_label_vector(findings)
        
        return img_low, img_high, label_vec, img_name
