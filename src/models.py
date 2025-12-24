# models.py
"""
支持多种 DenseNet121 预训练权重：
1. ImageNet 预训练（torchvision）
2. torchxrayvision 医学图像预训练（CheXpert, MIMIC-CXR, PadChest 等）
"""

import torch
import torch.nn as nn
from torchvision import models

# torchxrayvision 预训练权重选项
# 注意：不包含 all/nih/rsna，因为它们包含 NIH ChestX-ray14 数据，会导致数据泄漏
TORCHXRAYVISION_WEIGHTS = {
    "densenet121-res224-chex": "densenet121-res224-chex",         # CheXpert 数据集
    "densenet121-res224-pc": "densenet121-res224-pc",             # PadChest 数据集
    "densenet121-res224-mimic_nb": "densenet121-res224-mimic_nb", # MIMIC-CXR (NoBBox)
    "densenet121-res224-mimic_ch": "densenet121-res224-mimic_ch", # MIMIC-CXR (ChestOnly)
}


class DenseNetMultiLabel(nn.Module):
    """
    多标签 DenseNet121：
    - pretrained=True  时使用 ImageNet-1K 预训练
    - pretrained_ckpt_path 不为 None 时，再加载 CXR 专用预训练权重
      （例如 CheXpert / MIMIC-CXR 上训练好的 DenseNet121）
    """
    def __init__(
        self,
        num_classes: int = 14,
        pretrained: bool = True,
        pretrained_ckpt_path: str | None = None,
    ):
        super().__init__()

        # 1) 先加载 torchvision 自带 densenet121
        backbone = models.densenet121(pretrained=pretrained)
        in_features = backbone.classifier.in_features
        backbone.classifier = nn.Identity()  # 去掉原来的分类头

        self.backbone = backbone
        self.classifier = nn.Linear(in_features, num_classes)

        # 2) 如需加载 CXR 专用预训练权重
        if pretrained_ckpt_path is not None:
            print(f"===> Loading CXR pretrained weights from: {pretrained_ckpt_path}")
            self._load_cxr_pretrained(pretrained_ckpt_path)

    def _load_cxr_pretrained(self, ckpt_path: str):
        state = torch.load(ckpt_path, map_location="cpu")

        # 常见三种保存方式：
        # ① 直接是 state_dict
        # ② {'state_dict': ...}
        # ③ {'model_state': ...}
        if isinstance(state, dict):
            if "model_state" in state:
                state = state["model_state"]
            elif "state_dict" in state:
                state = state["state_dict"]

        # 有些 ckpt 可能带 'module.' 前缀，这里顺手去掉
        new_state = {}
        for k, v in state.items():
            if k.startswith("module."):
                k = k[len("module."):]
            new_state[k] = v
        state = new_state

        # 因为我们自己的结构是 backbone + classifier，
        # 而原 ckpt 可能只包含 backbone 的权重，所以 strict=False
        missing, unexpected = self.load_state_dict(state, strict=False)
        print("  >> CXR ckpt loaded. Missing keys:", missing)
        print("  >> Unexpected keys:", unexpected)

    def forward(self, x):
        feat = self.backbone(x)    # (B, in_features)
        logits = self.classifier(feat)  # (B, num_classes)
        return logits


class TorchXRayVisionDenseNet(nn.Module):
    """
    使用 torchxrayvision 库的 DenseNet121，支持多种医学图像预训练权重：
    - densenet121-res224-chex:     CheXpert 数据集预训练
    - densenet121-res224-pc:       PadChest 数据集预训练
    - densenet121-res224-mimic_nb: MIMIC-CXR (NoBBox) 预训练
    - densenet121-res224-mimic_ch: MIMIC-CXR (ChestOnly) 预训练
    
    注意：
    1. 不使用 all/nih/rsna 权重，因为它们包含 NIH ChestX-ray14 数据，会导致数据泄漏
    2. torchxrayvision 模型期望输入是单通道灰度图，像素值范围 [-1024, 1024]
    """
    def __init__(
        self,
        num_classes: int = 14,
        weights: str = "densenet121-res224-all",
        freeze_backbone: bool = False,
        feature_extract_only: bool = False,
    ):
        """
        Args:
            num_classes: 输出类别数（ChestX-ray14 为 14）
            weights: torchxrayvision 预训练权重名称
            freeze_backbone: 是否冻结 backbone 权重（仅训练分类头）
            feature_extract_only: 如果为 True，仅提取特征，不添加新分类头
        """
        super().__init__()
        
        try:
            import torchxrayvision as xrv
        except ImportError:
            raise ImportError(
                "请先安装 torchxrayvision: pip install torchxrayvision\n"
                "或: pip install git+https://github.com/mlmed/torchxrayvision.git"
            )
        
        self.weights_name = weights
        self.feature_extract_only = feature_extract_only
        
        # 加载 torchxrayvision 的 DenseNet121
        print(f"===> Loading torchxrayvision DenseNet with weights: {weights}")
        self.xrv_model = xrv.models.DenseNet(weights=weights)
        
        # torchxrayvision DenseNet 的输出特征维度是 1024
        self.in_features = 1024
        
        # 获取原始模型的输出类别（用于参考）
        self.xrv_pathologies = self.xrv_model.pathologies
        print(f"  >> Original pathologies ({len(self.xrv_pathologies)}): {self.xrv_pathologies}")
        
        if not feature_extract_only:
            # 替换分类头为我们需要的类别数
            self.classifier = nn.Linear(self.in_features, num_classes)
            # 初始化新的分类头
            nn.init.xavier_uniform_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)
        
        # 是否冻结 backbone
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """冻结 backbone 的所有参数"""
        for name, param in self.xrv_model.named_parameters():
            if "classifier" not in name:  # 保持分类头可训练
                param.requires_grad = False
        print("  >> Backbone frozen, only classifier will be trained")
    
    def unfreeze_backbone(self):
        """解冻 backbone 的所有参数"""
        for param in self.xrv_model.parameters():
            param.requires_grad = True
        print("  >> Backbone unfrozen, all parameters will be trained")
    
    def get_features(self, x):
        """提取特征（不经过分类头）"""
        # torchxrayvision 的 features2 方法返回 1024 维特征
        features = self.xrv_model.features2(x)
        return features
    
    def forward(self, x):
        """
        Args:
            x: 输入图像张量
               - 如果是 3 通道 RGB，会自动转换为单通道
               - 期望像素值范围已经是 [-1024, 1024]（由 dataset 预处理完成）
        """
        # 如果输入是 3 通道，取平均转为单通道
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        
        if self.feature_extract_only:
            # 仅返回特征
            return self.get_features(x)
        
        # 提取特征
        features = self.get_features(x)
        
        # 通过新的分类头
        logits = self.classifier(features)
        
        return logits


class TorchXRayVisionDenseNetWithMapping(nn.Module):
    """
    带标签映射的 torchxrayvision DenseNet121。
    
    可以选择：
    1. 使用原始模型的输出 + 标签映射（利用预训练的分类头知识）
    2. 替换分类头重新训练
    3. 两者结合（原始输出 + 新分类头的加权组合）
    """
    def __init__(
        self,
        target_pathologies: list,
        weights: str = "densenet121-res224-all",
        mode: str = "new_head",  # "mapping", "new_head", "hybrid"
        freeze_backbone: bool = False,
    ):
        """
        Args:
            target_pathologies: 目标病理类别列表（如 ChestX-ray14 的 14 类）
            weights: torchxrayvision 预训练权重
            mode: 
                - "mapping": 使用原始分类头 + 标签映射
                - "new_head": 替换新分类头
                - "hybrid": 原始 + 新分类头加权组合
            freeze_backbone: 是否冻结 backbone
        """
        super().__init__()
        
        try:
            import torchxrayvision as xrv
        except ImportError:
            raise ImportError("请先安装 torchxrayvision")
        
        self.mode = mode
        self.target_pathologies = target_pathologies
        self.num_classes = len(target_pathologies)
        
        # 加载模型
        print(f"===> Loading torchxrayvision DenseNet: {weights}, mode={mode}")
        self.xrv_model = xrv.models.DenseNet(weights=weights)
        self.xrv_pathologies = list(self.xrv_model.pathologies)
        
        # 建立标签映射
        self._build_label_mapping()
        
        if mode in ["new_head", "hybrid"]:
            self.in_features = 1024
            self.classifier = nn.Linear(self.in_features, self.num_classes)
            nn.init.xavier_uniform_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)
        
        if mode == "hybrid":
            # 可学习的融合权重
            self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        
        if freeze_backbone:
            self._freeze_backbone()
    
    def _build_label_mapping(self):
        """建立 torchxrayvision 输出到目标标签的映射"""
        # 映射字典：目标标签 -> torchxrayvision 标签索引
        self.label_mapping = {}
        self.mapped_indices = []
        
        for i, target in enumerate(self.target_pathologies):
            # 尝试精确匹配
            if target in self.xrv_pathologies:
                xrv_idx = self.xrv_pathologies.index(target)
                self.label_mapping[target] = xrv_idx
                self.mapped_indices.append((i, xrv_idx))
            else:
                # 尝试模糊匹配（处理命名差异）
                target_lower = target.lower().replace("_", " ")
                found = False
                for j, xrv_path in enumerate(self.xrv_pathologies):
                    if target_lower in xrv_path.lower() or xrv_path.lower() in target_lower:
                        self.label_mapping[target] = j
                        self.mapped_indices.append((i, j))
                        found = True
                        break
                if not found:
                    self.label_mapping[target] = None
        
        print(f"  >> Label mapping: {len([m for m in self.label_mapping.values() if m is not None])}/{self.num_classes} matched")
        for target, xrv_idx in self.label_mapping.items():
            if xrv_idx is not None:
                print(f"     {target} -> {self.xrv_pathologies[xrv_idx]} (idx={xrv_idx})")
            else:
                print(f"     {target} -> NOT FOUND")
    
    def _freeze_backbone(self):
        for name, param in self.xrv_model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
    
    def forward(self, x):
        # 转为单通道
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        
        if self.mode == "mapping":
            # 使用原始模型输出 + 映射
            xrv_output = self.xrv_model(x)  # (B, num_xrv_classes)
            logits = torch.zeros(x.size(0), self.num_classes, device=x.device)
            for target_idx, xrv_idx in self.mapped_indices:
                logits[:, target_idx] = xrv_output[:, xrv_idx]
            return logits
        
        elif self.mode == "new_head":
            # 提取特征 + 新分类头
            features = self.xrv_model.features2(x)
            logits = self.classifier(features)
            return logits
        
        elif self.mode == "hybrid":
            # 混合模式：原始输出 + 新分类头的加权组合
            features = self.xrv_model.features2(x)
            new_logits = self.classifier(features)
            
            # 原始模型输出映射
            xrv_output = self.xrv_model(x)
            mapped_logits = torch.zeros(x.size(0), self.num_classes, device=x.device)
            for target_idx, xrv_idx in self.mapped_indices:
                mapped_logits[:, target_idx] = xrv_output[:, xrv_idx]
            
            # 加权融合
            w = torch.sigmoid(self.fusion_weight)
            logits = w * new_logits + (1 - w) * mapped_logits
            return logits
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
