# models_attention_crop.py
"""
Attention-Guided Crop 增强模块

这是一个可插拔的模块，可以包装任何现有的 backbone（XRV 或 timm），
为其添加高分辨率局部特征提取能力。

使用方式：
    # 包装 XRV 模型
    base_model = TorchXRayVisionDenseNet(num_classes=14, weights="...")
    model = AttentionGuidedWrapper(base_model, backbone_type="xrv", ...)
    
    # 包装 timm 模型
    base_model = TimmMultiLabelModel("convnext_base", num_classes=14)
    model = AttentionGuidedWrapper(base_model, backbone_type="timm", ...)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Union

try:
    import torchxrayvision as xrv
except ImportError:
    xrv = None


class SpatialAttentionModule(nn.Module):
    """
    空间注意力模块：从特征图生成 attention map
    """
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        
        # 通道注意力
        channel_attn = self.channel_attention(x).view(B, C, 1, 1)
        x = x * channel_attn
        
        # 空间注意力
        attention_map = self.spatial_attention(x)
        attended_features = x * attention_map
        
        return attended_features, attention_map


class AttentionGuidedWrapper(nn.Module):
    """
    Attention-Guided Crop Wrapper
    
    可以包装任何现有的 backbone，为其添加：
    1. 低分辨率全图 -> 全局特征 + Attention Map
    2. 根据 Attention 热点从高分辨率图裁剪局部 Patch
    3. 局部 Patch -> 局部特征
    4. 融合全局 + 局部特征 -> 分类
    
    支持的 backbone 类型：
    - "xrv": torchxrayvision 的 DenseNet 模型
    - "timm": timm 库的模型
    - "custom": 自定义模型（需要实现 get_features 方法）
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        backbone_type: str = "xrv",  # "xrv", "timm", "custom"
        num_classes: int = 14,
        feature_dim: int = 1024,
        num_crops: int = 2,
        crop_size: int = 224,
        high_res_size: int = 512,
        fusion_type: str = "concat_attention",  # concat, add, concat_attention
        fusion_dim: int = 512,
        share_backbone: bool = True,  # 是否共享 backbone
        min_attention_threshold: float = 0.3,
    ):
        """
        Args:
            backbone: 基础模型（已经初始化好的）
            backbone_type: backbone 类型
            num_classes: 输出类别数
            feature_dim: backbone 特征维度（XRV=1024, ResNet50=2048, etc.）
            num_crops: 裁剪的局部区域数量
            crop_size: 裁剪后的尺寸
            high_res_size: 高分辨率图像尺寸
            fusion_type: 融合方式
            fusion_dim: 融合后的特征维度
            share_backbone: 全局/局部分支是否共享 backbone
            min_attention_threshold: 最小注意力阈值
        """
        super().__init__()
        
        self.backbone_type = backbone_type
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.num_crops = num_crops
        self.crop_size = crop_size
        self.high_res_size = high_res_size
        self.fusion_type = fusion_type
        self.min_attention_threshold = min_attention_threshold
        self.share_backbone = share_backbone
        
        # 保存原始 backbone
        self.global_backbone = backbone
        
        # 如果不共享，为局部分支创建独立的 backbone
        if not share_backbone:
            import copy
            self.local_backbone = copy.deepcopy(backbone)
        else:
            self.local_backbone = self.global_backbone
        
        # 移除原始 backbone 的分类头（我们会创建新的融合分类头）
        self._remove_classifier()
        
        # 空间注意力模块
        self.attention_module = SpatialAttentionModule(
            in_channels=feature_dim,
            reduction=16
        )
        
        # 融合模块
        self._build_fusion_module(fusion_dim)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_out_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        
        # 推断特征图尺寸（假设输入 224x224）
        self._infer_feature_map_size()
    
    def _remove_classifier(self):
        """移除原始分类头"""
        if self.backbone_type == "xrv":
            # XRV 的 DenseNet 分类头在 xrv_model.classifier
            if hasattr(self.global_backbone, 'classifier'):
                self.global_backbone.classifier = nn.Identity()
        elif self.backbone_type == "timm":
            # timm 模型的分类头通常是 model.fc 或 model.head
            if hasattr(self.global_backbone, 'model'):
                inner_model = self.global_backbone.model
                if hasattr(inner_model, 'fc'):
                    inner_model.fc = nn.Identity()
                elif hasattr(inner_model, 'head'):
                    inner_model.head = nn.Identity()
                elif hasattr(inner_model, 'classifier'):
                    inner_model.classifier = nn.Identity()
    
    def _build_fusion_module(self, fusion_dim: int):
        """构建融合模块"""
        global_dim = self.feature_dim
        local_dim = self.feature_dim
        
        if self.fusion_type == "concat":
            total_dim = global_dim + local_dim * self.num_crops
            self.fusion = nn.Sequential(
                nn.Linear(total_dim, fusion_dim),
                nn.BatchNorm1d(fusion_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
            )
            self.fusion_out_dim = fusion_dim
            
        elif self.fusion_type == "add":
            self.global_proj = nn.Linear(global_dim, fusion_dim)
            self.local_proj = nn.Linear(local_dim, fusion_dim)
            self.fusion_out_dim = fusion_dim
            
        elif self.fusion_type == "concat_attention":
            total_dim = global_dim + local_dim * self.num_crops
            self.attention_weights = nn.Sequential(
                nn.Linear(total_dim, total_dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(total_dim // 4, 1 + self.num_crops),
                nn.Softmax(dim=1)
            )
            self.global_proj = nn.Linear(global_dim, fusion_dim)
            self.local_proj = nn.Linear(local_dim, fusion_dim)
            self.fusion_out_dim = fusion_dim
    
    def _infer_feature_map_size(self):
        """推断特征图尺寸"""
        # 根据 backbone 类型设置默认值
        # DenseNet121 对于 224 输入: 7x7
        # ResNet50 对于 224 输入: 7x7
        # ConvNeXt 对于 224 输入: 7x7
        self.feature_map_size = 7
    
    def _to_single_channel(self, x: torch.Tensor) -> torch.Tensor:
        """将 3 通道转为单通道（仅 XRV 需要）"""
        if self.backbone_type == "xrv" and x.shape[1] == 3:
            return x.mean(dim=1, keepdim=True)
        return x
    
    def get_feature_map(self, backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        获取 backbone 的特征图（不经过全局池化）
        
        Returns:
            feature_map: (B, C, H, W)
        """
        if self.backbone_type == "xrv":
            x = self._to_single_channel(x)
            # XRV DenseNet 的 features 方法返回特征图
            if hasattr(backbone, 'xrv_model'):
                features = backbone.xrv_model.features(x)
            else:
                features = backbone.features(x)
            features = F.relu(features, inplace=True)
            return features
            
        elif self.backbone_type == "timm":
            # timm 模型需要特殊处理
            if hasattr(backbone, 'model'):
                inner_model = backbone.model
            else:
                inner_model = backbone
            
            # 尝试获取特征图
            if hasattr(inner_model, 'forward_features'):
                features = inner_model.forward_features(x)
                # 有些模型返回的是 (B, H, W, C)，需要转换
                if features.dim() == 4 and features.shape[1] != self.feature_dim:
                    features = features.permute(0, 3, 1, 2)
                return features
            else:
                # 回退方案：使用 hook 获取特征
                raise NotImplementedError(
                    f"该 timm 模型不支持 forward_features，请使用支持的模型"
                )
        
        elif self.backbone_type == "custom":
            if hasattr(backbone, 'get_feature_map'):
                return backbone.get_feature_map(x)
            else:
                raise NotImplementedError(
                    "custom backbone 需要实现 get_feature_map 方法"
                )
    
    def get_global_features(self, backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        获取全局特征向量（经过全局池化）
        
        Returns:
            features: (B, feature_dim)
        """
        if self.backbone_type == "xrv":
            x = self._to_single_channel(x)
            if hasattr(backbone, 'xrv_model'):
                return backbone.xrv_model.features2(x)
            elif hasattr(backbone, 'get_features'):
                return backbone.get_features(x)
            else:
                # 手动获取
                feature_map = self.get_feature_map(backbone, x)
                return F.adaptive_avg_pool2d(feature_map, 1).view(x.size(0), -1)
        
        elif self.backbone_type == "timm":
            feature_map = self.get_feature_map(backbone, x)
            if feature_map.dim() == 4:
                return F.adaptive_avg_pool2d(feature_map, 1).view(x.size(0), -1)
            elif feature_map.dim() == 3:
                # Transformer 风格: (B, N, C) -> 取 CLS token 或平均
                return feature_map.mean(dim=1)
            return feature_map
        
        elif self.backbone_type == "custom":
            if hasattr(backbone, 'get_global_features'):
                return backbone.get_global_features(x)
            else:
                feature_map = self.get_feature_map(backbone, x)
                return F.adaptive_avg_pool2d(feature_map, 1).view(x.size(0), -1)
    
    def find_attention_regions(
        self,
        attention_map: torch.Tensor,
        high_res_size: int,
    ) -> List[Tuple[int, int, int, int]]:
        """
        找到 attention map 中的热点区域坐标
        """
        attn = attention_map.squeeze().detach().cpu().numpy()
        H, W = attn.shape
        
        scale = high_res_size / self.feature_map_size
        crop_size_in_feature = int(self.crop_size / scale)
        
        regions = []
        attn_copy = attn.copy()
        
        for _ in range(self.num_crops):
            max_idx = np.unravel_index(np.argmax(attn_copy), attn_copy.shape)
            max_val = attn_copy[max_idx]
            
            if max_val < self.min_attention_threshold:
                cy, cx = H // 2, W // 2
            else:
                cy, cx = max_idx
            
            half_size = crop_size_in_feature // 2
            top = max(0, cy - half_size)
            left = max(0, cx - half_size)
            
            if top + crop_size_in_feature > H:
                top = H - crop_size_in_feature
            if left + crop_size_in_feature > W:
                left = W - crop_size_in_feature
            
            top = max(0, top)
            left = max(0, left)
            
            top_hr = int(top * scale)
            left_hr = int(left * scale)
            
            top_hr = min(top_hr, high_res_size - self.crop_size)
            left_hr = min(left_hr, high_res_size - self.crop_size)
            top_hr = max(0, top_hr)
            left_hr = max(0, left_hr)
            
            regions.append((top_hr, left_hr, self.crop_size, self.crop_size))
            
            suppress_top = max(0, cy - crop_size_in_feature)
            suppress_left = max(0, cx - crop_size_in_feature)
            suppress_bottom = min(H, cy + crop_size_in_feature)
            suppress_right = min(W, cx + crop_size_in_feature)
            attn_copy[suppress_top:suppress_bottom, suppress_left:suppress_right] = 0
        
        return regions
    
    def crop_regions(
        self,
        high_res_images: torch.Tensor,
        attention_maps: torch.Tensor,
    ) -> torch.Tensor:
        """根据注意力图从高分辨率图像中裁剪区域"""
        B, C, H_hr, W_hr = high_res_images.shape
        
        all_crops = []
        
        for i in range(B):
            attn_map = attention_maps[i]
            regions = self.find_attention_regions(attn_map, H_hr)
            
            for top, left, h, w in regions:
                crop = high_res_images[i:i+1, :, top:top+h, left:left+w]
                if crop.shape[2] != self.crop_size or crop.shape[3] != self.crop_size:
                    crop = F.interpolate(
                        crop, size=(self.crop_size, self.crop_size),
                        mode='bilinear', align_corners=False
                    )
                all_crops.append(crop)
        
        return torch.cat(all_crops, dim=0)
    
    def fuse_features(
        self,
        global_feat: torch.Tensor,
        local_feats: List[torch.Tensor],
    ) -> torch.Tensor:
        """融合全局和局部特征"""
        if self.fusion_type == "concat":
            all_feats = [global_feat] + local_feats
            concat_feat = torch.cat(all_feats, dim=1)
            return self.fusion(concat_feat)
        
        elif self.fusion_type == "add":
            global_proj = self.global_proj(global_feat)
            local_proj = sum(self.local_proj(lf) for lf in local_feats) / len(local_feats)
            return global_proj + local_proj
        
        elif self.fusion_type == "concat_attention":
            all_feats = [global_feat] + local_feats
            concat_for_attn = torch.cat(all_feats, dim=1)
            attn_weights = self.attention_weights(concat_for_attn)
            
            global_proj = self.global_proj(global_feat)
            local_projs = [self.local_proj(lf) for lf in local_feats]
            
            fused = attn_weights[:, 0:1] * global_proj
            for i, local_proj in enumerate(local_projs):
                fused = fused + attn_weights[:, i+1:i+2] * local_proj
            
            return fused
    
    def forward(
        self,
        x_low: torch.Tensor,
        x_high: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播
        
        Args:
            x_low: (B, C, 224, 224) 低分辨率全图
            x_high: (B, C, high_res, high_res) 高分辨率全图
                    如果为 None，则从 x_low 上采样
            return_attention: 是否返回注意力图
        """
        B = x_low.shape[0]
        device = x_low.device
        
        # 1. 获取全局特征图和注意力
        feature_map = self.get_feature_map(self.global_backbone, x_low)
        attended_features, attention_map = self.attention_module(feature_map)
        global_features = F.adaptive_avg_pool2d(attended_features, 1).view(B, -1)
        
        # 2. 准备高分辨率图像
        if x_high is None:
            x_high = F.interpolate(
                x_low, size=(self.high_res_size, self.high_res_size),
                mode='bilinear', align_corners=False
            )
        
        # 3. 根据注意力图裁剪高分辨率局部区域
        crops = self.crop_regions(x_high, attention_map)
        
        # 4. 获取局部特征
        local_features_flat = self.get_global_features(self.local_backbone, crops)
        
        # 重组为 List
        local_features_list = []
        for i in range(self.num_crops):
            indices = torch.arange(B, device=device) * self.num_crops + i
            local_feat = local_features_flat[indices]
            local_features_list.append(local_feat)
        
        # 5. 融合
        fused_features = self.fuse_features(global_features, local_features_list)
        
        # 6. 分类
        logits = self.classifier(fused_features)
        
        if return_attention:
            return logits, attention_map
        return logits
    
    def forward_inference(
        self,
        x_low: torch.Tensor,
        x_high: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """推理模式，返回 logits 和 attention_map"""
        return self.forward(x_low, x_high, return_attention=True)


def wrap_model_with_attention_crop(
    model: nn.Module,
    backbone_type: str = "xrv",
    num_classes: int = 14,
    feature_dim: int = 1024,
    num_crops: int = 2,
    crop_size: int = 224,
    high_res_size: int = 512,
    fusion_type: str = "concat_attention",
    share_backbone: bool = True,
) -> AttentionGuidedWrapper:
    """
    便捷函数：将现有模型包装成 Attention-Guided 版本
    
    Args:
        model: 原始模型
        backbone_type: "xrv" 或 "timm"
        其他参数同 AttentionGuidedWrapper
    
    Returns:
        包装后的模型
        
    Example:
        >>> from src.models import TorchXRayVisionDenseNet
        >>> base_model = TorchXRayVisionDenseNet(num_classes=14, weights="...")
        >>> ag_model = wrap_model_with_attention_crop(base_model, backbone_type="xrv")
    """
    return AttentionGuidedWrapper(
        backbone=model,
        backbone_type=backbone_type,
        num_classes=num_classes,
        feature_dim=feature_dim,
        num_crops=num_crops,
        crop_size=crop_size,
        high_res_size=high_res_size,
        fusion_type=fusion_type,
        share_backbone=share_backbone,
    )


# ==================== 可视化工具 ====================
def visualize_attention(
    image: torch.Tensor,
    attention_map: torch.Tensor,
    save_path: Optional[str] = None,
):
    """可视化注意力图"""
    import matplotlib.pyplot as plt
    
    if image.dim() == 3:
        if image.shape[0] == 1:
            img = image.squeeze(0).detach().cpu().numpy()
        else:
            img = image.permute(1, 2, 0).detach().cpu().numpy()
            img = img.mean(axis=2)
    else:
        img = image.detach().cpu().numpy()
    
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    attn = attention_map.squeeze().detach().cpu().numpy()
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
    
    from scipy.ndimage import zoom
    scale_h = img.shape[0] / attn.shape[0]
    scale_w = img.shape[1] / attn.shape[1]
    attn_upsampled = zoom(attn, (scale_h, scale_w), order=1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(attn, cmap='jet')
    axes[1].set_title('Attention Map')
    axes[1].axis('off')
    
    axes[2].imshow(img, cmap='gray')
    axes[2].imshow(attn_upsampled, cmap='jet', alpha=0.5)
    axes[2].set_title('Attention Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # 测试包装 XRV 模型
    print("测试 AttentionGuidedWrapper...")
    
    from src.models import TorchXRayVisionDenseNet
    
    # 创建基础模型
    base_model = TorchXRayVisionDenseNet(
        num_classes=14,
        weights="densenet121-res224-chex",
    )
    
    # 包装成 Attention-Guided 版本
    ag_model = wrap_model_with_attention_crop(
        base_model,
        backbone_type="xrv",
        num_classes=14,
        feature_dim=1024,
        num_crops=2,
    )
    
    # 测试前向传播
    x_low = torch.randn(2, 3, 224, 224)
    x_high = torch.randn(2, 3, 512, 512)
    
    logits, attn = ag_model(x_low, x_high, return_attention=True)
    print(f"Input low: {x_low.shape}")
    print(f"Input high: {x_high.shape}")
    print(f"Output logits: {logits.shape}")
    print(f"Attention map: {attn.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in ag_model.parameters())
    print(f"Total parameters: {total_params:,}")
