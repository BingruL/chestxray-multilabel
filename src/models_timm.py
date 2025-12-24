# models_timm.py

import torch
import torch.nn as nn
import timm

from src.cxr_config import NUM_CLASSES


def _strip_module_prefix(state_dict):
    """兼容 DDP 保存的权重，移除 'module.' 前缀。"""
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state[k] = v
    return new_state


class TimmMultiLabelModel(nn.Module):
    """
    使用 timm 调用各种主流 backbone，并支持自定义 ckpt。
    注意：新版 timm 使用 model_name.pretrained_tag 格式，例如：
    - ConvNeXt:      convnext_base.fb_in22k, convnext_base.fb_in22k_ft_in1k_384 ...
    - ViT/EVA/Swin:  vit_base_patch16_224.augreg_in21k, vit_base_patch16_384.augreg_in21k_ft_in1k ...
    - EfficientNet:  efficientnet_b0.ra_in1k, tf_efficientnet_b4.ns_jft_in1k ...
    """
    def __init__(
        self,
        backbone_name: str,
        pretrained: bool = True,
        checkpoint_path: str | None = None,
        num_classes: int = NUM_CLASSES,
        **create_kwargs,
    ):
        super().__init__()
        self.backbone_name = backbone_name

        # 保持 timm 的 create_model，可通过 create_kwargs 传更多参数（如 drop_path_rate）
        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=num_classes,  # 多标签 => 直接输出 num_classes 维 logits
            **create_kwargs,
        )

        # 如需加载额外的 CXR/自监督权重
        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            state = _strip_module_prefix(state)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            print(f"[Timm] loaded ckpt from {checkpoint_path}")
            print("  missing keys:", missing)
            print("  unexpected keys:", unexpected)

    def forward(self, x):
        return self.model(x)
