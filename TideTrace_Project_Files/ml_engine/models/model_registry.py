"""
Model Registry for TideTrace SAR Segmentation Architectures.
Provides unified model instantiation, weight loading, and inference interface.
"""
import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from ml_engine.models.unet_resnet34 import UNetResNet34
from ml_engine.models.unet_plusplus import UNetPlusPlus
from ml_engine.models.segformer import SegFormer
from ml_engine.unet_model import SAROilSpillUNet

MODEL_REGISTRY = {
    "unet_resnet34": UNetResNet34,
    "unet_standard": SAROilSpillUNet,
    "unet_plusplus": UNetPlusPlus,
    "segformer": SegFormer,
}

def create_model(model_name: str = "unet_resnet34", in_channels: int = 1, num_classes: int = 1, **kwargs) -> nn.Module:
    """Factory to instantiate any registered segmentation model."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(MODEL_REGISTRY.keys())}")
    model_cls = MODEL_REGISTRY[model_name]
    return model_cls(in_channels=in_channels, num_classes=num_classes, **kwargs)

def load_checkpoint(model: nn.Module, checkpoint_path: str, device: str = "cpu") -> Dict[str, Any]:
    """Loads weights into model from checkpoint dict."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in ckpt:
        # Filter matching keys
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in ckpt["model_state_dict"].items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    else:
        model.load_state_dict(ckpt)
        meta = {}
    model.eval()
    return meta
