"""
PyTorch U-Net with ResNet-34 Encoder Backbone for SAR Oil Spill Segmentation.

Architecture:
  - Input: C-band SAR backscatter (1-channel VV or 2-channel VV+VH)
  - Encoder: ResNet-34 residual blocks with 4 stages (64, 128, 256, 512 channels)
  - Decoder: Progressive upsampling with skip-connections from corresponding encoder stages
  - Head: 1x1 Convolution with Sigmoid activation outputting pixel probability [0, 1]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict, Any

class BasicResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class ResNet34Encoder(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet-34 layers: [3, 4, 6, 3] blocks
        self.layer1 = self._make_layer(BasicResidualBlock, 64, 3, stride=1)
        self.layer2 = self._make_layer(BasicResidualBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(BasicResidualBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(BasicResidualBlock, 512, 3, stride=2)

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x0 = self.relu(self.bn1(self.conv1(x))) # Stage 0: 64 ch, 1/2 res
        x1 = self.maxpool(x0)                   # Stage 1: 64 ch, 1/4 res
        x1 = self.layer1(x1)
        x2 = self.layer2(x1)                    # Stage 2: 128 ch, 1/8 res
        x3 = self.layer3(x2)                    # Stage 3: 256 ch, 1/16 res
        x4 = self.layer4(x3)                    # Stage 4: 512 ch, 1/32 res
        return [x, x0, x1, x2, x3, x4]

class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNetResNet34(nn.Module):
    """
    U-Net with ResNet-34 Backbone.
    Input: (B, C, H, W)
    Output: (B, num_classes, H, W) in range [0, 1]
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.encoder = ResNet34Encoder(in_channels=in_channels)

        # Center bridge
        self.center = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Decoder stages
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 32)
        self.dec0 = DecoderBlock(32, in_channels, 16)

        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_orig, x0, x1, x2, x3, x4 = self.encoder(x)
        c = self.center(x4)
        d4 = self.dec4(c, x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)
        d0 = self.dec0(d1, x_orig)
        logits = self.final_conv(d0)
        return torch.sigmoid(logits)

    def predict_large_raster(
        self,
        raster: np.ndarray,
        tile_size: int = 256,
        overlap: int = 32,
        threshold: float = 0.5,
        device: str = "cpu"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Infers across large 2D SAR rasters using sliding-window tiling and Hann window blending.
        Returns: (binary_mask, probability_map)
        """
        self.eval()
        self.to(device)

        H, W = raster.shape
        stride = tile_size - overlap
        prob_map = np.zeros((H, W), dtype=np.float32)
        weight_map = np.zeros((H, W), dtype=np.float32)

        hann_1d = np.hanning(tile_size)
        hann_2d = np.outer(hann_1d, hann_1d) + 1e-4

        if H < tile_size or W < tile_size:
            pad_h = max(0, tile_size - H)
            pad_w = max(0, tile_size - W)
            padded = np.pad(raster, ((0, pad_h), (0, pad_w)), mode='reflect')
            inp = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).float().to(device)
            with torch.no_grad():
                out = self(inp).squeeze().cpu().numpy()
            sub_out = out[:H, :W]
            return (sub_out > threshold).astype(np.uint8), sub_out

        y_steps = list(range(0, H - tile_size + 1, stride))
        if y_steps[-1] + tile_size < H:
            y_steps.append(H - tile_size)

        x_steps = list(range(0, W - tile_size + 1, stride))
        if x_steps[-1] + tile_size < W:
            x_steps.append(W - tile_size)

        for y in y_steps:
            for x in x_steps:
                tile = raster[y:y+tile_size, x:x+tile_size]
                inp = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).float().to(device)
                with torch.no_grad():
                    pred = self(inp).squeeze().cpu().numpy()
                prob_map[y:y+tile_size, x:x+tile_size] += pred * hann_2d
                weight_map[y:y+tile_size, x:x+tile_size] += hann_2d

        norm_prob = prob_map / np.maximum(weight_map, 1e-6)
        binary_mask = (norm_prob > threshold).astype(np.uint8)
        return binary_mask, norm_prob
