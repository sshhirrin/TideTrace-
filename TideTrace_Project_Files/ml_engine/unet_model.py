"""
PyTorch U-Net Architecture for SAR Oil Spill Segmentation.

Implements a deep convolutional Encoder-Decoder with skip connections,
tailored for Synthetic Aperture Radar (SAR) C-band VV/VH polarization imagery.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Handle shape differences due to padding/striding
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # Skip connection concatenation
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class SAROilSpillUNet(nn.Module):
    """
    Standard U-Net for SAR Oil Spill Segmentation.
    Input: (B, C, H, W) where C=1 (VV) or C=2 (VV + VH).
    Output: (B, 1, H, W) containing per-pixel oil slick probability [0.0, 1.0].
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 1, bilinear: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return torch.sigmoid(logits)

    def predict_large_raster(
        self,
        raster: np.ndarray,
        tile_size: int = 256,
        overlap: int = 32,
        threshold: float = 0.5,
        device: str = "cpu"
    ) -> np.ndarray:
        """
        Runs sliding-window inference across large SAR rasters with seamless blending.
        raster: 2D numpy array (H, W) normalized to [0.0, 1.0].
        Returns: binary mask (H, W) where 1 = Oil Slick, 0 = Sea Surface.
        """
        self.eval()
        self.to(device)

        H, W = raster.shape
        stride = tile_size - overlap
        prob_map = np.zeros((H, W), dtype=np.float32)
        weight_map = np.zeros((H, W), dtype=np.float32)

        # 2D Hanning window for smooth border blending
        hann_1d = np.hanning(tile_size)
        hann_2d = np.outer(hann_1d, hann_1d) + 1e-4

        y_steps = range(0, H - tile_size + 1, stride)
        x_steps = range(0, W - tile_size + 1, stride)

        # If raster is smaller than tile_size, pad it
        if H < tile_size or W < tile_size:
            pad_h = max(0, tile_size - H)
            pad_w = max(0, tile_size - W)
            padded = np.pad(raster, ((0, pad_h), (0, pad_w)), mode='reflect')
            inp = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).float().to(device)
            with torch.no_grad():
                out = self(inp).squeeze().cpu().numpy()
            return (out[:H, :W] > threshold).astype(np.uint8)

        for y in y_steps:
            for x in x_steps:
                tile = raster[y:y+tile_size, x:x+tile_size]
                inp = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).float().to(device)
                with torch.no_grad():
                    pred = self(inp).squeeze().cpu().numpy()
                
                prob_map[y:y+tile_size, x:x+tile_size] += pred * hann_2d
                weight_map[y:y+tile_size, x:x+tile_size] += hann_2d

        # Normalize overlapping blends
        weight_map[weight_map == 0] = 1.0
        final_prob = prob_map / weight_map
        return (final_prob > threshold).astype(np.uint8), final_prob
