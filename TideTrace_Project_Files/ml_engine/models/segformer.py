"""
PyTorch SegFormer Architecture for SAR Multi-Scale Feature Segmentation.
Implements hierarchical Mix-Transformer (MiT) blocks with all-MLP lightweight decoder head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 7, stride: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2
        )
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return self.norm(x)

class EfficientSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, sr_ratio: int = 1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, sr_ratio: int = 1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x))
        return x

class SegFormer(nn.Module):
    """
    SegFormer model designed for Marine radar Earth observation.
    Provides multi-scale receptive fields without positional encoding bias.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 1):
        super().__init__()
        dims = [32, 64, 128, 256]

        self.patch_embed1 = OverlapPatchEmbed(in_channels, dims[0], patch_size=7, stride=4)
        self.block1 = TransformerBlock(dims[0], num_heads=1, sr_ratio=8)

        self.patch_embed2 = OverlapPatchEmbed(dims[0], dims[1], patch_size=3, stride=2)
        self.block2 = TransformerBlock(dims[1], num_heads=2, sr_ratio=4)

        self.patch_embed3 = OverlapPatchEmbed(dims[1], dims[2], patch_size=3, stride=2)
        self.block3 = TransformerBlock(dims[2], num_heads=4, sr_ratio=2)

        self.patch_embed4 = OverlapPatchEmbed(dims[2], dims[3], patch_size=3, stride=2)
        self.block4 = TransformerBlock(dims[3], num_heads=8, sr_ratio=1)

        # All-MLP Decoder
        decoder_dim = 128
        self.linear_c4 = nn.Linear(dims[3], decoder_dim)
        self.linear_c3 = nn.Linear(dims[2], decoder_dim)
        self.linear_c2 = nn.Linear(dims[1], decoder_dim)
        self.linear_c1 = nn.Linear(dims[0], decoder_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_h, orig_w = x.shape[2], x.shape[3]

        # Stage 1
        x1 = self.patch_embed1(x)
        B, C1, H1, W1 = x1.shape
        x1_flat = x1.flatten(2).permute(0, 2, 1)
        x1_flat = self.block1(x1_flat, H1, W1)
        c1 = x1_flat.permute(0, 2, 1).reshape(B, C1, H1, W1)

        # Stage 2
        x2 = self.patch_embed2(c1)
        B, C2, H2, W2 = x2.shape
        x2_flat = x2.flatten(2).permute(0, 2, 1)
        x2_flat = self.block2(x2_flat, H2, W2)
        c2 = x2_flat.permute(0, 2, 1).reshape(B, C2, H2, W2)

        # Stage 3
        x3 = self.patch_embed3(c2)
        B, C3, H3, W3 = x3.shape
        x3_flat = x3.flatten(2).permute(0, 2, 1)
        x3_flat = self.block3(x3_flat, H3, W3)
        c3 = x3_flat.permute(0, 2, 1).reshape(B, C3, H3, W3)

        # Stage 4
        x4 = self.patch_embed4(c3)
        B, C4, H4, W4 = x4.shape
        x4_flat = x4.flatten(2).permute(0, 2, 1)
        x4_flat = self.block4(x4_flat, H4, W4)
        c4 = x4_flat.permute(0, 2, 1).reshape(B, C4, H4, W4)

        # Decoder fusion to 1/4 resolution (H1, W1)
        _c4 = self.linear_c4(c4.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).reshape(B, -1, H4, W4)
        _c4 = F.interpolate(_c4, size=(H1, W1), mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).reshape(B, -1, H3, W3)
        _c3 = F.interpolate(_c3, size=(H1, W1), mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).reshape(B, -1, H2, W2)
        _c2 = F.interpolate(_c2, size=(H1, W1), mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).reshape(B, -1, H1, W1)

        fused = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        logits = self.classifier(fused)
        logits = F.interpolate(logits, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        return torch.sigmoid(logits)
