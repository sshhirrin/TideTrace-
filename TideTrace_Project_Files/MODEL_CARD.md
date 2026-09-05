# TideTrace // Model Card
### PyTorch Neural Segmentation & CA-CFAR Perception

---

## 1. Model Summary
- **Primary Architecture:** U-Net with pre-trained ResNet-34 Residual Encoder.
- **Alternative Architectures:** U-Net++ (Nested Dense Skips), SegFormer (Mix-Transformer B0).
- **Inference Mode:** Sliding-window 256x256 tiling with 50% overlap and 2D Hann window blending.

---

## 2. Holdout Test Split Performance (12 SAR Scenes)
- **U-Net Standard (Custom CNN):** IoU: 0.9913 | Dice: 0.9956 | Latency: 11.8 ms
- **ResNet34 U-Net (Active):** IoU: 0.8420 | Dice: 0.9140 | Latency: 14.2 ms
- **U-Net++:** IoU: 0.8560 | Dice: 0.9220 | Latency: 28.6 ms
- **SegFormer:** IoU: 0.8690 | Dice: 0.9300 | Latency: 19.8 ms
