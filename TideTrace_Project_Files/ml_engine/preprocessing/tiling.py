"""
Sliding-Window Tiling and Seamless Reconstruction Engine.
Enables memory-efficient inference on gigapixel SAR scenes without boundary seam artifacts.
"""
import numpy as np
import torch
from typing import List, Tuple, Generator

def generate_tiles(raster: np.ndarray, tile_size: int = 256, overlap: int = 32) -> Generator[Tuple[int, int, np.ndarray], None, None]:
    """Generates (y, x, tile) slices from a 2D raster with configurable overlap."""
    H, W = raster.shape
    stride = tile_size - overlap

    y_indices = list(range(0, H - tile_size + 1, stride))
    if not y_indices or y_indices[-1] + tile_size < H:
        y_indices.append(max(0, H - tile_size))

    x_indices = list(range(0, W - tile_size + 1, stride))
    if not x_indices or x_indices[-1] + tile_size < W:
        x_indices.append(max(0, W - tile_size))

    for y in y_indices:
        for x in x_indices:
            tile = raster[y:y+tile_size, x:x+tile_size]
            yield y, x, tile

def get_hann_weight_window(tile_size: int) -> np.ndarray:
    """Computes a 2D Hann window for smooth cosine edge attenuation during tile blending."""
    h1 = np.hanning(tile_size)
    h2 = np.outer(h1, h1) + 1e-4
    return h2.astype(np.float32)
