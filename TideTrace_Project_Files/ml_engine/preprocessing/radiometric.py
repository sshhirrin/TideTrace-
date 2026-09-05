"""
SAR Radiometric Calibration and Backscatter Normalization.
Converts Digital Numbers (DN) to Sigma Naught (sigma_0 in dB) and applies contrast scaling.
"""
import numpy as np
from typing import Tuple, Dict, Any

def calibrate_sigma_naught_db(dn_array: np.ndarray, calibration_factor_db: float = 83.0) -> np.ndarray:
    """
    Computes radar backscatter coefficient sigma_0 in decibels (dB):
      sigma_0_dB = 10 * log10(DN^2) - K_cal
    Standard formula for Sentinel-1 GRD products.
    """
    dn_pos = np.maximum(dn_array.astype(np.float32), 1.0)
    sigma_0 = (dn_pos ** 2) / (10.0 ** (calibration_factor_db / 10.0))
    sigma_0_db = 10.0 * np.log10(np.maximum(sigma_0, 1e-7))
    return sigma_0_db

def normalize_raster_percentiles(raster: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    """
    Normalizes raster to [0.0, 1.0] using robust percentile clipping to resist extreme radar specularities.
    """
    p2, p98 = np.percentile(raster, (p_low, p_high))
    if p98 - p2 < 1e-5:
        p2 = float(np.min(raster))
        p98 = float(np.max(raster)) + 1e-4
    normalized = np.clip((raster - p2) / (p98 - p2 + 1e-7), 0.0, 1.0)
    return normalized.astype(np.float32)

def compute_polarimetric_ratio(vv_channel: np.ndarray, vh_channel: np.ndarray) -> np.ndarray:
    """
    Computes cross-polarization ratio VV/VH.
    Mineral oil strongly damps co-polarized VV capillary waves, altering the ratio.
    """
    ratio = (vv_channel + 1e-4) / (vh_channel + 1e-4)
    return ratio.astype(np.float32)
