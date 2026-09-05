"""
2D Cell-Averaging Constant False Alarm Rate (CA-CFAR) Radar Ship Detector.
Detects bright metallic ship hulls against ocean sea clutter in SAR imagery.
Calculates Oriented Bounding Boxes (OBBs) and generates thumbnail radar zoom insets.
"""
import base64
import cv2
import numpy as np
import rasterio
from typing import List, Dict, Any, Tuple, Optional
from ml_engine.metrics import haversine_distance_km

class ShipDetectorBase:
    """Base interface for SAR ship detection (CFAR, YOLOv8-OBB, CNN)."""
    def detect_ships(self, raster: np.ndarray, transform: Any, pixel_size_m: float = 10.0) -> List[Dict[str, Any]]:
        raise NotImplementedError

class CACFARShipDetector(ShipDetectorBase):
    def __init__(
        self,
        guard_cells: int = 8,
        training_cells: int = 16,
        cfar_multiplier: float = 3.0,
        min_vessel_area_px: int = 1,
        max_vessel_area_px: int = 800
    ):
        self.guard_cells = guard_cells
        self.training_cells = training_cells
        self.cfar_multiplier = cfar_multiplier
        self.min_vessel_area_px = min_vessel_area_px
        self.max_vessel_area_px = max_vessel_area_px

    def _generate_crop_base64(self, raster: np.ndarray, cx: int, cy: int, crop_radius: int = 24) -> str:
        """Extracts a thumbnail crop around detected ship and encodes as base64 PNG."""
        H, W = raster.shape
        y1, y2 = max(0, cy - crop_radius), min(H, cy + crop_radius)
        x1, x2 = max(0, cx - crop_radius), min(W, cx + crop_radius)
        patch = raster[y1:y2, x1:x2]
        
        # Scale to 8-bit [0, 255]
        patch_8u = np.clip(patch * 255.0, 0, 255).astype(np.uint8)
        # Apply false-color or radar enhancement
        patch_colored = cv2.applyColorMap(patch_8u, cv2.COLORMAP_INFERNO)
        
        # Draw red oriented center marker
        ch, cw, _ = patch_colored.shape
        cv2.drawMarker(patch_colored, (cw // 2, ch // 2), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 8, 1)

        _, buf = cv2.imencode(".png", patch_colored)
        return "data:image/png;base64," + base64.b64encode(buf).decode("utf-8")

    def detect_ships(
        self,
        raster: np.ndarray,
        transform: Any,
        pixel_size_m: float = 10.0
    ) -> List[Dict[str, Any]]:
        H, W = raster.shape
        outer_k = 2 * (self.guard_cells + self.training_cells) + 1
        inner_k = 2 * self.guard_cells + 1

        outer_mean = cv2.blur(raster, (outer_k, outer_k))
        inner_mean = cv2.blur(raster, (inner_k, inner_k))

        outer_area = outer_k * outer_k
        inner_area = inner_k * inner_k
        train_area = outer_area - inner_area

        clutter_mean = (outer_mean * outer_area - inner_mean * inner_area) / max(1, train_area)

        outer_sq = cv2.blur(raster ** 2, (outer_k, outer_k))
        inner_sq = cv2.blur(raster ** 2, (inner_k, inner_k))
        clutter_var = np.maximum(0.0, (outer_sq * outer_area - inner_sq * inner_area) / max(1, train_area) - (clutter_mean ** 2))
        clutter_std = np.sqrt(clutter_var)

        cfar_thresh = clutter_mean + self.cfar_multiplier * clutter_std
        detection_mask = ((raster > cfar_thresh) & (raster > 0.60)).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        detection_mask = cv2.morphologyEx(detection_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(detection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_vessels = []
        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = int(cnt[0][0][0])
                cy = int(cnt[0][0][1])

            # Rotated bounding box for ship orientation and length (OBB)
            rect = cv2.minAreaRect(cnt)
            (rx, ry), (width_px, height_px), angle_deg = rect
            length_px = max(1.0, max(width_px, height_px))
            beam_px = max(1.0, min(width_px, height_px))
            est_length_m = round(max(30.0, length_px * pixel_size_m), 1)
            est_beam_m = round(max(8.0, beam_px * pixel_size_m), 1)

            # OBB 4 corners in pixel space
            box_pts = cv2.boxPoints(rect) # 4x2 array
            obb_geo = []
            for bp in box_pts:
                bx, by = bp[0], bp[1]
                glng, glat = rasterio.transform.xy(transform, by, bx, offset='center')
                obb_geo.append([round(glng, 6), round(glat, 6)])
            obb_geo.append(obb_geo[0]) # Closed polygon ring

            lng, lat = rasterio.transform.xy(transform, cy, cx, offset='center')

            peak_intensity = float(np.max(raster[max(0, cy-2):min(H, cy+3), max(0, cx-2):min(W, cx+3)]))
            clutter_val = max(0.01, float(clutter_mean[min(H-1, cy), min(W-1, cx)]))
            snr_db = round(10.0 * np.log10(peak_intensity / clutter_val), 1)

            crop_b64 = self._generate_crop_base64(raster, cx, cy)

            detected_vessels.append({
                "id": f"RADAR-SHIP-{idx+1:02d}",
                "px": cx,
                "py": cy,
                "lat": round(lat, 6),
                "lng": round(lng, 6),
                "estimated_length_m": est_length_m,
                "estimated_beam_m": est_beam_m,
                "heading_deg": round((angle_deg + 360.0) % 360.0, 1),
                "radar_intensity": round(peak_intensity, 3),
                "cfar_snr_db": max(6.0, snr_db),
                "obb_coordinates": obb_geo,
                "radar_crop_base64": crop_b64
            })

        return detected_vessels
