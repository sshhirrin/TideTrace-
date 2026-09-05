"""
SAR Oil Spill Segmentation & Detector Pipeline.

Uses Convolutional / U-Net feature representations to identify dark-slick regions
in SAR GeoTIFF imagery and converts detections into georeferenced GeoJSON polygons.
"""
import numpy as np
from typing import List, Dict, Any, Tuple
from ml_engine.metrics import (
    calculate_polygon_centroid,
    calculate_polygon_area_sqkm,
    calculate_polygon_perimeter_km,
    calculate_major_axis_and_orientation,
    estimate_oil_volume_m3
)
from ml_engine.lookalike_filter import LookAlikeClassifier

class SAROilSpillDetector:
    def __init__(self):
        self.lookalike_classifier = LookAlikeClassifier()

    def process_sar_scene(
        self,
        scene_id: str,
        base_lat: float,
        base_lng: float,
        wind_speed_ms: float,
        slick_specs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Processes a SAR observation scene. In a production pipeline, this consumes raw
        Sentinel-1 Level-1 GRD GeoTIFF rasters, runs U-Net inference, thresholds speckle noise,
        and vectorizes the output.
        
        slick_specs defines the detected candidates in the scene.
        """
        results = []
        for idx, spec in enumerate(slick_specs):
            coords = spec["polygon_coords"]  # list of [lng, lat]
            centroid_lat, centroid_lng = calculate_polygon_centroid(coords)
            area_sqkm = calculate_polygon_area_sqkm(coords)
            perimeter_km = calculate_polygon_perimeter_km(coords)
            major_axis_km, orientation_deg = calculate_major_axis_and_orientation(coords)
            
            radar_damping_db = spec.get("radar_damping_db", 8.5)
            edge_sharpness = spec.get("edge_sharpness", 0.85)

            is_lookalike, reason, confidence = self.lookalike_classifier.evaluate_slick(
                area_sqkm=area_sqkm,
                perimeter_km=perimeter_km,
                major_axis_km=major_axis_km,
                wind_speed_ms=wind_speed_ms,
                radar_damping_db=radar_damping_db,
                edge_gradient_sharpness=edge_sharpness
            )

            vol_m3 = estimate_oil_volume_m3(area_sqkm, thickness_microns=spec.get("thickness_microns", 2.5))

            slick_dict = {
                "id": f"SLICK-{scene_id[-4:]}-{idx+1:02d}",
                "geojson": {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords]
                    },
                    "properties": {
                        "id": f"SLICK-{scene_id[-4:]}-{idx+1:02d}",
                        "area_sqkm": area_sqkm,
                        "confidence": confidence,
                        "is_lookalike": is_lookalike
                    }
                },
                "centroid": {"lat": centroid_lat, "lng": centroid_lng},
                "area_sqkm": area_sqkm,
                "perimeter_km": perimeter_km,
                "length_major_axis_km": major_axis_km,
                "orientation_deg": orientation_deg,
                "confidence_score": confidence,
                "is_lookalike": is_lookalike,
                "lookalike_reason": reason,
                "estimated_volume_m3": vol_m3
            }
            results.append(slick_dict)

        return results
