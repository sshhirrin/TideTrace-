"""
Geospatial Coordinate Reference System (CRS) & Vectorization Utilities.
Handles WGS84 (EPSG:4326), UTM projection, affine transformations, and contour polygon extraction.
"""
import rasterio
from rasterio.transform import xy, rowcol
import numpy as np
import cv2
from typing import List, Tuple, Any, Dict

def pixel_to_wgs84(px: int, py: int, transform: Any) -> Tuple[float, float]:
    """Converts pixel coordinate (x, y) to WGS84 (lng, lat)."""
    lng, lat = xy(transform, py, px, offset='center')
    return round(lng, 6), round(lat, 6)

def wgs84_to_pixel(lng: float, lat: float, transform: Any) -> Tuple[int, int]:
    """Converts WGS84 (lng, lat) to raster pixel coordinate (py, px)."""
    row, col = rowcol(transform, lng, lat)
    return int(col), int(row)

def extract_geojson_polygons(binary_mask: np.ndarray, transform: Any, min_area_px: int = 40) -> List[Dict[str, Any]]:
    """
    Vectorizes binary segmentation mask into valid GeoJSON FeatureCollection.
    """
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features = []

    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area_px:
            continue

        epsilon = 0.004 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        coords = []
        for pt in approx:
            px, py = pt[0][0], pt[0][1]
            lng, lat = xy(transform, py, px, offset='center')
            coords.append([round(lng, 6), round(lat, 6)])

        if len(coords) >= 3:
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            features.append({
                "type": "Feature",
                "id": f"SLICK-POLY-{idx+1:03d}",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords]
                },
                "properties": {
                    "pixel_area": float(area),
                    "vertex_count": len(coords)
                }
            })

    return features
