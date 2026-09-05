"""
Geospatial and geometric metrics calculation for detected SAR oil slicks.
"""
import math
import numpy as np
from typing import List, Tuple, Dict, Any

def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance between two geographic coordinates in kilometers."""
    R = 6371.0  # Earth radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2.0) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

def calculate_polygon_centroid(coords: List[List[float]]) -> Tuple[float, float]:
    """
    Computes geographic centroid of a polygon.
    coords: list of [lng, lat] pairs (GeoJSON standard)
    Returns: (lat, lng)
    """
    if not coords:
        return (0.0, 0.0)
    
    # Exclude duplicate closing point if present
    pts = coords[:-1] if len(coords) > 1 and coords[0] == coords[-1] else coords
    if not pts:
        pts = coords
        
    avg_lng = sum(p[0] for p in pts) / len(pts)
    avg_lat = sum(p[1] for p in pts) / len(pts)
    return (avg_lat, avg_lng)

def calculate_polygon_area_sqkm(coords: List[List[float]]) -> float:
    """
    Calculates geodesic polygon surface area in square kilometers
    using the spherical Shoelace formula.
    """
    if len(coords) < 3:
        return 0.0
    
    R = 6371.0  # km
    total = 0.0
    pts = coords[:-1] if coords[0] == coords[-1] else coords
    n = len(pts)
    
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        lat1, lon1 = math.radians(p1[1]), math.radians(p1[0])
        lat2, lon2 = math.radians(p2[1]), math.radians(p2[0])
        total += (lon2 - lon1) * (2.0 + math.sin(lat1) + math.sin(lat2))
        
    area = abs(total * (R ** 2) / 2.0)
    return max(round(area, 4), 0.01)

def calculate_polygon_perimeter_km(coords: List[List[float]]) -> float:
    """Calculates the perimeter of a polygon in km."""
    perimeter = 0.0
    for i in range(len(coords) - 1):
        p1, p2 = coords[i], coords[i + 1]
        perimeter += haversine_distance_km(p1[1], p1[0], p2[1], p2[0])
    return round(perimeter, 3)

def calculate_major_axis_and_orientation(coords: List[List[float]]) -> Tuple[float, float]:
    """
    Computes the major elongation axis length (km) and orientation angle (degrees relative to North)
    using Principal Component Analysis (PCA) on coordinates.
    """
    if len(coords) < 4:
        return 1.0, 0.0
    
    pts = np.array([[p[0], p[1]] for p in coords])
    # Center coordinates
    mean = np.mean(pts, axis=0)
    centered = pts - mean
    
    # Covariance matrix
    cov = np.cov(centered, rowvar=False)
    if np.all(np.isfinite(cov)):
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Principal direction
        primary_vec = eigvecs[:, -1]
        # Angle relative to North (bearing 0-360)
        angle_rad = math.atan2(primary_vec[0], primary_vec[1])
        bearing_deg = (math.degrees(angle_rad) + 360) % 180  # Symmetry across 180
    else:
        bearing_deg = 45.0
        
    # Approximate length by bounding box projected onto principal axis
    centroid_lat, centroid_lng = calculate_polygon_centroid(coords)
    max_dist = 0.0
    for p in coords:
        d = haversine_distance_km(centroid_lat, centroid_lng, p[1], p[0])
        if d > max_dist:
            max_dist = d
            
    major_axis_km = round(max_dist * 2.0, 3)
    return major_axis_km, round(bearing_deg, 1)

def estimate_oil_volume_m3(area_sqkm: float, thickness_microns: float = 2.0) -> float:
    """
    Estimates spill volume in cubic meters based on surface area and standard Bonn Agreement
    thickness classification for metallic/rainbow sheen to continuous dark oil.
    1 km^2 = 1,000,000 m^2
    Volume (m^3) = Area (m^2) * (thickness in microns * 1e-6)
    """
    area_m2 = area_sqkm * 1_000_000.0
    vol_m3 = area_m2 * (thickness_microns * 1e-6)
    return round(vol_m3, 2)
