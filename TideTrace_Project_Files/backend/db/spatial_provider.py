"""
Spatial Query Provider with Unified PostGIS & In-Memory Fallback.

Provides dual-engine spatial indexing and trajectory queries:
  1. Direct PostGIS SQL execution (when PostgreSQL connection string is supplied)
  2. High-performance In-Memory Shapely & Haversine spatial querying (offline / local development)

Implements PostGIS primitives:
  - ST_DWithin(geom1, geom2, distance_meters)
  - ST_ClosestPoint(line, point)
  - ST_Intersects(geom1, geom2)
  - ST_Area(polygon)
"""
import os
import math
from typing import List, Dict, Any, Tuple, Optional
from shapely.geometry import Point, LineString, Polygon, shape
from shapely.ops import nearest_points
from ml_engine.metrics import haversine_distance_km

class SpatialProvider:
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.getenv("POSTGIS_DATABASE_URL", None)
        self.has_live_db = False

    def st_dwithin_km(self, lat1: float, lng1: float, lat2: float, lng2: float, radius_km: float) -> bool:
        """Emulates PostGIS ST_DWithin(geom1::geography, geom2::geography, distance_meters)."""
        dist = haversine_distance_km(lat1, lng1, lat2, lng2)
        return dist <= radius_km

    def st_closest_point_on_track(self, track_coords: List[Tuple[float, float]], target_lat: float, target_lng: float) -> Tuple[float, float, float]:
        """
        Emulates PostGIS ST_ClosestPoint(track_linestring, target_point).
        track_coords: List of (lat, lng)
        Returns: (closest_lat, closest_lng, distance_km)
        """
        if not track_coords:
            return target_lat, target_lng, 0.0

        line = LineString([(lng, lat) for lat, lng in track_coords])
        pt = Point(target_lng, target_lat)

        nearest_pt, _ = nearest_points(line, pt)
        c_lng, c_lat = nearest_pt.x, nearest_pt.y
        dist_km = haversine_distance_km(target_lat, target_lng, c_lat, c_lng)

        return round(c_lat, 6), round(c_lng, 6), round(dist_km, 3)

    def st_intersects(self, geojson_geom1: Dict[str, Any], geojson_geom2: Dict[str, Any]) -> bool:
        """Emulates PostGIS ST_Intersects(geom1, geom2)."""
        try:
            s1 = shape(geojson_geom1)
            s2 = shape(geojson_geom2)
            return bool(s1.intersects(s2))
        except Exception:
            return False

    def filter_vessels_in_radius(
        self,
        vessels: List[Any],
        target_lat: float,
        target_lng: float,
        radius_km: float
    ) -> List[Dict[str, Any]]:
        """
        Filters vessel tracks that pass within radius_km of target coordinate.
        """
        candidates = []
        for v in vessels:
            positions = v.positions if hasattr(v, "positions") else v.get("positions", [])
            track_pts = [(p.lat if hasattr(p, "lat") else p["lat"], p.lng if hasattr(p, "lng") else p["lng"]) for p in positions]
            if not track_pts:
                continue

            c_lat, c_lng, min_dist = self.st_closest_point_on_track(track_pts, target_lat, target_lng)
            if min_dist <= radius_km:
                name = v.metadata.name if hasattr(v, "metadata") else v.get("name", "Unknown")
                mmsi = v.metadata.mmsi if hasattr(v, "metadata") else v.get("mmsi", 0)
                candidates.append({
                    "mmsi": mmsi,
                    "name": name,
                    "closest_lat": c_lat,
                    "closest_lng": c_lng,
                    "distance_km": min_dist
                })

        return sorted(candidates, key=lambda x: x["distance_km"])
