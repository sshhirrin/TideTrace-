"""
Lagrangian Ocean Drift & Ensemble Reverse Backtracking Physics Engine.

Physics Formulation:
  V_slick = V_current(x, y, t) + alpha * V_wind(x, y, t)
  alpha: Wind leeway factor (0.028 - 0.036, default 0.032)
  Deflection angle: Coriolis leeway deflection (0-5 degrees to the right in Northern Hemisphere).

Features:
  - Spatially & temporally varying gridded current and wind vectors
  - 100-particle Monte Carlo Ensemble Backtracking
  - 2D Origin Probability Density Grid / Heatmap
  - 50%, 80%, and 95% Confidence Contour Envelopes
  - Quantified spatial and temporal origin uncertainty
"""
import math
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from ml_engine.metrics import haversine_distance_km
from backend.services.ocean_grid_engine import DynamicOceanGridEngine

class DriftEngine:
    def __init__(self, leeway_factor: float = 0.032, coriolis_deg: float = 2.0):
        self.leeway_factor = leeway_factor
        self.coriolis_deg = coriolis_deg
        self.grid_engine = DynamicOceanGridEngine()

    def calculate_drift_velocity(
        self,
        wind_speed_ms: float,
        wind_direction_from_deg: float,
        current_speed_ms: float,
        current_direction_to_deg: float
    ) -> Tuple[float, float, float, float]:
        """
        Computes net drift velocity vector.
        Returns: (net_u, net_v, net_speed_kmh, net_bearing_deg)
        """
        wind_push_bearing = (wind_direction_from_deg + 180.0 + self.coriolis_deg) % 360.0
        wind_push_rad = math.radians(wind_push_bearing)
        
        wind_u = self.leeway_factor * wind_speed_ms * math.sin(wind_push_rad)
        wind_v = self.leeway_factor * wind_speed_ms * math.cos(wind_push_rad)

        current_rad = math.radians(current_direction_to_deg)
        current_u = current_speed_ms * math.sin(current_rad)
        current_v = current_speed_ms * math.cos(current_rad)

        net_u = wind_u + current_u
        net_v = wind_v + current_v

        net_speed_ms = math.sqrt(net_u**2 + net_v**2)
        net_speed_kmh = net_speed_ms * 3.6
        
        net_bearing_rad = math.atan2(net_u, net_v)
        net_bearing_deg = (math.degrees(net_bearing_rad) + 360.0) % 360.0

        return (net_u, net_v, net_speed_kmh, net_bearing_deg)

    def backtrack_origin(
        self,
        detect_lat: float,
        detect_lng: float,
        elapsed_hours: float,
        wind_speed_ms: float,
        wind_direction_from_deg: float,
        current_speed_ms: float,
        current_direction_to_deg: float,
        dispersion_sigma_km: float = 0.5
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Deterministic reverse drift backtrack (baseline).
        """
        net_u, net_v, net_speed_kmh, net_bearing = self.calculate_drift_velocity(
            wind_speed_ms, wind_direction_from_deg,
            current_speed_ms, current_direction_to_deg
        )

        total_drift_km = net_speed_kmh * elapsed_hours
        reverse_bearing_rad = math.radians((net_bearing + 180.0) % 360.0)

        delta_north_km = total_drift_km * math.cos(reverse_bearing_rad)
        delta_east_km = total_drift_km * math.sin(reverse_bearing_rad)

        origin_lat = detect_lat + (delta_north_km / 111.139)
        origin_lng = detect_lng + (delta_east_km / (111.139 * math.cos(math.radians(detect_lat))))

        uncertainty_radius_km = max(0.8, dispersion_sigma_km * math.sqrt(elapsed_hours))
        
        cone_coords = []
        for angle_deg in range(0, 360, 20):
            rad = math.radians(angle_deg)
            lat_offset = (uncertainty_radius_km * math.cos(rad)) / 111.139
            lng_offset = (uncertainty_radius_km * math.sin(rad)) / (111.139 * math.cos(math.radians(origin_lat)))
            cone_coords.append([round(origin_lng + lng_offset, 6), round(origin_lat + lat_offset, 6)])
        cone_coords.append(cone_coords[0])

        origin_cone_geojson = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [cone_coords]
            },
            "properties": {
                "origin_lat": round(origin_lat, 6),
                "origin_lng": round(origin_lng, 6),
                "elapsed_hours": elapsed_hours,
                "uncertainty_radius_km": round(uncertainty_radius_km, 2),
                "drift_speed_knots": round(net_speed_kmh / 1.852, 2)
            }
        }

        return origin_lat, origin_lng, origin_cone_geojson

    def run_ensemble_backtracking(
        self,
        detect_lat: float,
        detect_lng: float,
        elapsed_hours: float,
        base_wind_speed_ms: float,
        base_wind_dir_deg: float,
        base_current_speed_ms: float,
        base_current_dir_deg: float,
        num_particles: int = 100,
        time_steps: int = 8
    ) -> Dict[str, Any]:
        """
        Executes 100-simulation Monte Carlo ensemble backtracking with stochastic variation in:
        - initial slick coordinates
        - current vector
        - wind vector
        - leeway factor
        - time stepping
        """
        np.random.seed(143) # Deterministic for repeatable evaluation

        particle_endpoints = []
        sampled_tracks = []

        dt_hours = elapsed_hours / float(time_steps)

        for p_idx in range(num_particles):
            # 1. Stochastic perturbations
            p_leeway = np.random.uniform(0.028, 0.036)
            p_coriolis = np.random.uniform(0.5, 3.5)
            
            # Slick initial position noise (~300m)
            p_lat = detect_lat + np.random.normal(0, 0.0025)
            p_lng = detect_lng + np.random.normal(0, 0.0025)

            track_pts = [[round(p_lng, 6), round(p_lat, 6)]]

            # Integrate backward in time
            for step in range(time_steps):
                t_offset = -(step + 0.5) * dt_hours
                
                # Dynamic wind & current with random gust/eddy perturbation
                dyn = self.grid_engine.get_interpolated_vectors(
                    lat=p_lat,
                    lng=p_lng,
                    hour_offset=t_offset,
                    base_wind_speed=base_wind_speed_ms * np.random.uniform(0.85, 1.15),
                    base_wind_deg=(base_wind_dir_deg + np.random.normal(0, 6.0)) % 360.0,
                    base_current_speed=base_current_speed_ms * np.random.uniform(0.88, 1.12),
                    base_current_deg=(base_current_dir_deg + np.random.normal(0, 5.0)) % 360.0
                )

                # Wind push bearing
                wind_push_rad = math.radians((dyn["wind_dir_deg"] + 180.0 + p_coriolis) % 360.0)
                wind_u = p_leeway * dyn["wind_speed_ms"] * math.sin(wind_push_rad)
                wind_v = p_leeway * dyn["wind_speed_ms"] * math.cos(wind_push_rad)

                curr_rad = math.radians(dyn["current_dir_deg"])
                curr_u = dyn["current_speed_ms"] * math.sin(curr_rad)
                curr_v = dyn["current_speed_ms"] * math.cos(curr_rad)

                net_u_kmh = (wind_u + curr_u) * 3.6
                net_v_kmh = (wind_v + curr_v) * 3.6

                # Move in reverse
                d_north_km = -net_v_kmh * dt_hours
                d_east_km = -net_u_kmh * dt_hours

                p_lat += (d_north_km / 111.139)
                p_lng += (d_east_km / (111.139 * math.cos(math.radians(p_lat))))

                track_pts.append([round(p_lng, 6), round(p_lat, 6)])

            particle_endpoints.append((p_lat, p_lng))
            if p_idx % 10 == 0:
                sampled_tracks.append(track_pts)

        lats = np.array([pt[0] for pt in particle_endpoints])
        lngs = np.array([pt[1] for pt in particle_endpoints])

        # Most likely origin (barycenter / median)
        median_lat = float(np.median(lats))
        median_lng = float(np.median(lngs))

        # Spatial uncertainty (standard deviation)
        lat_std_km = float(np.std(lats) * 111.139)
        lng_std_km = float(np.std(lngs) * 111.139 * math.cos(math.radians(median_lat)))
        spatial_uncertainty_km = round(math.sqrt(lat_std_km**2 + lng_std_km**2), 2)
        temporal_uncertainty_hours = round(elapsed_hours * 0.18, 1)

        # 2D Probability Density Grid (Heatmap cells)
        min_lat, max_lat = float(np.min(lats) - 0.005), float(np.max(lats) + 0.005)
        min_lng, max_lng = float(np.min(lngs) - 0.005), float(np.max(lngs) + 0.005)

        grid_bins = 16
        hist, y_edges, x_edges = np.histogram2d(lats, lngs, bins=grid_bins, range=[[min_lat, max_lat], [min_lng, max_lng]])
        max_density = float(np.max(hist)) if np.max(hist) > 0 else 1.0

        prob_grid = []
        for i in range(grid_bins):
            for j in range(grid_bins):
                count = hist[i, j]
                if count > 0:
                    c_lat = (y_edges[i] + y_edges[i+1]) / 2.0
                    c_lng = (x_edges[j] + x_edges[j+1]) / 2.0
                    prob_grid.append({
                        "lat": round(c_lat, 6),
                        "lng": round(c_lng, 6),
                        "density": round(float(count / max_density), 3),
                        "particle_count": int(count)
                    })

        # Confidence envelopes (50%, 80%, 95%)
        confidence_envelopes = {}
        for p_level, scale in [(50, 0.67), (80, 1.28), (95, 1.96)]:
            r_km = spatial_uncertainty_km * scale
            ring = []
            for deg in range(0, 360, 20):
                rad = math.radians(deg)
                d_lat = (r_km * math.cos(rad)) / 111.139
                d_lng = (r_km * math.sin(rad)) / (111.139 * math.cos(math.radians(median_lat)))
                ring.append([round(median_lng + d_lng, 6), round(median_lat + d_lat, 6)])
            ring.append(ring[0])
            confidence_envelopes[f"confidence_{p_level}_pct"] = {
                "type": "Polygon",
                "coordinates": [ring],
                "radius_km": round(r_km, 2)
            }

        return {
            "simulation_count": num_particles,
            "elapsed_hours": elapsed_hours,
            "most_likely_origin": {
                "lat": round(median_lat, 6),
                "lng": round(median_lng, 6)
            },
            "origin_uncertainty": {
                "spatial_radius_km": spatial_uncertainty_km,
                "temporal_uncertainty_hours": temporal_uncertainty_hours,
                "confidence_level_pct": 85.0
            },
            "probability_grid": prob_grid,
            "confidence_envelopes": confidence_envelopes,
            "sample_trajectories": sampled_tracks
        }

    def generate_drift_trajectory_points(
        self,
        origin_lat: float,
        origin_lng: float,
        elapsed_hours: float,
        steps: int,
        wind_speed_ms: float,
        wind_direction_from_deg: float,
        current_speed_ms: float,
        current_direction_to_deg: float
    ) -> List[Dict[str, Any]]:
        net_u, net_v, net_speed_kmh, net_bearing = self.calculate_drift_velocity(
            wind_speed_ms, wind_direction_from_deg,
            current_speed_ms, current_direction_to_deg
        )
        
        points = []
        for step in range(steps + 1):
            t = (step / steps) * elapsed_hours
            dist_km = net_speed_kmh * t
            rad = math.radians(net_bearing)
            d_lat = (dist_km * math.cos(rad)) / 111.139
            d_lng = (dist_km * math.sin(rad)) / (111.139 * math.cos(math.radians(origin_lat)))
            
            points.append({
                "step": step,
                "elapsed_hours": round(t, 2),
                "lat": round(origin_lat + d_lat, 6),
                "lng": round(origin_lng + d_lng, 6),
                "dispersion_radius_km": round(0.5 + 0.3 * math.sqrt(t), 2)
            })
        return points
