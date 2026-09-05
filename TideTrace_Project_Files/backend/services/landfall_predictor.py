"""
Forward Lagrangian Drift Simulation, Coastal Landfall ETA & Environmental Risk Predictor.

Projects the future trajectory of detected oil slicks over a 72-hour horizon (+12h, +24h, +48h, +72h),
computes shoreline intersection and ETA confidence intervals (e.g. 14.2h [11.8 - 17.1h], 78% confidence),
and evaluates environmental, coastal, and infrastructure risk across sensitive marine assets.
"""
import math
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
from ml_engine.metrics import haversine_distance_km
from backend.services.ocean_grid_engine import DynamicOceanGridEngine
from backend.services.weathering_engine import OilWeatheringEngine

# Comprehensive Sensitive Coastal Assets & Critical Habitats
COASTAL_TARGETS = [
    # Arabian Sea / Mumbai Region
    {"name": "Manori & Gorai Coastal Wetlands & Marine Sanctuary", "lat": 19.240, "lng": 72.780, "type": "Mangrove Wetland", "vulnerability": "CRITICAL", "region": "Arabian Sea", "category": "ECOLOGICAL"},
    {"name": "Vasai Creek Fishery & Mangrove Ecosystem", "lat": 19.330, "lng": 72.810, "type": "Estuarine Fishery", "vulnerability": "CRITICAL", "region": "Arabian Sea", "category": "FISHERY"},
    {"name": "Alibag Coastal Reefs & Tourism Beaches", "lat": 18.640, "lng": 72.870, "type": "Tourist Beach & Coral Habitat", "vulnerability": "HIGH", "region": "Arabian Sea", "category": "TOURISM"},
    {"name": "JNPT Port International Navigation Channel", "lat": 18.940, "lng": 72.880, "type": "Major Port Channel", "vulnerability": "CRITICAL", "region": "Arabian Sea", "category": "INFRASTRUCTURE"},
    {"name": "Elephanta Island UNESCO Protected Marine Sanctuary", "lat": 18.960, "lng": 72.930, "type": "Marine Protected Area", "vulnerability": "CRITICAL", "region": "Arabian Sea", "category": "MPA"},
    {"name": "Thane Creek Flamingo & Mangrove Sanctuary", "lat": 19.040, "lng": 72.980, "type": "Mangrove Wetland Ecosystem", "vulnerability": "CRITICAL", "region": "Arabian Sea", "category": "ECOLOGICAL"},
    {"name": "Mumbai Offshore High Shipping Corridor", "lat": 18.880, "lng": 72.500, "type": "Traffic Separation Scheme", "vulnerability": "MEDIUM", "region": "Arabian Sea", "category": "SHIPPING_LANE"},

    # Singapore Strait
    {"name": "Sisters' Islands Marine Park", "lat": 1.215, "lng": 103.835, "type": "Coral Reef Sanctuary", "vulnerability": "CRITICAL", "region": "Singapore Strait", "category": "MPA"},
    {"name": "Sentosa Island Recreational Beaches", "lat": 1.250, "lng": 103.820, "type": "Tourism Beach", "vulnerability": "HIGH", "region": "Singapore Strait", "category": "TOURISM"},
    {"name": "Jurong Island Petrochemical Port Terminal", "lat": 1.265, "lng": 103.710, "type": "Petrochemical Port", "vulnerability": "CRITICAL", "region": "Singapore Strait", "category": "INFRASTRUCTURE"},
    {"name": "Batam Fast Ferry Coastal Corridor", "lat": 1.160, "lng": 103.950, "type": "Coastal Transit Lane", "vulnerability": "MEDIUM", "region": "Singapore Strait", "category": "SHIPPING_LANE"},

    # Bay of Bengal
    {"name": "Pulicat Lagoon Estuary & Mangroves", "lat": 13.420, "lng": 80.320, "type": "Estuarine Biosphere", "vulnerability": "CRITICAL", "region": "Bay of Bengal", "category": "ECOLOGICAL"},
    {"name": "Marina Coastal Biosphere & Artisanal Fishery", "lat": 13.040, "lng": 80.280, "type": "Public Coastline & Fishery", "vulnerability": "HIGH", "region": "Bay of Bengal", "category": "FISHERY"},
    {"name": "Chennai Port Approach Fairway", "lat": 13.100, "lng": 80.340, "type": "Major Port Channel", "vulnerability": "CRITICAL", "region": "Bay of Bengal", "category": "INFRASTRUCTURE"}
]

class CoastalLandfallPredictor:
    def __init__(self):
        self.grid_engine = DynamicOceanGridEngine()
        self.weathering_engine = OilWeatheringEngine()

    def simulate_forward_drift_72h(
        self,
        start_lat: float,
        start_lng: float,
        initial_volume_m3: float,
        base_wind_speed: float,
        base_wind_deg: float,
        base_current_speed: float,
        base_current_deg: float,
        start_time_iso: str = "2026-09-01T06:00:00Z"
    ) -> Dict[str, Any]:
        dt_start = datetime.fromisoformat(start_time_iso.replace("Z", "+00:00"))
        
        trajectory_points = []
        cur_lat, cur_lng = start_lat, start_lng

        landfall_hit = None
        min_shore_dist_km = float("inf")
        closest_target = None
        affected_assets = []

        key_forecast_horizons = {12: None, 24: None, 48: None, 72: None}

        for h in range(1, 73):
            dyn = self.grid_engine.get_interpolated_vectors(
                lat=cur_lat,
                lng=cur_lng,
                hour_offset=float(h),
                base_wind_speed=base_wind_speed,
                base_wind_deg=base_wind_deg,
                base_current_speed=base_current_speed,
                base_current_deg=base_current_deg
            )

            net_u_kmh = dyn["net_u_ms"] * 3.6
            net_v_kmh = dyn["net_v_ms"] * 3.6

            d_north_km = net_v_kmh * 1.0
            d_east_km = net_u_kmh * 1.0

            cur_lat += (d_north_km / 111.139)
            cur_lng += (d_east_km / (111.139 * math.cos(math.radians(cur_lat))))

            pt_time = (dt_start + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ")
            cone_radius_km = round(0.4 + 0.15 * math.sqrt(h), 2)

            pt_data = {
                "hour": h,
                "timestamp": pt_time,
                "lat": round(cur_lat, 6),
                "lng": round(cur_lng, 6),
                "uncertainty_radius_km": cone_radius_km,
                "wind_speed_ms": dyn["wind_speed_ms"],
                "current_speed_ms": dyn["current_speed_ms"]
            }
            trajectory_points.append(pt_data)

            if h in key_forecast_horizons:
                key_forecast_horizons[h] = {
                    "horizon_label": f"+{h}h",
                    "timestamp": pt_time,
                    "lat": round(cur_lat, 6),
                    "lng": round(cur_lng, 6),
                    "uncertainty_corridor_radius_km": cone_radius_km,
                    "estimated_slick_area_sqkm": round(0.08 * (initial_volume_m3 ** 0.5) * (1.0 + 0.08 * h), 2)
                }

            # Check proximity to coastal targets
            for target in COASTAL_TARGETS:
                dist = haversine_distance_km(cur_lat, cur_lng, target["lat"], target["lng"])
                if dist < min_shore_dist_km:
                    min_shore_dist_km = round(dist, 2)
                    closest_target = target

                if dist <= 3.5 and landfall_hit is None:
                    # Calculate ETA confidence interval based on current/wind variability (±18%)
                    eta_nominal = float(h)
                    eta_low = max(0.5, round(eta_nominal * 0.82, 1))
                    eta_high = round(eta_nominal * 1.20, 1)
                    confidence_pct = max(60, min(92, int(95 - 0.4 * h)))

                    landfall_hit = {
                        "target_name": target["name"],
                        "target_type": target["type"],
                        "vulnerability": target["vulnerability"],
                        "category": target.get("category", "ECOLOGICAL"),
                        "eta_hours": eta_nominal,
                        "eta_range_hours": [eta_low, eta_high],
                        "confidence_pct": confidence_pct,
                        "impact_timestamp": pt_time,
                        "impact_lat": round(cur_lat, 6),
                        "impact_lng": round(cur_lng, 6),
                        "distance_km": round(dist, 2),
                        "confidence_statement": f"Most likely landfall: {eta_nominal:.1f}h (Range: {eta_low}–{eta_high}h, Confidence: {confidence_pct}%)"
                    }

                if dist <= 8.0 and target["name"] not in [a["name"] for a in affected_assets]:
                    affected_assets.append({
                        "name": target["name"],
                        "type": target["type"],
                        "category": target.get("category", "ECOLOGICAL"),
                        "vulnerability": target["vulnerability"],
                        "approach_distance_km": round(dist, 2)
                    })

        # Fallback closest approach if no direct intersect <= 3.5 km
        if not landfall_hit and closest_target and min_shore_dist_km < 20.0:
            est_hours = round(min_shore_dist_km / 1.2, 1)
            eta_low = max(0.5, round(est_hours * 0.82, 1))
            eta_high = round(est_hours * 1.20, 1)
            conf_pct = max(55, min(80, int(88 - 0.5 * est_hours)))
            landfall_hit = {
                "target_name": closest_target["name"],
                "target_type": closest_target["type"],
                "vulnerability": closest_target["vulnerability"],
                "category": closest_target.get("category", "ECOLOGICAL"),
                "eta_hours": est_hours,
                "eta_range_hours": [eta_low, eta_high],
                "confidence_pct": conf_pct,
                "impact_timestamp": (dt_start + timedelta(hours=est_hours)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "impact_lat": closest_target["lat"],
                "impact_lng": closest_target["lng"],
                "distance_km": min_shore_dist_km,
                "is_closest_approach": True,
                "confidence_statement": f"Closest coastal approach: {est_hours:.1f}h (Range: {eta_low}–{eta_high}h, Confidence: {conf_pct}%)"
            }

        # Calculate modular multi-factor risk scores
        env_risk = 85.0 if any(a["category"] in ["ECOLOGICAL", "MPA"] for a in affected_assets) else 45.0
        coastal_risk = 90.0 if (landfall_hit and landfall_hit["eta_hours"] <= 24) else (65.0 if landfall_hit else 20.0)
        infra_risk = 80.0 if any(a["category"] == "INFRASTRUCTURE" for a in affected_assets) else 30.0

        if landfall_hit:
            threat_level = "CRITICAL_SHORELINE_THREAT" if landfall_hit["eta_hours"] <= 24 else "HIGH_RISK_APPROACH"
        else:
            threat_level = "OFFSHORE_SAFE_CORRIDOR"

        containment_plan = {
            "containment_booms_recommended_m": 1200 if initial_volume_m3 > 10 else 600,
            "skimmer_vessels_needed": 2 if initial_volume_m3 > 10 else 1,
            "chemical_dispersant_status": "PROHIBITED (Within 10km Coastal Buffer Zone)" if min_shore_dist_km < 10.0 else "AUTHORIZED FOR DEEP SEA APPLICATION",
            "priority_defense_site": closest_target["name"] if closest_target else "Open Sea Corridor",
            "suggested_barrier_coords": [round(start_lat + 0.04, 4), round(start_lng + 0.04, 4)]
        }

        return {
            "threat_level": threat_level,
            "landfall_impact": landfall_hit,
            "min_shoreline_distance_km": min_shore_dist_km,
            "risk_assessment": {
                "environmental_risk_score": env_risk,
                "coastal_impact_risk_score": coastal_risk,
                "infrastructure_risk_score": infra_risk,
                "composite_threat_index": round((env_risk * 0.4 + coastal_risk * 0.4 + infra_risk * 0.2), 1)
            },
            "affected_coastal_assets": affected_assets,
            "forecast_horizons": key_forecast_horizons,
            "containment_plan": containment_plan,
            "trajectory_72h": trajectory_points
        }
