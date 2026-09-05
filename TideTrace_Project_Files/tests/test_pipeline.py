"""
End-to-End Test Suite for TideTrace Oil Spill & AIS Correlation Pipeline.
"""
from ml_engine.metrics import (
    haversine_distance_km,
    calculate_polygon_centroid,
    calculate_polygon_area_sqkm,
    estimate_oil_volume_m3
)
from ml_engine.lookalike_filter import LookAlikeClassifier
from ml_engine.sar_detector import SAROilSpillDetector
from backend.services.drift_engine import DriftEngine
from backend.services.correlation_engine import AISCorrelationEngine
from backend.scenarios_data import SCENARIOS, build_scenario_alpha, build_scenario_gamma
from backend.services.report_generator import generate_markdown_dossier

def test_haversine_and_metrics():
    # Distance between Mumbai (18.92, 72.83) and Alibag (18.64, 72.87) ≈ 31 km
    dist = haversine_distance_km(18.92, 72.83, 18.64, 72.87)
    assert 28.0 < dist < 35.0

    # Test Shoelace polygon area
    square_coords = [
        [72.0, 18.0],
        [72.1, 18.0],
        [72.1, 18.1],
        [72.0, 18.1],
        [72.0, 18.0]
    ]
    area = calculate_polygon_area_sqkm(square_coords)
    assert area > 100.0  # 0.1 deg x 0.1 deg is approx 11km x 11km ≈ 120 km²

    vol = estimate_oil_volume_m3(area_sqkm=5.0, thickness_microns=2.0)
    assert vol == 10.0  # 5 * 10^6 m² * 2 * 10^-6 m = 10 m³

def test_lookalike_filter():
    clf = LookAlikeClassifier()
    
    # 1. Low wind trigger (< 2.5 m/s)
    is_lookalike, reason, conf = clf.evaluate_slick(
        area_sqkm=4.0, perimeter_km=8.0, major_axis_km=3.0,
        wind_speed_ms=1.5, radar_damping_db=3.0
    )
    assert is_lookalike is True
    assert "Calm Water" in reason
    assert conf < 0.30

    # 2. Confirmed mineral oil spill in prime wind window (6.0 m/s) with high damping
    is_lookalike, reason, conf = clf.evaluate_slick(
        area_sqkm=4.0, perimeter_km=8.0, major_axis_km=3.0,
        wind_speed_ms=6.0, radar_damping_db=9.5, edge_gradient_sharpness=0.85
    )
    assert is_lookalike is False
    assert "Confirmed Mineral Oil" in reason
    assert conf >= 0.85

def test_drift_physics_backtrack():
    engine = DriftEngine(leeway_factor=0.032)
    net_u, net_v, speed_kmh, bearing = engine.calculate_drift_velocity(
        wind_speed_ms=6.0,
        wind_direction_from_deg=225.0, # SW wind pushes NE
        current_speed_ms=0.5,
        current_direction_to_deg=45.0
    )
    assert speed_kmh > 0
    assert 40.0 <= bearing <= 55.0  # Net push towards NE

    origin_lat, origin_lng, origin_cone = engine.backtrack_origin(
        detect_lat=18.86,
        detect_lng=72.41,
        elapsed_hours=3.5,
        wind_speed_ms=6.0,
        wind_direction_from_deg=225.0,
        current_speed_ms=0.5,
        current_direction_to_deg=45.0
    )
    # Origin should be SW of detection point (smaller lat and lng)
    assert origin_lat < 18.86
    assert origin_lng < 72.41
    assert "geometry" in origin_cone

def test_scenario_alpha_correlation():
    sc = build_scenario_alpha()
    assert len(sc.culprits) == 3
    
    # Primary suspect must be MT Ocean Titan
    top_culprit = sc.culprits[0]
    assert top_culprit.vessel_name == "MT Ocean Titan"
    assert top_culprit.mmsi == 419001234
    assert top_culprit.verdict == "PRIMARY SUSPECT"
    assert top_culprit.composite_score >= 85.0
    assert top_culprit.closest_approach_distance_km < 0.1 # Direct hit on origin

    # CMA CGM Mumbai (container ship) must be cleared
    container_culprit = next(c for c in sc.culprits if c.mmsi == 228394000)
    assert container_culprit.verdict == "CLEARED"
    assert container_culprit.composite_score < 45.0

def test_forensic_dossier_generation():
    sc = build_scenario_alpha()
    dossier = generate_markdown_dossier(sc)
    assert "TIDETRACE: FORENSIC INCIDENT DOSSIER" in dossier
    assert "MT Ocean Titan" in dossier
    assert "419001234" in dossier
    assert "PRIMARY SUSPECT" in dossier
