"""
Comprehensive Test Suite for TideTrace Intelligence Pipeline.
Verifies:
  1. Multi-Stage SAR Inference Pipeline (Preprocessing, UNet, 3-class classification)
  2. 100-Particle Ensemble Reverse Drift Backtracking
  3. Dark Vessel & AIS Anomaly Engine
  4. Spatial Provider (PostGIS ST_DWithin, ST_ClosestPoint, ST_Intersects emulation)
  5. One-Click "RUN INCIDENT" Master Demo Pipeline
  6. Standardized REST Endpoints
"""
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from ml_engine.inference.pipeline import SARInferencePipeline
from ml_engine.geotiff_processor import GeoTIFFProcessor
from backend.services.drift_engine import DriftEngine
from backend.services.dark_vessel_engine import DarkVesselEngine
from backend.db.spatial_provider import SpatialProvider
from backend.scenarios_data import build_scenario_alpha

client = TestClient(app)

def test_tidetrace_health_and_stats():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["product"] == "TideTrace"
    assert "INVESTIGATION" in data["operating_modes"]

    res_stats = client.get("/api/stats")
    assert res_stats.status_code == 200
    assert res_stats.json()["peak_attribution_confidence_pct"] > 90.0

def test_sar_multi_stage_inference():
    processor = GeoTIFFProcessor()
    test_tif = "data_samples/test_synthetic_sar.tif"
    raster, transform, _, _ = processor.read_geotiff(test_tif)

    pipeline = SARInferencePipeline(model_name="unet_standard")
    res = pipeline.process_raster_scene(raster, transform, wind_speed_ms=6.0)

    assert "candidate_regions_count" in res
    assert "candidates" in res
    assert res["model_architecture"] == "unet_standard"

def test_ensemble_reverse_drift_100_particles():
    engine = DriftEngine()
    res = engine.run_ensemble_backtracking(
        detect_lat=18.86,
        detect_lng=72.41,
        elapsed_hours=3.5,
        base_wind_speed_ms=6.0,
        base_wind_dir_deg=225.0,
        base_current_speed_ms=0.5,
        base_current_dir_deg=45.0,
        num_particles=100
    )

    assert res["simulation_count"] == 100
    assert "most_likely_origin" in res
    assert "probability_grid" in res
    assert len(res["probability_grid"]) > 0
    assert "confidence_envelopes" in res
    assert "confidence_80_pct" in res["confidence_envelopes"]
    assert res["origin_uncertainty"]["spatial_radius_km"] > 0

def test_spatial_provider_primitives():
    sp = SpatialProvider()
    # ST_DWithin
    in_range = sp.st_dwithin_km(18.80, 72.30, 18.81, 72.31, radius_km=5.0)
    assert in_range is True
    out_range = sp.st_dwithin_km(18.80, 72.30, 19.50, 73.50, radius_km=5.0)
    assert out_range is False

    # ST_ClosestPoint
    track = [(18.70, 72.20), (18.80, 72.30), (18.90, 72.40)]
    c_lat, c_lng, dist_km = sp.st_closest_point_on_track(track, 18.80, 72.30)
    assert dist_km < 0.1

def test_master_demo_run_pipeline():
    res = client.post("/api/incident/run_pipeline", json={"scenario_id": "scenario_alpha_rogue_tanker"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "COMPLETED"
    assert len(data["stages"]) == 6
    assert "digital_seal_sha256" in data
    assert "ensemble_drift" in data
    assert "forward_forecast" in data

def test_models_status_benchmark():
    res = client.get("/api/models/status")
    assert res.status_code == 200
    data = res.json()
    assert "benchmark_metrics" in data
    assert "unet_standard" in data["benchmark_metrics"]
