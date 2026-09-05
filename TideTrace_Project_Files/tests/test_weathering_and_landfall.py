"""
Test Suite for Dynamic Weathering, Forward Landfall ETA, and Ocean Grid Engine.
"""
from backend.services.ocean_grid_engine import DynamicOceanGridEngine
from backend.services.weathering_engine import OilWeatheringEngine
from backend.services.landfall_predictor import CoastalLandfallPredictor

def test_dynamic_ocean_grid_engine():
    engine = DynamicOceanGridEngine()
    
    # Test vector interpolation at hour 0 vs hour 6 (tidal oscillation)
    v0 = engine.get_interpolated_vectors(18.85, 72.40, hour_offset=0.0, base_wind_speed=6.0, base_wind_deg=225.0, base_current_speed=0.5, base_current_deg=45.0)
    v6 = engine.get_interpolated_vectors(18.85, 72.40, hour_offset=6.0, base_wind_speed=6.0, base_wind_deg=225.0, base_current_speed=0.5, base_current_deg=45.0)
    
    assert "wind_speed_ms" in v0 and "current_speed_ms" in v0
    assert "net_u_ms" in v0 and "net_v_ms" in v0
    # Tidal current should oscillate
    assert v0["current_speed_ms"] != v6["current_speed_ms"] or v0["current_dir_deg"] != v6["current_dir_deg"]
    print("✓ Dynamic spatio-temporal ocean grid engine verified.")

def test_petroleum_weathering_engine():
    engine = OilWeatheringEngine()
    
    # 24-hour weathering of 15 m³ crude oil in 6.0 m/s wind
    state24 = engine.compute_weathering_state(elapsed_hours=24.0, initial_volume_m3=15.0, wind_speed_ms=6.0)
    
    # In 24h, 20-50% should evaporate
    assert 20.0 <= state24["evaporated_pct"] <= 55.0
    # Water content in emulsion should be between 40-75%
    assert 40.0 <= state24["emulsified_water_pct"] <= 78.0
    # Total volume of mousse should increase due to water uptake
    assert state24["current_emulsified_volume_m3"] > 0
    # Viscosity should be significantly higher than initial 25 cP
    assert state24["dynamic_viscosity_cp"] > 500.0
    
    # 72-hour timeline check
    curve = engine.generate_72h_weathering_curve(15.0, 6.0)
    assert len(curve) == 12
    assert curve[-1]["evaporated_pct"] >= state24["evaporated_pct"]
    print("✓ Petroleum weathering, evaporation, emulsification, and Fay spreading verified.")

def test_forward_landfall_prediction():
    predictor = CoastalLandfallPredictor()
    
    # Run 72-hour forward simulation starting offshore Mumbai
    forecast = predictor.simulate_forward_drift_72h(
        start_lat=18.855,
        start_lng=72.410,
        initial_volume_m3=14.5,
        base_wind_speed=6.2,
        base_wind_deg=225.0,
        base_current_speed=0.45,
        base_current_deg=45.0
    )
    
    assert len(forecast["trajectory_72h"]) == 72
    assert "threat_level" in forecast
    assert forecast["landfall_impact"] is not None
    assert "eta_hours" in forecast["landfall_impact"]
    assert "impact_timestamp" in forecast["landfall_impact"]
    assert forecast["containment_plan"]["containment_booms_recommended_m"] >= 600
    
    impact = forecast["landfall_impact"]
    print(f"✓ Forward Landfall Predictor: Impact with '{impact['target_name']}' in {impact['eta_hours']} hours.")

if __name__ == "__main__":
    test_dynamic_ocean_grid_engine()
    test_petroleum_weathering_engine()
    test_forward_landfall_prediction()
    print("\n🎉 ALL OCEAN PHYSICS, WEATHERING & LANDFALL TESTS PASSED!")
