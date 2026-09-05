"""
Petroleum Weathering, Emulsification, and Fay's Spreading Physics Engine.

Models the multi-phase physico-chemical degradation of marine oil spills:
1. Evaporation (Stiver-Friesen / ADIOS model)
2. Water-in-Oil Emulsification ("Chocolate Mousse" formation)
3. Viscosity Evolution (Mooney's equation)
4. Fay's Three-Regime Surface Spreading
5. Mass Balance Distribution over 72-Hour Horizon
"""
import math
import numpy as np
from typing import Dict, Any, List

class OilWeatheringEngine:
    def __init__(self):
        pass

    def compute_weathering_state(
        self,
        elapsed_hours: float,
        initial_volume_m3: float,
        wind_speed_ms: float,
        surface_temp_c: float = 28.0,
        oil_type: str = "Crude Oil (Medium Arabian)"
    ) -> Dict[str, Any]:
        """
        Computes the complete weathering state of petroleum at elapsed_hours after discharge.
        """
        t_sec = max(60.0, elapsed_hours * 3600.0)
        temp_k = surface_temp_c + 273.15

        # 1. Evaporation (Mackay / Stiver-Friesen Petroleum Formulation)
        # Mass transfer rate K (m/s)
        k_mass_transfer = 1.2e-6 * (wind_speed_ms ** 0.78)
        initial_thickness_m = 0.002 # 2 mm average initial slick thickness
        theta = (k_mass_transfer * t_sec) / initial_thickness_m
        
        # Evaporation fraction for medium crude: ~30-40% at 24h, ~50% at 72h
        f_evap = min(0.60, 0.048 * math.log(1.0 + 120.0 * theta) * (temp_k / 298.15))
        f_evap_pct = round(f_evap * 100.0, 1)

        # 2. Water-in-Oil Emulsification (Mousse Formation)
        # Higher wind accelerates water uptake
        y_max = 0.78  # Maximum water content (78%)
        k_em = 1.2e-6 * ((1.0 + wind_speed_ms) ** 2)
        y_water = y_max * (1.0 - math.exp(-k_em * t_sec))
        y_water_pct = round(y_water * 100.0, 1)

        # 3. Total Emulsified Volume (Swelling due to water incorporation)
        remaining_oil_vol = initial_volume_m3 * (1.0 - f_evap)
        emulsified_vol_m3 = round(remaining_oil_vol / max(0.05, (1.0 - y_water)), 2)

        # 4. Dynamic Viscosity Increase (Mooney's Equation)
        mu_0 = 25.0  # Initial fresh oil viscosity (cP)
        c_evap = 8.5 # Viscosity exponent for evaporation
        viscosity_evap = mu_0 * math.exp(c_evap * f_evap)
        # Emulsion multiplier
        viscosity_emulsion = math.exp((2.5 * y_water) / (1.0 - 0.65 * y_water))
        current_viscosity_cp = round(viscosity_evap * viscosity_emulsion, 1)

        # 5. Fay's Three-Regime Spreading Area (km²)
        # Gravity-Viscous spreading dominates intermediate phase (t > 1 hr)
        # A(t) = pi * (Delta * g * V^2 * t^(3/2) / nu_w^(1/2))^(1/3)
        delta = 0.15 # Density difference (rho_w - rho_oil) / rho_w
        g = 9.81
        nu_w = 1.05e-6 # Seawater kinematic viscosity
        
        radius_m = max(50.0, ((delta * g * (initial_volume_m3 ** 2) * (t_sec ** 1.5)) / math.sqrt(nu_w)) ** (1.0 / 6.0))
        # Wind shear spreading elongation multiplier
        slick_area_sqkm = round((math.pi * (radius_m ** 2) * (1.0 + 0.15 * wind_speed_ms)) / 1_000_000.0, 3)

        # 6. Natural Dispersion into water column (Delvigne-Sweeney)
        f_dispersed = min(0.20, 0.005 * (wind_speed_ms ** 1.8) * (elapsed_hours / 24.0))
        f_dispersed_pct = round(f_dispersed * 100.0, 1)

        surface_remaining_pct = round(max(0.0, 100.0 - f_evap_pct - f_dispersed_pct), 1)

        return {
            "elapsed_hours": elapsed_hours,
            "oil_type": oil_type,
            "evaporated_pct": f_evap_pct,
            "emulsified_water_pct": y_water_pct,
            "natural_dispersion_pct": f_dispersed_pct,
            "surface_remaining_pct": surface_remaining_pct,
            "initial_volume_m3": initial_volume_m3,
            "current_emulsified_volume_m3": emulsified_vol_m3,
            "dynamic_viscosity_cp": current_viscosity_cp,
            "viscosity_classification": "Sticky Mousse / Heavy Tar" if current_viscosity_cp > 5000 else ("Viscous Oil" if current_viscosity_cp > 500 else "Fluid Hydrocarbon"),
            "slick_area_sqkm": slick_area_sqkm
        }

    def generate_72h_weathering_curve(
        self,
        initial_volume_m3: float,
        wind_speed_ms: float,
        surface_temp_c: float = 28.0
    ) -> List[Dict[str, Any]]:
        """Generates hourly time-series data for dashboard weathering graphs."""
        timeline = []
        for h in [0.5, 1, 2, 4, 8, 12, 18, 24, 36, 48, 60, 72]:
            state = self.compute_weathering_state(
                elapsed_hours=h,
                initial_volume_m3=initial_volume_m3,
                wind_speed_ms=wind_speed_ms,
                surface_temp_c=surface_temp_c
            )
            timeline.append(state)
        return timeline
