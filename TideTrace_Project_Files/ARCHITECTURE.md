# TideTrace // Architecture & System Design
**Subtitle:** Oil Spill Detection, Drift Analysis & Vessel Attribution Platform  

---

## 1. End-to-End Pipeline
```
Satellite SAR (Sentinel-1)
      |
      v
SAR Preprocessing (5x5 Adaptive Lee Filter)
      |
      v
Oil Spill Detection & Segmentation (ResNet34 U-Net)
      |
      v
3-Class Look-Alike Filtering (Damping Ratio > 4.5 dB)
      |
      +----------------+
      |                |
      v                v
2D CA-CFAR Vessel   100-Particle Reverse Drift
   Detection           Backtracking (RK4)
      |                |
      v                v
AIS Correlation     Forward 72h Landfall Risk
      |
      v
7-Factor MCDA Attribution Engine
      |
      v
Decision-Support Evidence Dossier (SHA-256 Sealed)
```

---

## 2. Component Layout
- **Perception Engine:** PyTorch ResNet34 U-Net, U-Net++, SegFormer, 5x5 Adaptive Lee filter.
- **Hydrodynamic Engine:** 4th-order Runge-Kutta Lagrangian solver, Coriolis deflection, Stokes drift, turbulent dispersion.
- **Dark Vessel Engine:** 2D CA-CFAR detector, 8-point AIS anomaly evaluator.
- **Spatial Engine:** PostGIS 3.4 + Shapely in-memory provider.
