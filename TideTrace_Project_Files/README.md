# TideTrace
### Oil Spill Detection, Drift Analysis & Vessel Attribution Platform
**NTRO Problem Statement #26143** // Smart India Hackathon 2026  

---

## 🌊 Overview
**TideTrace** is an operational platform combining Sentinel-1 Synthetic Aperture Radar (SAR) remote sensing, PyTorch deep learning segmentation, 100-particle Monte Carlo Lagrangian reverse drift backtracking, 2D CA-CFAR ship detection, and 7-factor Multi-Criteria Decision Analysis (MCDA) vessel attribution.

---

## 🚀 11 Unified Operational Pages

1. **Command Center:** Real-time operational overview with interactive Leaflet map, active incidents, suspicious vessels, dark vessel alerts, and metocean conditions.
2. **Incident Analysis:** Deep incident investigation with 9 switchable GIS layers (SAR raster, oil segmentation, candidate dark spots, look-alike detections, AIS tracks, probability cone, drift vectors, risk zones).
3. **Dark Vessel:** Real-time vessel detection & correlation monitor with a live event feed, simulation playback controls (1x, 2x, 5x, 10x speed), SAR vs AIS cross-match matrix, and multi-factor dark vessel confidence scoring.
4. **SAR Analysis:** Interactive satellite raster viewer with GeoTIFF upload, Zoom/Pan/Brightness/Contrast/Opacity adjustments, and 8-class look-alike filtering.
5. **Vessel Intelligence:** Searchable vessel catalog with historical voyage tracks, anomaly scores, and static/dynamic message telemetry.
6. **Drift & Landfall:** 24h, 48h, and 72h forward trajectory forecasts with coastline intersection ETA intervals and Environmental Sensitivity Index (ESI) scoring.
7. **Attribution:** Explainable 7-factor MCDA attribution scoring with non-accusatory ranking (`PRIMARY SUSPECT`, `POTENTIAL CONTRIBUTOR`, `CANDIDATE`, `CLEARED`).
8. **Incident Replay:** Interactive historical timeline scrubber (T-72h to T+72h) with Play/Pause/Speed controls.
9. **Evidence Dossier:** Decision-support forensic report with SHA-256 digital sealing and legal provenance.
10. **ML Lab:** Holdout test split benchmarks for ResNet34 U-Net, U-Net++, SegFormer, confusion matrix, and training console.
11. **System Architecture:** End-to-end data flow and service topology diagrams.

---

## 🛠️ Quickstart & Server Launch

```powershell
# 1. Activate environment & start server
python start.py

# 2. Open dashboard in your browser
http://localhost:8000
```
