# TideTrace // Project State & Development Audit

**Project:** TideTrace  
**Subtitle:** Oil Spill Detection, Drift Analysis & Vessel Attribution Platform  
**Target:** NTRO Problem Statement #26143  
**Audit Date:** 2026-09-04  

---

## 1. What is Already Implemented
- **Deep Learning Architectures (`ml_engine/models/`):**
  - ResNet34 U-Net (`unet_resnet34.py`) with Hann sliding-window tiling and seamless boundary blending.
  - U-Net++ (`unet_plusplus.py`) with dense nested skip connections.
  - SegFormer (`segformer.py`) with hierarchical Mix-Transformer backbone.
  - Pre-trained PyTorch weights at `ml_engine/checkpoints/best_oil_spill_unet.pth`.
  - Model Registry (`model_registry.py`) factory loader.
- **Preprocessing & Remote Sensing (`ml_engine/preprocessing/`):**
  - Vectorized 5×5 and 7×7 Adaptive Lee filter and Frost filter.
  - Radiometric Sigma-0 (dB) calibration and percentile normalization.
  - Affine CRS transform and WGS84 GeoJSON polygon vectorizer.
- **3-Class Look-Alike Rejection (`ml_engine/lookalike_filter.py`):**
  - Feature extraction: backscatter damping ratio, shape compactness ($4\pi A / P^2$), boundary edge gradient, and wind speed operational regime ($3.0 - 12.0	ext{ m/s}$).
  - Multi-class output: `OIL`, `LOOK-ALIKE`, `UNCERTAIN`.
- **2D CA-CFAR Ship Detection (`ml_engine/cfar_ship_detector.py`):**
  - Cell-Averaging CFAR detector with guard and training rings.
  - Oriented Bounding Boxes (OBBs), estimated ship length/beam, heading vectors, and base64 radar thumbnail extraction.
- **Dark Vessel Intelligence Core (`backend/services/dark_vessel_engine.py`):**
  - 8-point AIS anomaly evaluator (transponder gaps, MMSI validity, SOG vs Doppler deviation, COG vs heading offset, rate of turn jitter, origin proximity, draft drop, TSS compliance).
  - Non-accusatory status classification (`PRIMARY SUSPECT`, `POSSIBLE DARK VESSEL`, `REQUIRES REVIEW`, `NORMAL`).
- **Lagrangian Reverse & Forward Drift Models (`backend/services/`):**
  - 100-particle Monte Carlo ensemble backtracking with wind leeway, Coriolis deflection, Stokes drift, and turbulent diffusion.
  - 2D origin probability density grid with 50%, 80%, and 95% confidence envelopes.
  - Forward 72h landfall prediction with ETA confidence intervals.
  - Mackay physico-chemical petroleum weathering (evaporation, emulsification, dynamic viscosity).
- **PostGIS 3.4 Spatial Architecture (`backend/db/`):**
  - PostgreSQL 16 + PostGIS 3.4 DDL in `backend/db/postgis_schema.sql`.
  - Spatial provider supporting both native PostGIS queries and in-memory Shapely fallback.
- **Decision-Support Reporting (`backend/services/report_generator.py`):**
  - Deterministic SHA-256 digital sealing and legal decision-support disclaimer.
- **Automated Verification:**
  - 18/18 pytest unit and integration tests passing.

---

## 2. What is Partially Implemented
- **Frontend Navigation & Multi-Page Separation (`frontend/index.html`):**
  - Needs a dedicated multi-page layout with a top navigation bar and sidebar allowing seamless switching across all 11 required pages.
  - Needs large interactive Leaflet maps dedicated to each specific page with specialized layer controls (e.g. Incident Analysis map, Dark Vessel real-time map, Vessel Intelligence map, Drift & Landfall map, Replay map).
- **Dark Vessel Real-Time Event Simulation:**
  - Needs an interactive real-time event pipeline (WebSocket/Polling with START, PAUSE, RESUME, RESET, and 1x/2x/5x/10x SPEED controls) generating live events into the event feed and dynamically updating the radar/AIS map.
- **Interactive GeoTIFF / SAR Image Viewer:**
  - Needs high-res raster canvas controls (zoom, pan, opacity, brightness, contrast, and layer toggles for RAW SAR, PROCESSED SAR, OIL MASK, LOOK-ALIKE MASK, VESSEL DETECTIONS).

---

## 3. What is Broken / Outdated
- **Terminology:** References to "TideTrace", "Marine AI", "Oil Spill & Vessel Intelligence" exist in some legacy docstrings and comments. These must all be updated to **TideTrace** (Subtitle: *"Oil Spill Detection, Drift Analysis & Vessel Attribution Platform"*) with the word "Marine" completely removed.
- **Map Container Resizing on Tab Switch:** Switching pages needs explicit `invalidateSize()` calls and independent map state management per page so tiles never glitch.

---

## 4. What is Missing
- **All 11 Dedicated Pages in UI:**
  1. PAGE 1 — COMMAND CENTER
  2. PAGE 2 — INCIDENT ANALYSIS (Detailed incident metrics, 9-layer interactive map)
  3. PAGE 3 — DARK VESSEL (Real-time live event feed, simulation speed controller, SAR vs AIS correlation table)
  4. PAGE 4 — SAR ANALYSIS (Interactive satellite viewer with brightness/contrast/opacity sliders and GeoTIFF upload)
  5. PAGE 5 — VESSEL INTELLIGENCE (Search, timeline inspector, historical tracks, anomaly breakdown)
  6. PAGE 6 — DRIFT & LANDFALL (24h/48h/72h ensemble forecasts, probability bands, ESI risk zones)
  7. PAGE 7 — ATTRIBUTION (Explainable attribution score, non-accusatory ranking)
  8. PAGE 8 — INCIDENT REPLAY (T-72h to T+72h timeline scrubber with Play/Pause/Speed)
  9. PAGE 9 — EVIDENCE DOSSIER (SHA-256 decision-support evidence package)
  10. PAGE 10 — ML LAB (Model training/evaluation, holdout metrics, confusion matrix)
  11. PAGE 11 — SYSTEM / ARCHITECTURE (Visual end-to-end architecture diagram)
- **Documentation Suite:**
  - `README.md`, `PROJECT_STATE.md`, `ARCHITECTURE.md`, `MODEL_CARD.md`, `DATA_CARD.md`, `DEMO_GUIDE.md`, `RESEARCH.md`.

---

## 5. Where Development Stopped
Development stopped after implementing the core Python backend, PyTorch ML models, drift engines, and initial HTML prototype. The next step is building the complete 11-page command-and-control frontend with real-time event simulation, multi-layer GIS maps, and full documentation.

---

## 6. What Should Be Completed Next
1. Remove the word "Marine" across all codebase files, comments, titles, and templates.
2. Build the complete, professional 11-page UI in `frontend/index.html` with dark operational C2 design, multi-page routing, dedicated interactive Leaflet maps, real-time dark vessel event simulator with speed controls, SAR image viewer with brightness/contrast, and incident replay engine.
3. Add WebSocket / polling endpoint in `backend/main.py` for real-time dark vessel events.
4. Generate the full documentation suite (`README.md`, `ARCHITECTURE.md`, `MODEL_CARD.md`, `DATA_CARD.md`, `DEMO_GUIDE.md`, `RESEARCH.md`).
5. Run automated tests and end-to-end verification.
