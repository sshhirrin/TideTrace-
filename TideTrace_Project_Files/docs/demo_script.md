# TideTrace // Smart India Hackathon 2026 Presentation Script
**Problem Statement:** NTRO PS 26143: Domain Awareness for Oil Spill Detection & Vessel Attribution  
**System:** TideTrace  

---

## ⏱️ 5-Minute High-Impact Jury Presentation Flow

```mermaid
timeline
    title 5-Minute Jury Demonstration Script
    00:00 - 01:00 : Executive Problem & Single-Click Master Demo ("RUN INCIDENT")
    01:00 - 02:00 : SAR Neural Segmentation & 3-Class Look-Alike Rejection
    02:00 - 03:00 : 2D CA-CFAR Radar Detection & Dark Vessel Inset Zoom
    03:00 - 04:00 : 100-Particle Reverse Drift & 7-Factor MCDA Attribution
    04:00 - 05:00 : 72h Landfall Forecast, Model Lab & SHA-256 Sealed Dossier
```

---

### Phase 1: The Challenge & One-Click Master Demo (00:00 - 01:00)
- **Speaker:** *"Respected Jury, illegal bilge dumping under cover of darkness causes severe ecological damage while offending vessels disable AIS transponders to evade detection. We present TideTrace: a complete Oil Spill Detection & Drift Analysis and Vessel Attribution System."*
- **Action:** Select **Scenario Alpha (The Rogue Tanker - Arabian Sea)** in the top bar and click **`RUN INCIDENT`**.
- **Visual Impact:** Top execution banner animates across 6 pipeline stages (SAR Preprocessing $\to$ Neural Segmentation $\to$ CA-CFAR Dark Vessels $\to$ 100-Particle Ensemble Backtracking $\to$ MCDA Attribution $\to$ SHA-256 Digital Sealing).

---

### Phase 2: SAR AI & 3-Class Look-Alike Rejection (01:00 - 02:00)
- **Speaker:** *"A fundamental rule of satellite remote sensing: not every dark patch is oil. In our SAR Analysis module, we inspect calibrated Sentinel-1 C-Band backscatter through our 5x5 Adaptive Lee filter and PyTorch ResNet34 U-Net segmentation."*
- **Action:** Click **`SAR Analysis`** nav tab.
- **Visual Impact:** Point to the 4 comparative raster views (Raw VV, 5x5 Lee Filtered, U-Net Mask, 3-Class Risk Map). Show the 3-class distribution (**88% Mineral Oil**, 8% Look-alike, 4% Uncertain) with Damping Ratio $4.82\text{ dB}$ and valid wind window ($5.8\text{ m/s}$).

---

### Phase 3: CA-CFAR & Dark Vessel Red Zoom Inset (02:00 - 03:00)
- **Speaker:** *"What if the offending vessel turned off its AIS transponder? TideTrace runs 2D CA-CFAR radar detection to locate physical radar echoes regardless of AIS transmission."*
- **Action:** Click **`Dark Vessels`** nav tab.
- **Visual Impact:** Highlight the tactical radar scope showing Green (AIS matched), Amber (Kinematic anomaly), and **Red (Dark Vessel)** Oriented Bounding Boxes. Point out the **Red Zoom Inset box** in the bottom-left showing the magnified $8\times$ radar echo with hull length ($142\text{m}$) and heading vector.

---

### Phase 4: 100-Particle Reverse Drift & Attribution Leaderboard (03:00 - 04:00)
- **Speaker:** *"To locate the discharge event, our hydrodynamic engine launches 100 Monte Carlo particles backwards in time under wind leeway, Coriolis deflection, and turbulent dispersion."*
- **Action:** Click **`Reverse Drift`** then **`Attribution`**.
- **Visual Impact:** Show the 50%, 80%, and 95% confidence origin envelopes and the 7-factor Chart.js Radar Chart proving **MT AL-WASL** as Primary Suspect ($94.2\%$ confidence) based on spatial intercept, temporal transit, AIS silence gap ($184\text{ min}$), and $-2.1\text{m}$ draft drop.

---

### Phase 5: Forward Landfall, Model Lab & Evidence Dossier (04:00 - 05:00)
- **Speaker:** *"For emergency containment, TideTrace predicts 72h forward landfall and physico-chemical weathering. Finally, we generate a cryptographically sealed forensic dossier under MARPOL 73/78 Annex I."*
- **Action:** Click **`Forward Impact`**, **`Model Lab`** (showing real holdout benchmark metrics), and **`Evidence Dossier`** (showing the SHA-256 seal).
- **Closing:** *"TideTrace bridges satellite earth observation, ocean physics, and Marine law enforcement into a unified, actionable sovereign surveillance platform. Thank you!"*\n