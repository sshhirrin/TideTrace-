-- ============================================================================
-- TideTrace: Enterprise PostgreSQL 16 + PostGIS 3.4 Production Schema
-- NTRO Problem Statement #26143: Oil Spill Detection & Drift Analysis & Vessel Attribution
-- ============================================================================

-- 1. Enable PostGIS Extension
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS btree_gist;

-- 2. Vessels Registry
CREATE TABLE IF NOT EXISTS vessels (
    mmsi INT PRIMARY KEY,
    imo INT,
    vessel_name VARCHAR(128) NOT NULL,
    callsign VARCHAR(32),
    ship_type VARCHAR(64) NOT NULL,
    flag VARCHAR(64) NOT NULL,
    length_m NUMERIC(6, 2) NOT NULL,
    beam_m NUMERIC(6, 2) NOT NULL,
    gross_tonnage INT,
    risk_weight NUMERIC(4, 2) DEFAULT 1.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. High-Frequency AIS Position Telemetry
CREATE TABLE IF NOT EXISTS ais_positions (
    id BIGSERIAL PRIMARY KEY,
    mmsi INT REFERENCES vessels(mmsi) ON DELETE CASCADE,
    telemetry_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    geom GEOMETRY(Point, 4326) NOT NULL,
    sog_knots NUMERIC(5, 2) NOT NULL,
    cog_degrees NUMERIC(5, 2) NOT NULL,
    heading_degrees NUMERIC(5, 2),
    nav_status VARCHAR(64) DEFAULT 'Under way using engine'
);

-- Spatial GIST index on AIS positions for rapid spatio-temporal lookups
CREATE INDEX IF NOT EXISTS idx_ais_positions_geom ON ais_positions USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_ais_positions_mmsi_time ON ais_positions (mmsi, telemetry_timestamp DESC);

-- 4. Reconstructed Vessel Trajectories (LineStrings)
CREATE TABLE IF NOT EXISTS vessel_trajectories (
    id BIGSERIAL PRIMARY KEY,
    mmsi INT REFERENCES vessels(mmsi) ON DELETE CASCADE,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE NOT NULL,
    track GEOMETRY(LineString, 4326) NOT NULL,
    waypoint_count INT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_vessel_trajectories_track ON vessel_trajectories USING GIST (track);

-- 5. Marine Incidents & Investigation Cases
CREATE TABLE IF NOT EXISTS incidents (
    id VARCHAR(64) PRIMARY KEY,
    title VARCHAR(256) NOT NULL,
    status VARCHAR(32) DEFAULT 'ACTIVE_INVESTIGATION',
    region_name VARCHAR(128) NOT NULL,
    detected_at TIMESTAMP WITH TIME ZONE NOT NULL,
    scene_id VARCHAR(128) NOT NULL,
    satellite_sensor VARCHAR(64) NOT NULL,
    slick_area_sqkm NUMERIC(10, 4) NOT NULL,
    estimated_volume_m3 NUMERIC(10, 2) NOT NULL,
    bounding_box GEOMETRY(Polygon, 4326),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 6. Delineated SAR Oil Slick Polygons
CREATE TABLE IF NOT EXISTS spill_polygons (
    id VARCHAR(64) PRIMARY KEY,
    incident_id VARCHAR(64) REFERENCES incidents(id) ON DELETE CASCADE,
    geom GEOMETRY(Polygon, 4326) NOT NULL,
    centroid GEOMETRY(Point, 4326) NOT NULL,
    area_sqkm NUMERIC(10, 4) NOT NULL,
    perimeter_km NUMERIC(10, 3) NOT NULL,
    radar_damping_db NUMERIC(5, 2) NOT NULL,
    oil_probability NUMERIC(4, 3) NOT NULL,
    lookalike_probability NUMERIC(4, 3) NOT NULL,
    uncertainty NUMERIC(4, 3) NOT NULL,
    classification VARCHAR(32) NOT NULL -- 'OIL', 'LOOK-ALIKE', 'UNCERTAIN'
);

CREATE INDEX IF NOT EXISTS idx_spill_polygons_geom ON spill_polygons USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_spill_polygons_centroid ON spill_polygons USING GIST (centroid);

-- 7. SAR Radar Ship Detections (CFAR / OBB)
CREATE TABLE IF NOT EXISTS sar_ship_detections (
    id VARCHAR(64) PRIMARY KEY,
    incident_id VARCHAR(64) REFERENCES incidents(id) ON DELETE CASCADE,
    centroid GEOMETRY(Point, 4326) NOT NULL,
    obb_polygon GEOMETRY(Polygon, 4326) NOT NULL,
    estimated_length_m NUMERIC(6, 2) NOT NULL,
    estimated_beam_m NUMERIC(6, 2) NOT NULL,
    cfar_snr_db NUMERIC(5, 2) NOT NULL,
    is_dark_vessel BOOLEAN DEFAULT FALSE,
    matched_mmsi INT REFERENCES vessels(mmsi) ON DELETE SET NULL,
    intelligence_status VARCHAR(64) NOT NULL -- 'NORMAL', 'POSSIBLE DARK VESSEL', 'PRIMARY SUSPECT', 'REQUIRES REVIEW'
);

CREATE INDEX IF NOT EXISTS idx_sar_ship_detections_centroid ON sar_ship_detections USING GIST (centroid);

-- 8. Backtracked Origin Probability Density Regions
CREATE TABLE IF NOT EXISTS origin_probability_regions (
    id BIGSERIAL PRIMARY KEY,
    incident_id VARCHAR(64) REFERENCES incidents(id) ON DELETE CASCADE,
    confidence_level_pct INT NOT NULL, -- 50, 80, 95
    confidence_polygon GEOMETRY(Polygon, 4326) NOT NULL,
    radius_km NUMERIC(6, 2) NOT NULL,
    estimated_elapsed_hours NUMERIC(5, 2) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_origin_prob_regions ON origin_probability_regions USING GIST (confidence_polygon);

-- ============================================================================
-- SPATIAL QUERY PROCEDURES & FORENSIC QUERIES
-- ============================================================================

-- Query 1: Find vessels passing within X km of backtracked origin cone using ST_DWithin
-- SELECT v.mmsi, v.vessel_name, v.ship_type,
--        ST_Distance(p.geom::geography, origin_pt::geography) / 1000.0 AS distance_km
-- FROM ais_positions p
-- JOIN vessels v ON p.mmsi = v.mmsi
-- WHERE ST_DWithin(p.geom::geography, ST_SetSRID(ST_MakePoint(72.35, 18.82), 4326)::geography, 8000.0)
--   AND p.telemetry_timestamp BETWEEN '2026-09-01 02:00:00Z' AND '2026-09-01 04:00:00Z'
-- ORDER BY distance_km ASC;

-- Query 2: Find Closest Point of Approach (CPA) along trajectory using ST_ClosestPoint
-- SELECT v.mmsi, v.vessel_name,
--        ST_AsText(ST_ClosestPoint(t.track, ST_SetSRID(ST_MakePoint(72.35, 18.82), 4326))) AS closest_point_geom
-- FROM vessel_trajectories t
-- JOIN vessels v ON t.mmsi = v.mmsi
-- WHERE ST_DWithin(t.track::geography, ST_SetSRID(ST_MakePoint(72.35, 18.82), 4326)::geography, 10000.0);

-- Query 3: Intersect forward drift forecast with sensitive Marine Protected Areas (MPAs)
-- SELECT mpa.name, mpa.vulnerability, ST_Intersects(fwd_drift.geom, mpa.geom) AS is_impacted
-- FROM sensitive_coastal_areas mpa, forward_drift_corridors fwd_drift
-- WHERE ST_Intersects(fwd_drift.geom, mpa.geom);
