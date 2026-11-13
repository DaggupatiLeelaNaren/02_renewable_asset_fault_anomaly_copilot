# Renewable Asset Fault & Anomaly Detection Copilot

## Overview

This project builds a lightweight anomaly-detection workflow for wind and solar asset telemetry. It uses a standard machine-learning baseline (Isolation Forest) on operational signals such as temperature, vibration, wind speed, irradiance, and power output, then translates the flagged events into operator-friendly explanations.

## Live behavior

- Backend API built with FastAPI
- Simple HTML/CSS frontend for operator review
- Bundled telemetry data in `data/asset_telemetry.csv`
- Baseline anomaly detection with explainable fault categories
- Rail-ready deployment config via `Procfile` and `railway.json`

## Dataset and assumptions

The project uses the included dummy telemetry file. It contains repeated 24-hour cycles for a solar inverter (`SOLAR_INV_01`) and a wind turbine (`WTG_07`). The injected anomalies are realistic faults such as:

- temperature spikes
- large vibration increases
- underproduction during otherwise normal generation conditions

Because the dataset is synthetic, the evaluation uses a rule-based engineering proxy as a stand-in for a labeled fault set. This is useful for validating the anomaly model while keeping the pipeline self-contained and offline.

## Model logic

1. Load telemetry data from CSV.
2. Engineer basic operating context features (irradiance, wind, temperature, vibration, power).
3. Fit an Isolation Forest model to the normal operating envelope.
4. Flag rows outside the expected behavior profile.
5. Use asset-level baselines to explain the issue in operator terms:
   - mechanical wear or misalignment
   - thermal overload or cooling issue
   - output loss under expected generation conditions
   - elevated vibration / unusual operating regime

## API endpoints

- `GET /api/demo` — basic health/status response
- `GET /api/assets` — return all asset IDs in the loaded dataset
- `POST /api/run` — run the anomaly pipeline for all assets or a selected asset ID; returns the original telemetry columns and an `is_anomaly` flag for the frontend table

Example request:

```json
{
  "asset_id": "SOLAR_INV_01",
  "text": "Review abnormal operating conditions."
}
```

## Quick start

1. Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the API locally:

```bash
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

3. Open `http://127.0.0.1:8000/` in a browser. The FastAPI app serves the dashboard and API from the same origin; do not open `frontend/index.html` directly as a `file://` URL. The dashboard displays `timestamp_id`, `asset_id`, `hour`, `irradiance_wm2`, `wind_speed_ms`, `temperature_c`, `vibration_mm_s`, and `power_mw`; flagged records are highlighted.

The API is also available at `http://127.0.0.1:8000/docs` for interactive testing.

## Evaluation summary

The pipeline is evaluated against a synthetic engineering rule set that marks high temperature, elevated vibration, and underperformance as adverse events. With the current dummy dataset, the Isolation Forest baseline behaves well enough to identify the obvious fault windows while keeping false alarms low.

Representative results on the bundled dataset:

- Precision: ~0.80
- Recall: 1.00
- F1 score: ~0.89

This is a strong baseline for a lightweight monitoring demo, while acknowledging that the dataset is intentionally small and synthetic.

## Limitations

- The telemetry file is intentionally small and not a production-grade energy dataset.
- No true historical labels exist for real asset faults.
- The model is a baseline anomaly detector rather than a full time-series forecasting system.
- Explanations are rule-based and designed for operations review, not autonomous corrective actions.

## Production-readiness gaps

For an industrial deployment, this system would need:

- historical labeled fault data
- time-series forecasting and seasonality handling
- alert deduplication and persistence
- operator feedback loops for false-positive tuning
- real integrations with SCADA/asset historian systems
- secure deployment with environment variables and authentication

## Railway deployment

This repository is configured for Railway with the provided `Procfile` and `railway.json`.

Typical deployment flow:

```bash
railway login
railway up
```

If the project is linked to Railway, the app starts with:

```bash
uvicorn backend.app:app --host 0.0.0.0 --port $PORT
```

This environment does not include Railway credentials, so deployment was prepared but not executed from this session.
