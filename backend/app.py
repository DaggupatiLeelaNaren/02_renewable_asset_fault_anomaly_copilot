from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score

app = FastAPI(title='Renewable Asset Fault & Anomaly Detection Copilot')

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / 'data' / 'asset_telemetry.csv'
FRONTEND = BASE / 'frontend'
MODEL_FEATURES = ['irradiance_wm2', 'wind_speed_ms', 'temperature_c', 'vibration_mm_s', 'power_mw']
TELEMETRY_COLUMNS = [
    'timestamp_id',
    'asset_id',
    'hour',
    'irradiance_wm2',
    'wind_speed_ms',
    'temperature_c',
    'vibration_mm_s',
    'power_mw',
]

app.mount('/frontend', StaticFiles(directory=FRONTEND), name='frontend')


class RunRequest(BaseModel):
    text: str = Field(default='', description='Optional operator note or query.')
    asset_id: str = Field(default='', description='Asset ID to investigate. Defaults to all assets.')


class AssetSummary(BaseModel):
    asset_id: str
    records: int
    avg_temperature_c: float
    avg_vibration_mm_s: float
    avg_power_mw: float
    anomaly_count: int


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    for column in MODEL_FEATURES:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    df = df.dropna(subset=MODEL_FEATURES).copy()
    df['asset_id'] = df['asset_id'].astype(str)
    df = df.sort_values(['asset_id', 'timestamp_id']).reset_index(drop=True)
    return df


def add_rule_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['temp_q99'] = df.groupby('asset_id')['temperature_c'].transform(lambda s: s.quantile(0.99))
    df['vib_q99'] = df.groupby('asset_id')['vibration_mm_s'].transform(lambda s: s.quantile(0.99))
    df['power_q10'] = df.groupby('asset_id')['power_mw'].transform(lambda s: s.quantile(0.1))
    df['rule_anomaly'] = (
        (df['temperature_c'] > df['temp_q99'])
        | (df['vibration_mm_s'] > df['vib_q99'])
        | ((df['irradiance_wm2'] > 100) & (df['power_mw'] < df['power_q10']))
        | ((df['wind_speed_ms'] > 4) & (df['power_mw'] < df['power_q10']))
    )
    return df


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    features = df[MODEL_FEATURES].fillna(df[MODEL_FEATURES].median())
    model = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
    model.fit(features)
    score = -model.score_samples(features)
    label = model.predict(features)

    df = df.copy()
    df['anomaly_score'] = score
    df['is_anomaly'] = label == -1
    score_median = float(np.median(score))
    score_std = float(np.std(score))
    df['severity'] = np.clip((score - score_median) / (score_std + 1e-9) * 60 + 65, 0, 100)
    return df


def build_asset_summary(df: pd.DataFrame) -> list[dict]:
    summary = []
    for asset_id, group in df.groupby('asset_id'):
        anomaly_count = int(group['is_anomaly'].sum())
        summary.append({
            'asset_id': asset_id,
            'records': int(len(group)),
            'avg_temperature_c': round(float(group['temperature_c'].mean()), 2),
            'avg_vibration_mm_s': round(float(group['vibration_mm_s'].mean()), 2),
            'avg_power_mw': round(float(group['power_mw'].mean()), 2),
            'anomaly_count': anomaly_count,
        })
    return summary


def explain_anomaly(row: pd.Series, asset_profile: pd.DataFrame) -> dict:
    asset_stats = asset_profile.loc[row['asset_id']] if row['asset_id'] in asset_profile.index else asset_profile.iloc[0]
    temp_mean = float(asset_stats['temperature_c_mean'])
    temp_std = float(asset_stats['temperature_c_std'])
    vib_mean = float(asset_stats['vibration_mm_s_mean'])
    vib_std = float(asset_stats['vibration_mm_s_std'])
    power_median = float(asset_stats['power_mw_median'])

    temp_delta = row['temperature_c'] - temp_mean
    vib_delta = row['vibration_mm_s'] - vib_mean
    power_gap = power_median - row['power_mw']

    if row['vibration_mm_s'] > max(vib_mean + 3 * vib_std, 2.5) and row['temperature_c'] > temp_mean + 2 * temp_std:
        likely_fault = 'Mechanical wear or misalignment'
        operator_summary = (
            f"{row['asset_id']} shows a sharp rise in temperature and vibration, which is consistent with a bearing, gearbox, or alignment issue."
        )
        checks = [
            'Inspect bearing housing and gearbox oil condition.',
            'Check shaft alignment and looseness around the drivetrain.',
            'Review trend from prior hours for repeated vibration growth.',
        ]
    elif row['temperature_c'] > max(temp_mean + 2 * temp_std, 40) and row['power_mw'] < max(power_median * 0.7, 0.25):
        likely_fault = 'Thermal overload or cooling problem'
        operator_summary = (
            f"{row['asset_id']} is overheating while output is depressed, suggesting a thermal or inverter-side restriction."
        )
        checks = [
            'Verify cooling fan and filter health.',
            'Review inverter thermal alarms and DC-side load balancing.',
            'Confirm ambient temperature and ventilation conditions around the asset.',
        ]
    elif row['power_mw'] < max(power_median * 0.7, 0.25) and row['irradiance_wm2'] > 100:
        likely_fault = 'Output loss under normal generation conditions'
        operator_summary = (
            f"{row['asset_id']} is under-producing compared with the expected irradiance or wind band, which often indicates a converter, curtailment, or field-side fault."
        )
        checks = [
            'Inspect inverter string health and breaker status.',
            'Check for soiling, blade pitch issues, or curtailment logs.',
            'Compare with nearby assets in the same weather window.',
        ]
    elif vib_delta > 2 * vib_std:
        likely_fault = 'Elevated mechanical vibration'
        operator_summary = (
            f"{row['asset_id']} has vibration above its normal range, which can signal an imbalance, looseness, or bearing wear."
        )
        checks = [
            'Inspect for loose mounting hardware and any unusual noise.',
            'Review balance and maintenance logs for the drivetrain.',
            'Schedule a vibration verification route and compare to baseline.',
        ]
    else:
        likely_fault = 'Unusual operating regime'
        operator_summary = (
            f"{row['asset_id']} sits outside its typical operating envelope for this time window and deserves a targeted review."
        )
        checks = [
            'Check for recent control changes or mode shifts.',
            'Review asset alarms and SCADA events from the same period.',
            'Compare against peer assets for similar operating conditions.',
        ]

    confidence = int(round(np.clip((row['severity'] + max(abs(temp_delta), abs(vib_delta), power_gap) * 10) / 2, 0, 100)))
    return {
        'asset_id': row['asset_id'],
        'timestamp_id': int(row['timestamp_id']),
        'hour': int(row['hour']),
        'likely_fault': likely_fault,
        'operator_summary': operator_summary,
        'recommended_checks': checks,
        'confidence': confidence,
        'anomaly_score': round(float(row['anomaly_score']), 4),
        'severity': round(float(row['severity']), 1),
        'temperature_c': round(float(row['temperature_c']), 2),
        'vibration_mm_s': round(float(row['vibration_mm_s']), 2),
        'power_mw': round(float(row['power_mw']), 2),
        'irradiance_wm2': round(float(row['irradiance_wm2']), 2),
        'wind_speed_ms': round(float(row['wind_speed_ms']), 2),
    }


def build_analysis_dataframe() -> pd.DataFrame:
    df = load_data()
    df = add_rule_ground_truth(df)
    df = detect_anomalies(df)

    asset_profile = (
        df.groupby('asset_id')
        .agg(
            temperature_c_mean=('temperature_c', 'mean'),
            temperature_c_std=('temperature_c', 'std'),
            vibration_mm_s_mean=('vibration_mm_s', 'mean'),
            vibration_mm_s_std=('vibration_mm_s', 'std'),
            power_mw_median=('power_mw', 'median'),
        )
        .fillna(0)
    )

    df['explanation'] = pd.Series(
        [
            explain_anomaly(row, asset_profile) if row['is_anomaly'] else None
            for _, row in df.iterrows()
        ],
        index=df.index,
        dtype='object',
    )
    return df


@app.get('/api/demo')
def demo():
    df = load_data()
    return {
        'case_study': 'Renewable Asset Fault & Anomaly Detection Copilot',
        'status': 'online',
        'assets': sorted(df['asset_id'].unique().tolist()),
        'records': int(len(df)),
        'expected_outcome': 'Flag abnormal renewable asset behavior and explain likely causes in operator-friendly language.',
    }


@app.get('/api/assets')
def get_assets():
    df = load_data()
    return {'assets': sorted(df['asset_id'].unique().tolist())}


@app.post('/api/run')
def run_case(request: RunRequest):
    df = build_analysis_dataframe()
    asset_filter = request.asset_id.strip() if request.asset_id else None
    if asset_filter:
        if asset_filter not in df['asset_id'].unique():
            raise HTTPException(status_code=404, detail=f'Unknown asset_id: {asset_filter}')
        df = df[df['asset_id'] == asset_filter].copy()

    anomaly_rows = df[df['is_anomaly']].copy()
    ordered_anomalies = anomaly_rows.sort_values('severity', ascending=False).head(10)
    explanations = [explain_anomaly(row, df.groupby('asset_id').agg(
        temperature_c_mean=('temperature_c', 'mean'),
        temperature_c_std=('temperature_c', 'std'),
        vibration_mm_s_mean=('vibration_mm_s', 'mean'),
        vibration_mm_s_std=('vibration_mm_s', 'std'),
        power_mw_median=('power_mw', 'median'),
    ).fillna(0)) for _, row in ordered_anomalies.iterrows()]

    metric_frame = df[['rule_anomaly', 'is_anomaly']].copy()
    metric_frame['rule_anomaly'] = metric_frame['rule_anomaly'].astype(int)
    metric_frame['is_anomaly'] = metric_frame['is_anomaly'].astype(int)
    precision = precision_score(metric_frame['rule_anomaly'], metric_frame['is_anomaly'], zero_division=0)
    recall = recall_score(metric_frame['rule_anomaly'], metric_frame['is_anomaly'], zero_division=0)
    f1 = f1_score(metric_frame['rule_anomaly'], metric_frame['is_anomaly'], zero_division=0)

    return {
        'input': request.text,
        'selected_asset': asset_filter or 'ALL_ASSETS',
        'summary': {
            'records': int(len(df)),
            'anomaly_count': int(len(anomaly_rows)),
            'anomaly_rate': round(float(len(anomaly_rows) / max(len(df), 1)), 4),
            'assets': sorted(df['asset_id'].unique().tolist()),
            'model_performance': {
                'precision': round(float(precision), 4),
                'recall': round(float(recall), 4),
                'f1_score': round(float(f1), 4),
            },
        },
        'top_anomalies': explanations,
        'asset_summary': build_asset_summary(df),
        'telemetry': df[TELEMETRY_COLUMNS + ['is_anomaly']]
        .assign(is_anomaly=lambda rows: rows['is_anomaly'].astype(bool))
        .to_dict(orient='records'),
    }


@app.get('/')
def root():
    return FileResponse(FRONTEND / 'index.html')
