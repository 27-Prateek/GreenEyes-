"""
preprocessing/preprocess.py
============================
Complete preprocessing pipeline for all 5 Indian AQI CSVs.
"""

import sys
import warnings
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import *

set_seed(RANDOM_SEED)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_city_day(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['Date'])
    df = df.rename(columns={'Date': 'datetime'})
    df['granularity'] = 'daily'
    df['source']      = 'city'
    df['StationId']   = df['City'].apply(lambda c: f'CITY_{c.upper()[:4]}')
    print(f"  city_day:     {len(df):>8,} rows | {df['City'].nunique()} cities | "
          f"{df['datetime'].min().date()} → {df['datetime'].max().date()}")
    return df


def load_city_hour(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['Datetime'])
    df = df.rename(columns={'Datetime': 'datetime'})
    df['granularity'] = 'hourly'
    df['source']      = 'city'
    df['StationId']   = df['City'].apply(lambda c: f'CITY_{c.upper()[:4]}')
    print(f"  city_hour:    {len(df):>8,} rows | {df['City'].nunique()} cities | "
          f"{df['datetime'].min().date()} → {df['datetime'].max().date()}")
    return df


def load_station_day(path: Path, stations: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['Date'])
    df = df.rename(columns={'Date': 'datetime'})
    df = df.merge(stations[['StationId', 'City', 'State']], on='StationId', how='left')
    df['granularity'] = 'daily'
    df['source']      = 'station'
    print(f"  station_day:  {len(df):>8,} rows | {df['StationId'].nunique()} stations | "
          f"{df['datetime'].min().date()} → {df['datetime'].max().date()}")
    return df


def load_station_hour(path: Path, stations: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['Datetime'])
    df = df.rename(columns={'Datetime': 'datetime'})
    df['StationId'] = df['StationId'].replace('', np.nan).ffill()
    all_null  = df[POLLUTANT_COLS].isnull().all(axis=1)
    n_dropped = all_null.sum()
    if n_dropped:
        print(f"  [station_hour] Dropping {n_dropped} fully-empty rows")
    df = df[~all_null].reset_index(drop=True)
    df = df.merge(stations[['StationId', 'City', 'State']], on='StationId', how='left')
    df['granularity'] = 'hourly'
    df['source']      = 'station'
    print(f"  station_hour: {len(df):>8,} rows | {df['StationId'].nunique()} stations | "
          f"{df['datetime'].min().date()} → {df['datetime'].max().date()}")
    return df


def load_stations(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['Status'] = df['Status'].replace('', 'Unknown').fillna('Unknown')
    print(f"  stations:     {len(df):>8,} stations | "
          f"{(df['Status']=='Active').sum()} active")
    return df


def load_all_data(data_dir: Path) -> dict:
    print("\n[1] Loading raw CSVs...")
    stations     = load_stations(data_dir / 'stations.csv')
    city_day     = load_city_day(data_dir / 'city_day.csv')
    city_hour    = load_city_hour(data_dir / 'city_hour.csv')
    station_day  = load_station_day(data_dir / 'station_day.csv', stations)
    station_hour = load_station_hour(data_dir / 'station_hour.csv', stations)
    return dict(city_day=city_day, city_hour=city_hour,
                station_day=station_day, station_hour=station_hour,
                stations=stations)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MERGE
# ══════════════════════════════════════════════════════════════════════════════

COMMON_COLS = ['datetime', 'City', 'StationId', 'granularity', 'source',
               'AQI', 'AQI_Bucket'] + POLLUTANT_COLS


def merge_hourly(city_hour: pd.DataFrame, station_hour: pd.DataFrame) -> pd.DataFrame:
    print("\n[2] Merging hourly datasets...")

    def _align(df):
        for c in COMMON_COLS:
            if c not in df.columns:
                df[c] = np.nan
        return df[COMMON_COLS].copy()

    combined = pd.concat([_align(city_hour), _align(station_hour)], ignore_index=True)
    combined['datetime'] = pd.to_datetime(combined['datetime'])
    combined = combined.sort_values(['City', 'StationId', 'datetime'])
    combined['_priority'] = (combined['source'] == 'city').astype(int)
    combined = (combined
                .sort_values('_priority', ascending=False)
                .drop_duplicates(subset=['City', 'datetime'], keep='first')
                .drop(columns=['_priority'])
                .reset_index(drop=True))
    print(f"  Merged: {len(combined):,} rows | {combined['City'].nunique()} unique cities")
    print(f"  Date range: {combined['datetime'].min().date()} → {combined['datetime'].max().date()}")
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MISSING VALUE IMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def report_missing(df: pd.DataFrame, label: str = ''):
    cols  = [c for c in POLLUTANT_COLS + ['AQI'] if c in df.columns]
    total = len(df)
    print(f"\n  Missing values [{label}]:")
    for c in cols:
        n = df[c].isna().sum()
        if n > 0:
            print(f"    {c:<12} {n:>7,} / {total:,}  ({n/total*100:.1f}%)")


def _identify_gap_lengths(s: pd.Series) -> pd.Series:
    null_mask = s.isna()
    run_id    = (null_mask != null_mask.shift()).cumsum()
    return null_mask.groupby(run_id).transform('sum')


def impute_series(s: pd.Series, times: pd.Series, granularity: str = 'hourly') -> pd.Series:
    short  = SHORT_GAP_HOURS  if granularity == 'hourly' else 1
    medium = MEDIUM_GAP_HOURS if granularity == 'hourly' else 3
    s       = s.copy()
    gap_len = _identify_gap_lengths(s)
    s = s.interpolate(method='linear', limit=short)
    still_null = s.isna()
    if still_null.any():
        if granularity == 'hourly':
            hour_med = pd.Series(s.values, index=times).groupby(times.dt.hour).transform('median')
            hour_med.index = s.index
            s[still_null] = hour_med[still_null]
        else:
            dow_med = pd.Series(s.values, index=times).groupby(times.dt.dayofweek).transform('median')
            dow_med.index = s.index
            s[still_null] = dow_med[still_null]
    return s


def impute_dataframe(df: pd.DataFrame, granularity: str = 'hourly') -> pd.DataFrame:
    print(f"\n[3] Imputing missing values ({granularity})...")
    df = df.copy().sort_values(['City', 'datetime'])
    for col in POLLUTANT_COLS:
        if col not in df.columns:
            continue
        df[col] = df.groupby('City', group_keys=False).apply(
            lambda g: impute_series(g[col], g['datetime'], granularity)
        )
        df[f'{col}_mask'] = df[col].isna().astype(np.int8)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — AQI RECOMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

_PM25_BP = [
    (0.0,   12.0,   0,   50),
    (12.1,  35.4,  51,  100),
    (35.5,  55.4, 101,  150),
    (55.5, 150.4, 151,  200),
    (150.5,250.4, 201,  300),
    (250.5,500.4, 301,  500),
]


def pm25_to_aqi(pm25: float) -> float:
    if pd.isna(pm25) or pm25 < 0:
        return np.nan
    for bp_lo, bp_hi, aqi_lo, aqi_hi in _PM25_BP:
        if bp_lo <= pm25 <= bp_hi:
            return (aqi_hi - aqi_lo) / (bp_hi - bp_lo) * (pm25 - bp_lo) + aqi_lo
    return 500.0


def aqi_to_bucket(aqi: float) -> str:
    if pd.isna(aqi): return ''
    if aqi <= 50:    return 'Good'
    if aqi <= 100:   return 'Satisfactory'
    if aqi <= 200:   return 'Moderate'
    if aqi <= 300:   return 'Poor'
    if aqi <= 400:   return 'Very Poor'
    return 'Severe'


def recompute_aqi(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[4] Recomputing missing AQI from PM2.5...")
    df           = df.copy()
    missing_mask = df['AQI'].isna() & df['PM2.5'].notna()
    df.loc[missing_mask, 'AQI'] = df.loc[missing_mask, 'PM2.5'].apply(pm25_to_aqi)
    bucket_missing = df['AQI_Bucket'].isna() | (df['AQI_Bucket'] == '')
    df.loc[bucket_missing, 'AQI_Bucket'] = df.loc[bucket_missing, 'AQI'].apply(aqi_to_bucket)
    df['AQI_label'] = df['AQI_Bucket'].map(AQI_CATEGORY_MAP).fillna(-1).astype(int)
    filled    = missing_mask.sum()
    remaining = df['AQI'].isna().sum()
    print(f"  Filled {filled:,} AQI values | {remaining:,} still missing (will be imputed)")
    df['AQI'] = df.groupby('City')['AQI'].transform(
        lambda s: s.interpolate('linear').ffill().bfill()
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — POLYGONALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def polygonalize(series: pd.Series, window: int, polyorder: int = 2) -> pd.Series:
    s      = series.copy().astype(float)
    filled = s.interpolate('linear').ffill().bfill()
    if len(filled) < window:
        return s
    smoothed = savgol_filter(filled.values, window_length=window, polyorder=polyorder)
    result   = pd.Series(np.clip(smoothed, 0, 500), index=series.index)
    result[series.isna()] = np.nan
    return result


def polygonalize_dataframe(df: pd.DataFrame, granularity: str = 'hourly') -> pd.DataFrame:
    w = SAVGOL_WINDOW if granularity == 'hourly' else SAVGOL_WINDOW_D
    print(f"\n[5] Polygonalizing AQI (Savitzky-Golay, window={w}, "
          f"polyorder={SAVGOL_POLYORDER})...")
    df = df.copy()
    df['AQI_poly'] = df.groupby('City')['AQI'].transform(
        lambda s: polygonalize(s, window=w, polyorder=SAVGOL_POLYORDER)
    )
    corr = df[['AQI', 'AQI_poly']].dropna().corr().iloc[0, 1]
    print(f"  AQI vs AQI_poly correlation: {corr:.4f} (should be > 0.95)")
    if corr < 0.95:
        print(f"  WARNING: correlation still below 0.95 — consider increasing "
              f"SAVGOL_WINDOW to 73 in config.py")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df['datetime']
    df['hour_sin']       = np.sin(2 * np.pi * dt.dt.hour      / 24)
    df['hour_cos']       = np.cos(2 * np.pi * dt.dt.hour      / 24)
    df['dow_sin']        = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    df['dow_cos']        = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    df['month_sin']      = np.sin(2 * np.pi * dt.dt.month     / 12)
    df['month_cos']      = np.cos(2 * np.pi * dt.dt.month     / 12)
    df['doy_sin']        = np.sin(2 * np.pi * dt.dt.dayofyear / 365)
    df['doy_cos']        = np.cos(2 * np.pi * dt.dt.dayofyear / 365)
    df['is_weekend']     = (dt.dt.dayofweek >= 5).astype(np.int8)
    df['is_festive']     = dt.dt.month.isin([10, 11]).astype(np.int8)
    df['is_winter_smog'] = dt.dt.month.isin([11, 12, 1, 2]).astype(np.int8)
    df['is_monsoon']     = dt.dt.month.isin([6, 7, 8, 9]).astype(np.int8)
    return df


def add_lag_features(df: pd.DataFrame, target: str = 'AQI_poly') -> pd.DataFrame:
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
        df[f'{target}_lag_{lag}h'] = df.groupby('City')[target].transform(
            lambda s: s.shift(lag)
        )
    for w in [6, 12, 24, 48, 72, 168]:
        df[f'{target}_rmean_{w}h'] = df.groupby('City')[target].transform(
            lambda s: s.shift(1).rolling(w, min_periods=max(1, w // 4)).mean()
        )
        df[f'{target}_rstd_{w}h'] = df.groupby('City')[target].transform(
            lambda s: s.shift(1).rolling(w, min_periods=max(1, w // 4)).std()
        )
    df[f'{target}_diff_1h']  = df.groupby('City')[target].transform(lambda s: s.diff(1))
    df[f'{target}_diff_24h'] = df.groupby('City')[target].transform(lambda s: s.diff(24))
    return df


def add_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    if 'PM2.5' in df.columns and 'PM10' in df.columns:
        df['PM_ratio']  = (df['PM2.5'] / df['PM10'].replace(0, np.nan)).clip(0, 1)
        df['PM_total']  = df['PM2.5'].fillna(0) + df['PM10'].fillna(0)
    if 'O3' in df.columns and 'NO2' in df.columns:
        df['oxidant_idx'] = df['O3'].fillna(0) + df['NO2'].fillna(0)
    if 'NO' in df.columns and 'NO2' in df.columns:
        df['NOx_ratio'] = (df['NO'] / (df['NO2'] + 1e-6)).clip(0, 10)
    if 'Benzene' in df.columns and 'Toluene' in df.columns:
        df['BT_ratio']  = (df['Benzene'] / (df['Toluene'] + 1e-6)).clip(0, 10)
    df['n_sensors_ok'] = df[POLLUTANT_COLS].notna().sum(axis=1).astype(np.int8)
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[6] Feature engineering...")
    df = add_temporal_features(df)
    df = add_lag_features(df, target='AQI_poly')
    df = add_composite_features(df)
    df['AQI_label'] = df['AQI_Bucket'].map(AQI_CATEGORY_MAP).fillna(-1).astype(int)
    n_feat = len([c for c in df.columns if c not in
                  ['datetime', 'City', 'StationId', 'granularity', 'source',
                   'AQI', 'AQI_Bucket', 'AQI_label', 'AQI_poly']])
    print(f"  Total engineered features: {n_feat}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — CITY GRAPH
# ══════════════════════════════════════════════════════════════════════════════

CITY_COORDS = {
    'Ahmedabad': (23.0225, 72.5714), 'Aizawl': (23.7271, 92.7176),
    'Amaravati': (16.5728, 80.3582), 'Amritsar': (31.6340, 74.8723),
    'Bengaluru': (12.9716, 77.5946), 'Bhopal': (23.2599, 77.4126),
    'Brajrajnagar': (21.8253, 83.9175), 'Chandigarh': (30.7333, 76.7794),
    'Chennai': (13.0827, 80.2707), 'Coimbatore': (11.0168, 76.9558),
    'Delhi': (28.7041, 77.1025), 'Ernakulam': (9.9816, 76.2999),
    'Gurugram': (28.4595, 77.0266), 'Guwahati': (26.1445, 91.7362),
    'Hyderabad': (17.3850, 78.4867), 'Jaipur': (26.9124, 75.7873),
    'Jorapokhar': (23.7271, 86.4048), 'Kochi': (9.9312, 76.2673),
    'Kolkata': (22.5726, 88.3639), 'Lucknow': (26.8467, 80.9462),
    'Mumbai': (19.0760, 72.8777), 'Patna': (25.5941, 85.1376),
    'Shillong': (25.5788, 91.8933), 'Talcher': (20.9500, 85.2333),
    'Thiruvananthapuram': (8.5241, 76.9366), 'Visakhapatnam': (17.6868, 83.2185),
}


def haversine(lat1, lon1, lat2, lon2):
    R  = 6371.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a  = np.sin(dp/2)**2 + np.cos(p1) * np.cos(p2) * np.sin(dl/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def build_city_graph(cities: list, threshold_km: float = CITY_GRAPH_KM):
    print(f"\n[7] Building city graph (threshold={threshold_km}km)...")
    cities      = [c for c in cities if c in CITY_COORDS]
    n           = len(cities)
    city_to_idx = {c: i for i, c in enumerate(cities)}
    adj         = np.zeros((n, n), dtype=np.float32)
    for i, c1 in enumerate(cities):
        lat1, lon1 = CITY_COORDS[c1]
        for j, c2 in enumerate(cities):
            if i == j:
                continue
            lat2, lon2 = CITY_COORDS[c2]
            dist = haversine(lat1, lon1, lat2, lon2)
            if dist <= threshold_km:
                adj[i, j] = 1.0 / (dist + 1e-6)
    row_sums              = adj.sum(axis=1, keepdims=True)
    row_sums[row_sums==0] = 1
    adj                  = adj / row_sums
    n_edges = (adj > 0).sum()
    print(f"  {n} cities | {n_edges} directed edges | avg degree: {n_edges/n:.1f}")
    return adj, city_to_idx


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — TRAIN / VAL / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def temporal_split(df: pd.DataFrame):
    print(f"\n[8] Temporal split (val={VAL_YEAR}, test={TEST_YEAR})...")
    train = df[df['datetime'].dt.year <  VAL_YEAR].copy()
    val   = df[df['datetime'].dt.year == VAL_YEAR].copy()
    test  = df[df['datetime'].dt.year >= TEST_YEAR].copy()
    for name, d in [('Train', train), ('Val', val), ('Test', test)]:
        if len(d):
            print(f"  {name}: {len(d):>8,} rows | "
                  f"{d['datetime'].min().date()} → {d['datetime'].max().date()}")
        else:
            print(f"  {name}: 0 rows")
    return train, val, test


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — SCALING
# ══════════════════════════════════════════════════════════════════════════════

def get_feature_columns(df: pd.DataFrame) -> list:
    exclude   = {'datetime', 'City', 'StationId', 'granularity', 'source',
                 'AQI', 'AQI_Bucket', 'AQI_poly', 'AQI_label', 'State'}
    mask_cols  = [c for c in df.columns if c.endswith('_mask')]
    scale_cols = [c for c in df.columns
                  if c not in exclude
                  and not c.endswith('_mask')
                  and pd.api.types.is_numeric_dtype(df[c])]
    return scale_cols + mask_cols


def fit_scalers(train_df: pd.DataFrame, feature_cols: list) -> dict:
    scalers    = {}
    scale_only = [c for c in feature_cols if not c.endswith('_mask')]
    for col in scale_only:
        if col not in train_df.columns:
            continue
        valid = train_df[col].dropna()
        if len(valid) == 0:
            continue
        sc = StandardScaler()
        sc.fit(valid.values.reshape(-1, 1))
        scalers[col] = sc
    return scalers


def apply_scalers(df: pd.DataFrame, scalers: dict) -> pd.DataFrame:
    df = df.copy()
    for col, sc in scalers.items():
        if col not in df.columns:
            continue
        valid = df[col].notna()
        if valid.any():
            df.loc[valid, col] = sc.transform(
                df.loc[valid, col].values.reshape(-1, 1)
            ).flatten()
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — SEQUENCE BUILDING (memory-mapped, 500 MB chunks)
# ══════════════════════════════════════════════════════════════════════════════

CHUNK_MB = 500


class AQIDataset(Dataset):
    """Lazy dataset backed by memory-mapped .npy files."""
    def __init__(self, data_dir: Path, split: str):
        self.data_dir = Path(data_dir)
        self._split   = split
        info          = joblib.load(self.data_dir / f'{split}_info.joblib')
        self.n        = info['n']
        self.shape_X  = info['shape_X']
        self.shape_yr = info['shape_yr']
        self.meta     = info['meta']
        self._X     = np.load(str(self.data_dir / f'{split}_X.npy'),     mmap_mode='r')
        self._y_reg = np.load(str(self.data_dir / f'{split}_y_reg.npy'), mmap_mode='r')
        self._y_cls = np.load(str(self.data_dir / f'{split}_y_cls.npy'), mmap_mode='r')

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return (
            torch.tensor(np.array(self._X[idx]),     dtype=torch.float32),
            torch.tensor(np.array(self._y_reg[idx]), dtype=torch.float32),
            torch.tensor(int(self._y_cls[idx]),       dtype=torch.long),
        )


def _chunk_size(window_size: int, n_feat: int, budget_mb: int) -> int:
    bytes_per_seq = window_size * n_feat * 4
    return max(1, int((budget_mb * 1024 * 1024) // bytes_per_seq))


def _write_empty_split(out_dir, split_name, window_size, n_feat, n_horizons):
    np.save(str(out_dir / f'{split_name}_X.npy'),
            np.empty((0, window_size, n_feat), dtype=np.float32))
    np.save(str(out_dir / f'{split_name}_y_reg.npy'),
            np.empty((0, n_horizons), dtype=np.float32))
    np.save(str(out_dir / f'{split_name}_y_cls.npy'),
            np.empty((0,), dtype=np.int64))
    joblib.dump(
        dict(n=0, shape_X=(0, window_size, n_feat),
             shape_yr=(0, n_horizons), meta=[]),
        out_dir / f'{split_name}_info.joblib'
    )


def build_sequences(
    df,
    feature_cols:      list,
    out_dir:           Path,
    split_name:        str,
    target_col:        str   = 'AQI_poly',
    label_col:         str   = 'AQI_label',
    window_size:       int   = WINDOW_SIZE,
    forecast_horizons: list  = None,
    city_to_idx:       dict  = None,
    nan_threshold:     float = 0.30,
    chunk_mb:          int   = CHUNK_MB,
) -> AQIDataset:
    if forecast_horizons is None:
        forecast_horizons = [1, 24, 72]

    out_dir       = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_h         = max(forecast_horizons)
    n_feat        = len(feature_cols)
    n_horizons    = len(forecast_horizons)
    n_mask_cols   = sum(1 for c in feature_cols if c.endswith('_mask'))
    n_signal_cols = n_feat - n_mask_cols
    chunk         = _chunk_size(window_size, n_feat, chunk_mb)

    print(f"  [{split_name}] chunk_mb={chunk_mb} → {chunk:,} seqs/flush")
    print(f"  [{split_name}] Pass 1/2 — counting valid sequences...")
    n_valid = skipped_nan = skipped_short = 0

    for city, grp in df.groupby('City'):
        grp     = grp.sort_values('datetime').reset_index(drop=True)
        feats   = grp[feature_cols].values.astype(np.float32)
        targets = grp[target_col].values.astype(np.float32)
        n       = len(grp)
        if n < window_size + max_h:
            skipped_short += 1
            continue
        for i in range(window_size, n - max_h + 1):
            if np.isnan(feats[i-window_size:i, :n_signal_cols]).mean() > nan_threshold:
                skipped_nan += 1
                continue
            if all((i+h-1 < n) and not np.isnan(targets[i+h-1])
                   for h in forecast_horizons):
                n_valid += 1

    print(f"  [{split_name}] Valid={n_valid:,} | "
          f"skipped_nan={skipped_nan:,} | skipped_short={skipped_short}")

    if n_valid == 0:
        _write_empty_split(out_dir, split_name, window_size, n_feat, n_horizons)
        print(f"  [{split_name}] Empty split — skipping.")
        return AQIDataset(out_dir, split_name)

    mmap_X     = np.lib.format.open_memmap(
        str(out_dir / f'{split_name}_X.npy'),
        mode='w+', dtype=np.float32, shape=(n_valid, window_size, n_feat))
    mmap_y_reg = np.lib.format.open_memmap(
        str(out_dir / f'{split_name}_y_reg.npy'),
        mode='w+', dtype=np.float32, shape=(n_valid, n_horizons))
    mmap_y_cls = np.lib.format.open_memmap(
        str(out_dir / f'{split_name}_y_cls.npy'),
        mode='w+', dtype=np.int64, shape=(n_valid,))

    print(f"  [{split_name}] Pass 2/2 — writing in {chunk_mb} MB chunks...")
    buf_X     = np.empty((chunk, window_size, n_feat), dtype=np.float32)
    buf_y_reg = np.empty((chunk, n_horizons),          dtype=np.float32)
    buf_y_cls = np.empty((chunk,),                     dtype=np.int64)

    idx_out = buf_pos = 0
    meta    = []

    def _flush(buf_pos, idx_out):
        if buf_pos == 0:
            return idx_out
        mmap_X    [idx_out:idx_out+buf_pos] = buf_X    [:buf_pos]
        mmap_y_reg[idx_out:idx_out+buf_pos] = buf_y_reg[:buf_pos]
        mmap_y_cls[idx_out:idx_out+buf_pos] = buf_y_cls[:buf_pos]
        return idx_out + buf_pos

    for city, grp in df.groupby('City'):
        grp     = grp.sort_values('datetime').reset_index(drop=True)
        feats   = grp[feature_cols].values.astype(np.float32)
        targets = grp[target_col].values.astype(np.float32)
        labels  = grp[label_col].values.astype(np.int64)
        n       = len(grp)
        if n < window_size + max_h:
            continue
        for i in range(window_size, n - max_h + 1):
            window = feats[i-window_size:i]
            if np.isnan(window[:, :n_signal_cols]).mean() > nan_threshold:
                continue
            y_horizons, valid = [], True
            for h in forecast_horizons:
                tidx = i + h - 1
                if tidx >= n or np.isnan(targets[tidx]):
                    valid = False; break
                y_horizons.append(targets[tidx])
            if not valid:
                continue
            buf_X    [buf_pos] = np.nan_to_num(window, nan=0.0)
            buf_y_reg[buf_pos] = y_horizons
            buf_y_cls[buf_pos] = int(labels[i])
            meta.append({'city': city, 'datetime': grp['datetime'].iloc[i],
                         'city_idx': city_to_idx.get(city, 0) if city_to_idx else 0})
            buf_pos += 1
            if buf_pos == chunk:
                idx_out = _flush(buf_pos, idx_out)
                buf_pos = 0
                print(f"    [{split_name}] {idx_out:,}/{n_valid:,} written")

    idx_out = _flush(buf_pos, idx_out)
    del mmap_X, mmap_y_reg, mmap_y_cls

    joblib.dump(
        dict(n=n_valid, shape_X=(n_valid, window_size, n_feat),
             shape_yr=(n_valid, n_horizons), meta=meta),
        out_dir / f'{split_name}_info.joblib'
    )
    print(f"  [{split_name}] Done — {n_valid:,} sequences → {out_dir}")
    return AQIDataset(out_dir, split_name)


def load_split(data_processed_dir: Path, split: str) -> AQIDataset:
    return AQIDataset(Path(data_processed_dir), split)


# ══════════════════════════════════════════════════════════════════════════════
# MASTER PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  AQI-Sense India — Preprocessing Pipeline")
    print("=" * 60)

    raw = load_all_data(DATA_RAW)
    df  = merge_hourly(raw['city_hour'], raw['station_hour'])

    report_missing(df, 'before imputation')
    df = impute_dataframe(df, granularity='hourly')
    report_missing(df, 'after imputation')

    df = recompute_aqi(df)
    df = polygonalize_dataframe(df, granularity='hourly')
    df = feature_engineering(df)

    cities           = sorted(df['City'].dropna().unique().tolist())
    adj, city_to_idx = build_city_graph(cities)

    train_df, val_df, test_df = temporal_split(df)

    print("\n[9] Fitting scalers on training set...")
    feature_cols = get_feature_columns(train_df)
    feature_cols = [c for c in feature_cols
                    if c in train_df.columns and train_df[c].notna().sum() > 100]
    scalers  = fit_scalers(train_df, feature_cols)
    train_df = apply_scalers(train_df, scalers)
    val_df   = apply_scalers(val_df,   scalers)
    test_df  = apply_scalers(test_df,  scalers)
    print(f"  Feature columns: {len(feature_cols)}")

    print("\n[10] Building sliding-window sequences (500 MB chunks)...")
    kw = dict(
        feature_cols      = feature_cols,
        out_dir           = DATA_PROCESSED,
        target_col        = 'AQI_poly',
        label_col         = 'AQI_label',
        window_size       = WINDOW_SIZE,
        forecast_horizons = FORECAST_HOURS,
        city_to_idx       = city_to_idx,
        chunk_mb          = 500,
    )
    train_ds = build_sequences(train_df, split_name='train', **kw)
    val_ds   = build_sequences(val_df,   split_name='val',   **kw)
    test_ds  = build_sequences(test_df,  split_name='test',  **kw)

    print("\n[11] Saving metadata...")
    df.to_parquet(DATA_PROCESSED / 'hourly_clean.parquet', index=False)
    meta = {
        'feature_cols':   feature_cols,
        'city_to_idx':    city_to_idx,
        'adj':            adj,
        'n_features':     len(feature_cols),
        'forecast_hours': FORECAST_HOURS,
        'window_size':    WINDOW_SIZE,
        'n_train':        len(train_ds),
        'n_val':          len(val_ds),
        'n_test':         len(test_ds),
    }
    joblib.dump(meta,    DATA_PROCESSED / 'meta.joblib')
    joblib.dump(scalers, DATA_PROCESSED / 'scalers.joblib')

    print("\n" + "=" * 60)
    print("  Preprocessing complete.")
    print(f"  Features : {len(feature_cols)}")
    print(f"  Train    : {len(train_ds):,} sequences  (2015–2018)")
    print(f"  Val      : {len(val_ds):,} sequences  (2019)")
    print(f"  Test     : {len(test_ds):,} sequences  (2020)")
    print(f"  Saved to : {DATA_PROCESSED}")
    print("=" * 60)
    return meta


if __name__ == '__main__':
    run_pipeline()