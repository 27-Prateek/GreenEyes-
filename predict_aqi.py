"""
predict_aqi.py
==============
Interactive AQI predictor for GreenEyes+.

Two modes:
  1. MANUAL  — you type the current pollutant readings for any city
  2. LIVE    — fetches real-time air quality from OpenWeatherMap API
               (free API key from https://openweathermap.org/api)

Usage:
    python predict_aqi.py                        # interactive menu
    python predict_aqi.py --mode manual          # manual input only
    python predict_aqi.py --mode live            # live data only
    python predict_aqi.py --mode live --city Delhi --api_key YOUR_KEY

Requirements:
    pip install torch joblib numpy requests rich
"""

import sys
import json
import math
import argparse
import warnings
import datetime
from pathlib import Path

import numpy as np
import torch
import joblib
import requests

warnings.filterwarnings("ignore")

# ── Try rich for pretty output, fallback to plain print ──────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt, FloatPrompt
    from rich import print as rprint
    from rich.text import Text
    RICH = True
    console = Console()
except ImportError:
    RICH = False
    console = None

# ─────────────────────────────────────────────────────────────────────────────
# PATH SETUP — adjust ROOT_DIR if you move this script
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).parent
DATA_PROCESSED  = ROOT_DIR / "data" / "processed"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"

sys.path.append(str(ROOT_DIR))

# ─────────────────────────────────────────────────────────────────────────────
# AQI CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
AQI_CATEGORIES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
AQI_COLORS     = ["green", "yellow", "orange", "red", "purple", "maroon"]

AQI_EMOJI = {
    "Good":         "🟢",
    "Satisfactory": "🟡",
    "Moderate":     "🟠",
    "Poor":         "🔴",
    "Very Poor":    "🟣",
    "Severe":       "⚫",
}

AQI_ADVICE = {
    "Good":         "Air quality is satisfactory. Enjoy outdoor activities!",
    "Satisfactory": "Air quality is acceptable. Unusually sensitive people should consider limiting prolonged outdoor exertion.",
    "Moderate":     "Members of sensitive groups may experience health effects. General public is less likely to be affected.",
    "Poor":         "Everyone may begin to experience health effects. Sensitive groups should limit outdoor activity.",
    "Very Poor":    "Health warnings. Everyone should limit outdoor activity.",
    "Severe":       "Health alert — emergency conditions. Everyone should avoid outdoor activity.",
}

_PM25_BP = [
    (0.0,   12.0,   0,   50),
    (12.1,  35.4,  51,  100),
    (35.5,  55.4, 101,  150),
    (55.5, 150.4, 151,  200),
    (150.5,250.4, 201,  300),
    (250.5,500.4, 301,  500),
]

POLLUTANT_COLS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"
]

POLLUTANT_UNITS = {
    "PM2.5": "μg/m³", "PM10": "μg/m³",
    "NO":    "μg/m³", "NO2": "μg/m³", "NOx": "μg/m³",
    "NH3":   "μg/m³", "CO": "mg/m³",  "SO2": "μg/m³",
    "O3":    "μg/m³", "Benzene": "μg/m³",
    "Toluene": "μg/m³", "Xylene": "μg/m³",
}

TYPICAL_RANGES = {
    "PM2.5":   (0,   500), "PM10":    (0,   600),
    "NO":      (0,   200), "NO2":     (0,   200), "NOx": (0, 300),
    "NH3":     (0,   100), "CO":      (0,    10), "SO2": (0, 100),
    "O3":      (0,   200), "Benzene": (0,    50),
    "Toluene": (0,   100), "Xylene":  (0,   100),
}

# Typical Indian city values (used as defaults in manual mode)
TYPICAL_DEFAULTS = {
    "PM2.5": 60.0, "PM10": 110.0, "NO": 15.0,  "NO2": 30.0,
    "NOx":   42.0, "NH3":  12.0,  "CO":  1.2,  "SO2": 14.0,
    "O3":    35.0, "Benzene": 3.0, "Toluene": 8.0, "Xylene": 4.0
}

CITY_COORDS = {
    "Ahmedabad":         (23.0225, 72.5714), "Aizawl":       (23.7271, 92.7176),
    "Amaravati":         (16.5728, 80.3582), "Amritsar":     (31.6340, 74.8723),
    "Bengaluru":         (12.9716, 77.5946), "Bhopal":       (23.2599, 77.4126),
    "Brajrajnagar":      (21.8253, 83.9175), "Chandigarh":   (30.7333, 76.7794),
    "Chennai":           (13.0827, 80.2707), "Coimbatore":   (11.0168, 76.9558),
    "Delhi":             (28.7041, 77.1025), "Ernakulam":    (9.9816,  76.2999),
    "Gurugram":          (28.4595, 77.0266), "Guwahati":     (26.1445, 91.7362),
    "Hyderabad":         (17.3850, 78.4867), "Jaipur":       (26.9124, 75.7873),
    "Jorapokhar":        (23.7271, 86.4048), "Kochi":        (9.9312,  76.2673),
    "Kolkata":           (22.5726, 88.3639), "Lucknow":      (26.8467, 80.9462),
    "Mumbai":            (19.0760, 72.8777), "Patna":        (25.5941, 85.1376),
    "Shillong":          (25.5788, 91.8933), "Talcher":      (20.9500, 85.2333),
    "Thiruvananthapuram":(8.5241,  76.9366), "Visakhapatnam":(17.6868, 83.2185),
}

HORIZON_LABELS = {1: "t+1h", 24: "t+24h", 72: "t+72h"}


# ─────────────────────────────────────────────────────────────────────────────
# PRINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def pprint(msg, style=""):
    if RICH:
        console.print(msg, style=style)
    else:
        print(msg)

def header(title):
    if RICH:
        console.rule(f"[bold cyan]{title}[/bold cyan]")
    else:
        print(f"\n{'='*60}\n  {title}\n{'='*60}")

def success(msg):
    pprint(f"✅  {msg}", "bold green")

def warn(msg):
    pprint(f"⚠️   {msg}", "bold yellow")

def error(msg):
    pprint(f"❌  {msg}", "bold red")


# ─────────────────────────────────────────────────────────────────────────────
# AQI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def pm25_to_aqi(pm25: float) -> float:
    if np.isnan(pm25) or pm25 < 0:
        return float("nan")
    for lo, hi, alo, ahi in _PM25_BP:
        if lo <= pm25 <= hi:
            return (ahi - alo) / (hi - lo) * (pm25 - lo) + alo
    return 500.0

def aqi_to_category(aqi: float) -> str:
    if np.isnan(aqi): return "Unknown"
    if aqi <= 50:     return "Good"
    if aqi <= 100:    return "Satisfactory"
    if aqi <= 200:    return "Moderate"
    if aqi <= 300:    return "Poor"
    if aqi <= 400:    return "Very Poor"
    return "Severe"

def aqi_to_label(aqi: float) -> int:
    cat = aqi_to_category(aqi)
    mapping = {"Good": 0, "Satisfactory": 1, "Moderate": 2,
               "Poor": 3, "Very Poor": 4, "Severe": 5}
    return mapping.get(cat, -1)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING (mirrors preprocess.py)
# ─────────────────────────────────────────────────────────────────────────────
def add_temporal_features(dt: datetime.datetime) -> dict:
    feats = {}
    feats["hour_sin"]       = math.sin(2 * math.pi * dt.hour      / 24)
    feats["hour_cos"]       = math.cos(2 * math.pi * dt.hour      / 24)
    feats["dow_sin"]        = math.sin(2 * math.pi * dt.weekday() / 7)
    feats["dow_cos"]        = math.cos(2 * math.pi * dt.weekday() / 7)
    feats["month_sin"]      = math.sin(2 * math.pi * dt.month     / 12)
    feats["month_cos"]      = math.cos(2 * math.pi * dt.month     / 12)
    feats["doy_sin"]        = math.sin(2 * math.pi * dt.timetuple().tm_yday / 365)
    feats["doy_cos"]        = math.cos(2 * math.pi * dt.timetuple().tm_yday / 365)
    feats["is_weekend"]     = int(dt.weekday() >= 5)
    feats["is_festive"]     = int(dt.month in [10, 11])
    feats["is_winter_smog"] = int(dt.month in [11, 12, 1, 2])
    feats["is_monsoon"]     = int(dt.month in [6, 7, 8, 9])
    return feats

def add_composite_features(poll: dict) -> dict:
    feats = {}
    pm25 = poll.get("PM2.5", 0) or 0
    pm10 = poll.get("PM10",  0) or 0
    no   = poll.get("NO",    0) or 0
    no2  = poll.get("NO2",   0) or 0
    o3   = poll.get("O3",    0) or 0
    benz = poll.get("Benzene",  0) or 0
    tol  = poll.get("Toluene",  0) or 0

    feats["PM_ratio"]    = np.clip(pm25 / (pm10 + 1e-6), 0, 1)
    feats["PM_total"]    = pm25 + pm10
    feats["oxidant_idx"] = o3 + no2
    feats["NOx_ratio"]   = np.clip(no / (no2 + 1e-6), 0, 10)
    feats["BT_ratio"]    = np.clip(benz / (tol + 1e-6), 0, 10)
    feats["n_sensors_ok"]= sum(1 for p in POLLUTANT_COLS if poll.get(p) is not None)
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC WINDOW BUILDER
# Builds a 168-step window from a single snapshot by adding realistic noise
# The model expects 7 days of hourly history — we synthesise it from now.
# ─────────────────────────────────────────────────────────────────────────────
def build_window(
    poll_values: dict,
    aqi_now: float,
    city_name: str,
    meta: dict,
    scalers: dict,
    window_size: int = 168,
) -> tuple:
    """
    Creates a (168, n_features) window from a single pollutant snapshot.
    Steps back in time by adding small Gaussian noise around the snapshot values
    to simulate a plausible 7-day history.
    """
    feature_cols = meta["feature_cols"]
    n_feat       = len(feature_cols)
    now          = datetime.datetime.now()

    # Build 168 historical rows (oldest → newest)
    rows = []
    for step in range(window_size):
        dt   = now - datetime.timedelta(hours=(window_size - 1 - step))
        frac = step / window_size  # 0 = oldest, 1 = newest

        row = {}

        # Pollutants: add decaying noise (more noise further back)
        noise_scale = 0.15 * (1 - frac * 0.5)
        for p in POLLUTANT_COLS:
            base = poll_values.get(p, 0.0) or 0.0
            noise = np.random.normal(0, noise_scale * (base + 1e-3))
            row[p] = max(0.0, base + noise)
            row[f"{p}_mask"] = 0  # fully observed

        # AQI_poly: smooth version of AQI
        aqi_noise = np.random.normal(0, 5 * (1 - frac * 0.5))
        aqi_step  = max(0.0, aqi_now + aqi_noise)
        row["AQI_poly"] = aqi_step

        # Temporal features at this step's datetime
        row.update(add_temporal_features(dt))

        # Composite features
        row.update(add_composite_features(row))

        rows.append(row)

    # Add lag/rolling features using the synthetic history
    aqi_history = [r["AQI_poly"] for r in rows]
    for step, row in enumerate(rows):
        for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
            idx = step - lag
            row[f"AQI_poly_lag_{lag}h"] = aqi_history[idx] if idx >= 0 else aqi_history[0]
        for w in [6, 12, 24, 48, 72, 168]:
            hist_slice = aqi_history[max(0, step - w): step]
            if hist_slice:
                row[f"AQI_poly_rmean_{w}h"] = float(np.mean(hist_slice))
                row[f"AQI_poly_rstd_{w}h"]  = float(np.std(hist_slice)) if len(hist_slice) > 1 else 0.0
            else:
                row[f"AQI_poly_rmean_{w}h"] = aqi_now
                row[f"AQI_poly_rstd_{w}h"]  = 0.0
        row["AQI_poly_diff_1h"]  = (aqi_history[step] - aqi_history[step - 1]) if step >= 1 else 0.0
        row["AQI_poly_diff_24h"] = (aqi_history[step] - aqi_history[step - 24]) if step >= 24 else 0.0

    # Convert to numpy matrix aligned to feature_cols
    mat = np.zeros((window_size, n_feat), dtype=np.float32)
    for i, col in enumerate(feature_cols):
        for t, row in enumerate(rows):
            mat[t, i] = float(row.get(col, 0.0) or 0.0)

    # Apply scalers (same as training)
    for i, col in enumerate(feature_cols):
        if col.endswith("_mask"):
            continue
        if col in scalers:
            sc  = scalers[col]
            mat[:, i] = sc.transform(mat[:, i].reshape(-1, 1)).flatten()

    city_to_idx = meta.get("city_to_idx", {})
    city_idx    = city_to_idx.get(city_name, 0)

    return mat, city_idx


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_model_and_meta(model_name: str = "greeneyes"):
    try:
        from configs.config import (
            N_CITIES, HIDDEN_DIM, WAVENET_LAYERS, LSTM_LAYERS,
            KERNEL_SIZE, LSTM_DROPOUT, GNN_HEADS, N_CATEGORIES, DROPOUT
        )
        from models.model import GreenEyesPlus
    except ImportError as e:
        error(f"Cannot import project modules: {e}")
        error("Make sure you run this script from the project root directory.")
        sys.exit(1)

    ckpt_path = CHECKPOINTS_DIR / model_name / "best_model.pt"
    if not ckpt_path.exists():
        error(f"No checkpoint found at {ckpt_path}")
        sys.exit(1)

    meta_path    = DATA_PROCESSED / "meta.joblib"
    scalers_path = DATA_PROCESSED / "scalers.joblib"
    if not meta_path.exists():
        error(f"meta.joblib not found at {meta_path}")
        sys.exit(1)

    meta    = joblib.load(meta_path)
    scalers = joblib.load(scalers_path) if scalers_path.exists() else {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)

    n_feat   = meta["n_features"]
    horizons = meta["forecast_hours"]

    model = GreenEyesPlus(
        n_features    = n_feat,
        n_cities      = N_CITIES,
        hidden        = HIDDEN_DIM,
        wavenet_layers= WAVENET_LAYERS,
        lstm_layers   = LSTM_LAYERS,
        horizons      = horizons,
        dropout       = DROPOUT,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    adj = torch.tensor(meta["adj"], dtype=torch.float32).to(device)

    return model, meta, scalers, adj, device, horizons


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model, window_mat, city_idx_val, adj, device, scalers, meta):
    X        = torch.tensor(window_mat, dtype=torch.float32).unsqueeze(0).to(device)
    city_idx = torch.tensor([city_idx_val], dtype=torch.long).to(device)

    reg_preds, cat_logits = model(X, adj=adj, city_idx=city_idx)

    horizons = meta["forecast_hours"]
    results  = {}

    # Inverse-transform regression predictions
    if "AQI_poly" in scalers:
        sc = scalers["AQI_poly"]
        for i, h in enumerate(horizons):
            val = reg_preds[i].squeeze().item()
            aqi = sc.inverse_transform([[val]])[0][0]
            aqi = float(np.clip(aqi, 0, 500))
            results[h] = {
                "aqi":      round(aqi, 1),
                "category": aqi_to_category(aqi),
            }
    else:
        for i, h in enumerate(horizons):
            val = reg_preds[i].squeeze().item()
            aqi = float(np.clip(val, 0, 500))
            results[h] = {
                "aqi":      round(aqi, 1),
                "category": aqi_to_category(aqi),
            }

    probs    = torch.softmax(cat_logits, dim=1).squeeze().cpu().numpy()
    pred_cat = int(torch.argmax(cat_logits, dim=1).item())

    return results, probs, pred_cat


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY RESULTS
# ─────────────────────────────────────────────────────────────────────────────
def display_results(
    results: dict,
    probs:   np.ndarray,
    pred_cat:int,
    city:    str,
    aqi_now: float,
    poll_values: dict,
    source:  str = "Manual Input",
):
    header(f"GreenEyes+ Prediction — {city}")

    now_cat   = aqi_to_category(aqi_now)
    now_emoji = AQI_EMOJI.get(now_cat, "")
    now_label = AQI_CATEGORIES[pred_cat]

    print(f"\n📍 City        : {city}")
    print(f"🕐 Time        : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"📡 Data source : {source}")
    print(f"🌫️  Current AQI  : {aqi_now:.1f}  →  {now_emoji} {now_cat}")
    print()

    # ── Forecast table ──────────────────────────────────────────────────
    if RICH:
        t = Table(title="[bold]AQI Forecast[/bold]", border_style="cyan")
        t.add_column("Horizon",  style="bold", width=12)
        t.add_column("Pred AQI", justify="center", width=12)
        t.add_column("Category", justify="center", width=16)
        t.add_column("Advice",   width=46)
        for h in sorted(results):
            r     = results[h]
            emoji = AQI_EMOJI.get(r["category"], "")
            advice= AQI_ADVICE.get(r["category"], "")[:45]
            lbl   = HORIZON_LABELS.get(h, f"t+{h}h")
            t.add_row(lbl, str(r["aqi"]), f"{emoji} {r['category']}", advice)
        console.print(t)
    else:
        print(f"{'Horizon':<10} {'Pred AQI':>10} {'Category':<16} Advice")
        print("-" * 75)
        for h in sorted(results):
            r     = results[h]
            emoji = AQI_EMOJI.get(r["category"], "")
            lbl   = HORIZON_LABELS.get(h, f"t+{h}h")
            print(f"{lbl:<10} {r['aqi']:>10.1f} {r['category']:<16} {AQI_ADVICE.get(r['category'],'')[:30]}")

    # ── Category probabilities ───────────────────────────────────────────
    print()
    if RICH:
        t2 = Table(title="[bold]Current Category Probabilities[/bold]", border_style="blue")
        t2.add_column("Category", width=18)
        t2.add_column("Probability", justify="right", width=14)
        t2.add_column("Bar", width=30)
        for i, (cat, prob) in enumerate(zip(AQI_CATEGORIES, probs)):
            bar   = "█" * int(prob * 30)
            emoji = AQI_EMOJI.get(cat, "")
            mark  = " ◀ predicted" if i == pred_cat else ""
            t2.add_row(f"{emoji} {cat}", f"{prob*100:.1f}%", f"{bar}{mark}")
        console.print(t2)
    else:
        print("Current Category Probabilities:")
        for i, (cat, prob) in enumerate(zip(AQI_CATEGORIES, probs)):
            bar  = "█" * int(prob * 30)
            mark = " <-- predicted" if i == pred_cat else ""
            print(f"  {cat:<16} {prob*100:5.1f}%  {bar}{mark}")

    # ── Input pollutants used ────────────────────────────────────────────
    print()
    if RICH:
        t3 = Table(title="[bold]Pollutant Inputs Used[/bold]", border_style="dim")
        t3.add_column("Pollutant", width=12)
        t3.add_column("Value",     justify="right", width=10)
        t3.add_column("Unit",      width=10)
        for p in POLLUTANT_COLS:
            val = poll_values.get(p, 0.0)
            t3.add_row(p, f"{val:.2f}", POLLUTANT_UNITS.get(p, ""))
        console.print(t3)
    else:
        print("Pollutant inputs:")
        for p in POLLUTANT_COLS:
            print(f"  {p:<12} {poll_values.get(p, 0.0):8.2f}  {POLLUTANT_UNITS.get(p,'')}")

    print()
    dominant = sorted(results.items(), key=lambda x: x[1]["aqi"], reverse=True)[0]
    worst_cat = dominant[1]["category"]
    if worst_cat in ("Poor", "Very Poor", "Severe"):
        warn(f"72h forecast: AQI may reach {dominant[1]['aqi']} ({worst_cat}). Health precautions advised.")
    elif worst_cat == "Good":
        success("All forecasts look clean. Good air quality expected!")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MODE 1 — MANUAL INPUT
# ─────────────────────────────────────────────────────────────────────────────
def mode_manual(model, meta, scalers, adj, device, horizons):
    header("Manual Input Mode")
    print("Enter current pollutant concentrations for your location.")
    print("Press ENTER to use the default typical Indian city value.\n")

    # City selection
    known_cities = sorted(CITY_COORDS.keys())
    print("Known cities (type exact name or any custom name):")
    for i, c in enumerate(known_cities):
        end = "\n" if (i + 1) % 4 == 0 else "  "
        print(f"  {c}", end=end)
    print()

    if RICH:
        city = Prompt.ask("City name", default="Delhi")
    else:
        city = input("City name [Delhi]: ").strip() or "Delhi"

    # Pollutant inputs
    poll_values = {}
    print()
    for p in POLLUTANT_COLS:
        default = TYPICAL_DEFAULTS.get(p, 0.0)
        lo, hi  = TYPICAL_RANGES.get(p, (0, 500))
        unit    = POLLUTANT_UNITS.get(p, "")
        while True:
            if RICH:
                raw = Prompt.ask(
                    f"  {p:<12} [{unit}]  (typical {lo}–{hi})",
                    default=str(default)
                )
            else:
                raw = input(f"  {p} [{unit}] (default {default}): ").strip()
                if not raw:
                    raw = str(default)
            try:
                val = float(raw)
                poll_values[p] = val
                break
            except ValueError:
                print("  Please enter a number.")

    # Compute current AQI from PM2.5
    aqi_now = pm25_to_aqi(poll_values.get("PM2.5", 60.0))
    print(f"\n  Computed current AQI from PM2.5: {aqi_now:.1f}  ({aqi_to_category(aqi_now)})")

    print("\n⚙️  Building synthetic 168-hour window and running inference...")
    window_mat, city_idx = build_window(poll_values, aqi_now, city, meta, scalers)
    results, probs, pred_cat = predict(model, window_mat, city_idx, adj, device, scalers, meta)

    display_results(results, probs, pred_cat, city, aqi_now, poll_values, source="Manual Input")


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2 — LIVE DATA  (OpenWeatherMap Air Pollution API)
# ─────────────────────────────────────────────────────────────────────────────
OWM_AQI_MAP = {1: "Good", 2: "Satisfactory", 3: "Moderate", 4: "Poor", 5: "Very Poor"}

def fetch_live_data(city_name: str, api_key: str) -> dict:
    """
    Uses OpenWeatherMap:
      1. Geocoding API  → lat/lon for city
      2. Air Pollution API → real-time pollutant components
    Free tier: 60 calls/min, no credit card needed.
    Get key at: https://home.openweathermap.org/api_keys
    """
    session = requests.Session()
    session.headers.update({"User-Agent": "GreenEyesAQI/1.0"})

    # Step 1: Geocode
    geo_url = (
        f"https://api.openweathermap.org/geo/1.0/direct"
        f"?q={city_name},IN&limit=1&appid={api_key}"
    )
    try:
        geo_r = session.get(geo_url, timeout=30, verify=True)
        geo_r.raise_for_status()
        geo_data = geo_r.json()
    except requests.exceptions.SSLError:
        # Fallback: skip SSL verification (some corporate networks intercept SSL)
        try:
            geo_r = session.get(geo_url, timeout=30, verify=False)
            geo_r.raise_for_status()
            geo_data = geo_r.json()
        except Exception as e:
            raise RuntimeError(f"Geocoding failed (SSL error): {e}")
    except requests.exceptions.ConnectTimeout:
        raise RuntimeError(
            "Connection timed out. Possible causes:\n"
            "  1. No internet connection\n"
            "  2. Firewall/VPN blocking the request\n"
            "  3. New API key not yet activated (wait 10 min after creating key)\n"
            "Try: ping api.openweathermap.org in a terminal to check connectivity."
        )
    except Exception as e:
        raise RuntimeError(f"Geocoding failed: {e}")

    if not geo_data:
        # Try without country code
        geo_url2 = (
            f"https://api.openweathermap.org/geo/1.0/direct"
            f"?q={city_name}&limit=1&appid={api_key}"
        )
        geo_r = requests.get(geo_url2, timeout=30)
        geo_data = geo_r.json()
        if not geo_data:
            raise RuntimeError(
                f"City '{city_name}' not found. Try a different spelling or use --mode manual."
            )

    lat = geo_data[0]["lat"]
    lon = geo_data[0]["lon"]
    found_name = geo_data[0].get("name", city_name)

    # Step 2: Air pollution
    ap_url = (
        f"https://api.openweathermap.org/data/2.5/air_pollution"
        f"?lat={lat}&lon={lon}&appid={api_key}"
    )
    try:
        ap_r = requests.get(ap_url, timeout=30)
        ap_r.raise_for_status()
        ap_data = ap_r.json()
    except Exception as e:
        raise RuntimeError(f"Air pollution API failed: {e}")

    comp = ap_data["list"][0]["components"]
    owm_aqi = ap_data["list"][0]["main"]["aqi"]

    # Map OWM component names → our pollutant names
    # OWM provides: co (μg/m³), no (μg/m³), no2 (μg/m³), o3 (μg/m³),
    #               so2 (μg/m³), pm2_5 (μg/m³), pm10 (μg/m³), nh3 (μg/m³)
    poll = {
        "PM2.5":   comp.get("pm2_5",  60.0),
        "PM10":    comp.get("pm10",   100.0),
        "NO":      comp.get("no",      15.0),
        "NO2":     comp.get("no2",     30.0),
        "NOx":     comp.get("no", 0) + comp.get("no2", 0),
        "NH3":     comp.get("nh3",     12.0),
        "CO":      comp.get("co", 500) / 1000,  # OWM gives CO in μg/m³ → convert to mg/m³
        "SO2":     comp.get("so2",     14.0),
        "O3":      comp.get("o3",      35.0),
        # OWM doesn't provide VOCs — use typical values
        "Benzene": 3.0,
        "Toluene": 8.0,
        "Xylene":  4.0,
    }

    return {
        "poll":       poll,
        "lat":        lat,
        "lon":        lon,
        "found_name": found_name,
        "owm_aqi":    owm_aqi,
    }


def mode_live(model, meta, scalers, adj, device, horizons, city_name: str, api_key: str):
    header("Live Data Mode — OpenWeatherMap")

    if not api_key:
        if RICH:
            api_key = Prompt.ask(
                "Enter your OpenWeatherMap API key\n"
                "  (Get free key at https://openweathermap.org/api)",
            )
        else:
            print("Enter your OpenWeatherMap API key")
            print("  (Get free key at https://openweathermap.org/api)")
            api_key = input("API key: ").strip()

    if not city_name:
        if RICH:
            city_name = Prompt.ask("City name", default="Delhi")
        else:
            city_name = input("City name [Delhi]: ").strip() or "Delhi"

    print(f"\n🌐 Fetching live air quality data for '{city_name}'...")

    try:
        live = fetch_live_data(city_name, api_key)
    except RuntimeError as e:
        error(str(e))
        sys.exit(1)

    poll_values = live["poll"]
    found_name  = live["found_name"]
    success(f"Data fetched for: {found_name} (lat={live['lat']:.3f}, lon={live['lon']:.3f})")
    success(f"OpenWeatherMap AQI index: {live['owm_aqi']} ({OWM_AQI_MAP.get(live['owm_aqi'], '?')})")

    aqi_now = pm25_to_aqi(poll_values["PM2.5"])
    print(f"  PM2.5 = {poll_values['PM2.5']:.1f} μg/m³  →  AQI = {aqi_now:.1f} ({aqi_to_category(aqi_now)})")

    # Match to known city for graph lookup
    city_match = city_name
    for known in CITY_COORDS:
        if known.lower() in city_name.lower() or city_name.lower() in known.lower():
            city_match = known
            break

    print(f"\n⚙️  Building synthetic 168-hour window and running inference...")
    window_mat, city_idx = build_window(poll_values, aqi_now, city_match, meta, scalers)
    results, probs, pred_cat = predict(model, window_mat, city_idx, adj, device, scalers, meta)

    display_results(
        results, probs, pred_cat, found_name, aqi_now, poll_values,
        source="OpenWeatherMap Live API"
    )


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE MENU
# ─────────────────────────────────────────────────────────────────────────────
def interactive_menu(model, meta, scalers, adj, device, horizons):
    if RICH:
        console.print(Panel.fit(
            "[bold cyan]GreenEyes+ AQI Predictor[/bold cyan]\n"
            "[dim]Multi-horizon Air Quality Forecasting for India[/dim]",
            border_style="cyan"
        ))
    else:
        print("\n" + "="*50)
        print("  GreenEyes+ AQI Predictor")
        print("  Multi-horizon Air Quality Forecasting")
        print("="*50)

    while True:
        print("\nSelect mode:")
        print("  [1] Manual Input  — enter pollutant values yourself")
        print("  [2] Live Data     — fetch real-time data from OpenWeatherMap API")
        print("  [3] Quick Test    — predict for Delhi with typical winter values")
        print("  [q] Quit")
        print()

        if RICH:
            choice = Prompt.ask("Choice", choices=["1", "2", "3", "q"], default="1")
        else:
            choice = input("Choice [1/2/3/q]: ").strip() or "1"

        if choice == "1":
            mode_manual(model, meta, scalers, adj, device, horizons)
        elif choice == "2":
            mode_live(model, meta, scalers, adj, device, horizons, city_name="", api_key="")
        elif choice == "3":
            quick_test(model, meta, scalers, adj, device, horizons)
        elif choice.lower() == "q":
            print("Goodbye!")
            break
        else:
            warn("Invalid choice. Please enter 1, 2, 3, or q.")

        if RICH:
            again = Prompt.ask("\nRun another prediction?", choices=["y", "n"], default="y")
        else:
            again = input("\nRun another prediction? [y/n]: ").strip().lower() or "y"
        if again != "y":
            print("Goodbye!")
            break


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST — Delhi winter pollution scenario
# ─────────────────────────────────────────────────────────────────────────────
def quick_test(model, meta, scalers, adj, device, horizons):
    header("Quick Test — Delhi Winter Smog Scenario")
    poll_values = {
        "PM2.5": 185.0, "PM10": 310.0, "NO": 45.0,  "NO2": 85.0,
        "NOx":   125.0, "NH3":  28.0,  "CO":  3.5,  "SO2": 32.0,
        "O3":    18.0,  "Benzene": 8.5, "Toluene": 22.0, "Xylene": 12.0
    }
    aqi_now = pm25_to_aqi(poll_values["PM2.5"])
    print(f"\nUsing Delhi winter scenario: PM2.5={poll_values['PM2.5']} μg/m³ → AQI={aqi_now:.1f}")
    print("Building window and running inference...")
    window_mat, city_idx = build_window(poll_values, aqi_now, "Delhi", meta, scalers)
    results, probs, pred_cat = predict(model, window_mat, city_idx, adj, device, scalers, meta)
    display_results(results, probs, pred_cat, "Delhi", aqi_now, poll_values, source="Quick Test (Winter Scenario)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="GreenEyes+ AQI Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_aqi.py                                   # interactive menu
  python predict_aqi.py --mode manual                     # manual pollutant entry
  python predict_aqi.py --mode live --city Mumbai         # live data for Mumbai (will ask for API key)
  python predict_aqi.py --mode live --city Delhi --api_key abc123
  python predict_aqi.py --mode test                       # Delhi winter quick test
        """
    )
    parser.add_argument("--mode",    choices=["manual", "live", "test", "menu"], default="menu")
    parser.add_argument("--city",    default="",  help="City name for live mode")
    parser.add_argument("--api_key", default="",  help="OpenWeatherMap API key for live mode")
    parser.add_argument("--model",   default="greeneyes", help="Model checkpoint name")
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────
    print(f"\n⏳ Loading GreenEyes+ checkpoint '{args.model}'...")
    model, meta, scalers, adj, device, horizons = load_model_and_meta(args.model)
    dev_name = "CUDA" if device.type == "cuda" else "CPU"
    success(f"Model loaded on {dev_name}  |  Features: {meta['n_features']}  |  Horizons: {horizons}")

    np.random.seed(42)  # reproducible synthetic window noise

    if args.mode == "manual":
        mode_manual(model, meta, scalers, adj, device, horizons)
    elif args.mode == "live":
        mode_live(model, meta, scalers, adj, device, horizons, args.city, args.api_key)
    elif args.mode == "test":
        quick_test(model, meta, scalers, adj, device, horizons)
    else:
        interactive_menu(model, meta, scalers, adj, device, horizons)


if __name__ == "__main__":
    main()