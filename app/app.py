from flask import Flask, render_template, request, jsonify
import joblib
import json
from pathlib import Path
import pandas as pd
from datetime import datetime

app = Flask(__name__, template_folder="templates", static_folder="static")

# --- Paths ---
# app.py lives at: Smart-Agriculture/app/app.py
# .parent       -> Smart-Agriculture/app/
# .parent.parent-> Smart-Agriculture/
APP_DIR  = Path(__file__).resolve().parent          # Smart-Agriculture/app/
BASE_DIR = APP_DIR.parent                           # Smart-Agriculture/

MODEL_PATH  = BASE_DIR / "notebooks" / "models" / "model.pkl"
CONFIG_PATH = BASE_DIR / "notebooks" / "models" / "config.json"

# --- Debug output (safe to leave in; harmless in production) ---
print("=" * 60)
print(f"APP_DIR     : {APP_DIR}")
print(f"BASE_DIR    : {BASE_DIR}")
print(f"MODEL_PATH  : {MODEL_PATH}")
print(f"Model found : {MODEL_PATH.exists()}")
print("=" * 60)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"\n[ERROR] Model file not found at:\n  {MODEL_PATH}\n"
        f"Make sure you are running app.py from inside Smart-Agriculture/app/\n"
        f"  cd Smart-Agriculture/app\n"
        f"  python app.py\n"
    )

# Load full pipeline (preprocessor + model); pipeline.predict(X_raw) expects raw feature DataFrame
pipeline = joblib.load(MODEL_PATH)
print("[OK] Pipeline loaded successfully.")

# --- Load config: feature order, threshold ---
FEATURE_ORDER = None
THRESHOLD = 0.9
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
        FEATURE_ORDER = cfg.get("features")
        THRESHOLD = float(cfg.get("threshold", 0.9))
    print(f"[OK] Config loaded. Feature order: {FEATURE_ORDER}, threshold: {THRESHOLD}")
else:
    print(f"[WARN] No config found at {CONFIG_PATH}. Using default feature order and threshold.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def _sensors_to_row(sensors: dict, ts: pd.Timestamp) -> dict:
    """Build one feature row from sensors dict and timestamp. Uses nutrient_index (no N,P,K columns)."""
    t = safe_float(sensors.get("tempreature") or sensors.get("temperature"), 0.0)
    h = safe_float(sensors.get("humidity"), 0.0)
    w = safe_float(sensors.get("water_level"), 0.0)
    n = safe_float(sensors.get("N"), 0.0)
    p = safe_float(sensors.get("P"), 0.0)
    k = safe_float(sensors.get("K"), 0.0)
    nutrient_index = (n + p + k) / 3.0
    return {
        "tempreature": t,
        "humidity": h,
        "water_level": w,
        "month": ts.month,
        "hour": ts.hour,
        "day_of_week": ts.dayofweek,
        "is_weekend": 1 if ts.dayofweek >= 5 else 0,
        "nutrient_index": nutrient_index,
    }


def build_df_from_payload(timestamp: str, sensors: dict) -> pd.DataFrame:
    """Build a one-row DataFrame for 'what to do now' from timestamp and sensors."""
    ts = pd.to_datetime(timestamp)
    row = _sensors_to_row(sensors, ts)
    df = pd.DataFrame([row])
    if FEATURE_ORDER:
        df = df[[c for c in FEATURE_ORDER if c in df.columns]]
    return df


def build_df_from_rows(rows: list[dict]) -> pd.DataFrame:
    """
    Convert a list of row dicts into a DataFrame for pipeline.predict().
    Each row can have timestamp + sensors, or explicit hour/day_of_week/month + sensors.
    Uses nutrient_index (from N,P,K); no N,P,K columns. No humidity_temp_ratio.
    """
    processed = []
    for r in rows:
        ts = pd.to_datetime(r.get("timestamp", "2024-01-01 12:00:00"))
        sensors = r.get("sensors", r)
        row = _sensors_to_row(sensors, ts)
        # Override with explicit time fields if provided
        if "month" in r:
            row["month"] = safe_float(r["month"], row["month"])
        if "hour" in r:
            row["hour"] = safe_float(r["hour"], row["hour"])
        if "day_of_week" in r:
            row["day_of_week"] = safe_float(r["day_of_week"], row["day_of_week"])
            row["is_weekend"] = 1 if row["day_of_week"] >= 5 else 0
        processed.append(row)

    df = pd.DataFrame(processed)
    if FEATURE_ORDER:
        df = df[[c for c in FEATURE_ORDER if c in df.columns]]
    return df


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON body (what to do now):
    {
      "timestamp": "2024-06-15T14:30:00",
      "sensors": { "temperature": 28, "humidity": 65, "water_level": 90, "N": 200, "P": 210, "K": 220 }
    }
    Or legacy: { "rows": [ { "timestamp": "...", "sensors": {...} }, ... ] }

    Returns:
    {
      "predictions": [
        { "irrigation": { "label": "YES", "probability": 0.95 }, "fan": { "label": "NO", "probability": 0.1 } }
      ]
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid payload. Expected JSON."}), 400

    try:
        if "timestamp" in data and "sensors" in data:
            input_df = build_df_from_payload(data["timestamp"], data["sensors"])
        elif "rows" in data:
            rows = data["rows"]
            if not isinstance(rows, list) or len(rows) == 0:
                return jsonify({"error": "'rows' must be a non-empty list."}), 400
            input_df = build_df_from_rows(rows)
        else:
            return jsonify({"error": "Expected 'timestamp' and 'sensors', or 'rows'."}), 400

        # Pipeline: predict_proba returns list of (n, 2) arrays per target
        probas = pipeline.predict_proba(input_df)

        predictions = []
        for i in range(len(input_df)):
            irrigation_prob = float(probas[0][i][1]) if probas[0] is not None else None
            fan_prob = float(probas[1][i][1]) if len(probas) > 1 and probas[1] is not None else None
            irrigation_on = irrigation_prob is not None and irrigation_prob >= THRESHOLD
            fan_on = fan_prob is not None and fan_prob >= THRESHOLD
            predictions.append({
                "irrigation": {
                    "label": "YES" if irrigation_on else "NO",
                    "probability": irrigation_prob,
                },
                "fan": {
                    "label": "YES" if fan_on else "NO",
                    "probability": fan_prob,
                },
            })

        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/schedule", methods=["POST"])
def schedule():
    """
    Expects JSON body:
    {
      "start_time": "2024-06-15T00:00:00",
      "sensors": { "temperature": 28, "humidity": 65, "water_level": 90, "N": 200, "P": 210, "K": 220 },
      "interval_minutes": 60
    }
    Returns list of { "time": "<iso>", "irrigation": 0|1, "fan": 0|1 } for next 24h.
    """
    data = request.get_json(silent=True)
    if not data or "start_time" not in data or "sensors" not in data:
        return jsonify({"error": "Expected JSON with 'start_time' and 'sensors'."}), 400

    start = pd.to_datetime(data["start_time"])
    sensors = data["sensors"]
    interval_minutes = int(data.get("interval_minutes", 60))

    try:
        slots = []
        t = start
        end = start + pd.Timedelta(hours=24)
        while t < end:
            row = _sensors_to_row(sensors, t)
            df = pd.DataFrame([row])
            if FEATURE_ORDER:
                df = df[[c for c in FEATURE_ORDER if c in df.columns]]
            probas = pipeline.predict_proba(df)
            irr_p = float(probas[0][0][1]) if probas[0] is not None else 0.0
            fan_p = float(probas[1][0][1]) if len(probas) > 1 and probas[1] is not None else 0.0
            slots.append({
                "time": t.isoformat(),
                "irrigation": 1 if irr_p >= THRESHOLD else 0,
                "fan": 1 if fan_p >= THRESHOLD else 0,
            })
            t = t + pd.Timedelta(minutes=interval_minutes)

        return jsonify({"schedule": slots})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run from Smart-Agriculture/app/:
    #   cd Smart-Agriculture/app
    #   python app.py
    app.run(host="0.0.0.0", port=5000, debug=True)