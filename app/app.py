from flask import Flask, render_template, request, jsonify
import joblib
import json
from pathlib import Path
import pandas as pd

app = Flask(__name__, template_folder="templates", static_folder="static")

# --- Paths ---
# app.py lives at: Smart-Agriculture/app/app.py
# .parent       -> Smart-Agriculture/app/
# .parent.parent-> Smart-Agriculture/
APP_DIR  = Path(__file__).resolve().parent          # Smart-Agriculture/app/
BASE_DIR = APP_DIR.parent                           # Smart-Agriculture/

MODEL_PATH  = BASE_DIR / "notebooks" / "models" / "full_multioutput_model.pkl"
CONFIG_PATH = BASE_DIR / "notebooks" / "models" / "model_config.json"

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

model = joblib.load(MODEL_PATH)
print("[OK] Model loaded successfully.")

# --- Optional: load feature order from config ---
FEATURE_ORDER = None
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
        FEATURE_ORDER = cfg.get("features")
    print(f"[OK] Config loaded. Feature order: {FEATURE_ORDER}")
else:
    print(f"[WARN] No model_config.json found at {CONFIG_PATH}. Using default feature order.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_row_keys(row: dict) -> dict:
    """Map common aliases/typos from the UI to canonical model feature names."""
    aliases = {
        "temperature": ["temperature", "tempreature", "temp", "t"],
        "humidity":    ["humidity", "hum"],
        "water_level": ["water_level", "waterlevel", "water"],
        "N":           ["N", "n"],
        "P":           ["P", "p"],
        "K":           ["K", "k"],
        "hour":        ["hour", "h"],
        "day_of_week": ["day_of_week", "dow", "day"],
    }
    out = {}
    for canonical, keys in aliases.items():
        for k in keys:
            if k in row:
                out[canonical] = row[k]
                break
    return out


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def build_df_from_rows(rows: list[dict]) -> pd.DataFrame:
    """
    Convert a list of UI row dicts into a DataFrame ready for model.predict().
    Adds derived features: nutrient_index, humidity_temp_ratio.
    Enforces FEATURE_ORDER if loaded from config.
    """
    processed = []
    for r in rows:
        r = normalize_row_keys(r)

        base = {
            "temperature": safe_float(r.get("temperature")),
            "humidity":    safe_float(r.get("humidity")),
            "water_level": safe_float(r.get("water_level")),
            "N":           safe_float(r.get("N")),
            "P":           safe_float(r.get("P")),
            "K":           safe_float(r.get("K")),
            "hour":        safe_float(r.get("hour")),
            "day_of_week": safe_float(r.get("day_of_week")),
        }

        # Derived features used during training
        base["nutrient_index"]      = (base["N"] + base["P"] + base["K"]) / 3.0
        base["humidity_temp_ratio"] = base["humidity"] / (base["temperature"] + 1e-6)

        processed.append(base)

    df = pd.DataFrame(processed)

    # Enforce training feature order if available
    if FEATURE_ORDER:
        for col in FEATURE_ORDER:
            if col not in df.columns:
                df[col] = 0.0
        df = df[FEATURE_ORDER]

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
    Expects JSON body:
    {
      "rows": [
        {
          "temperature": 29, "humidity": 70, "water_level": 100,
          "N": 185, "P": 190, "K": 160, "hour": 6, "day_of_week": 1
        },
        ...
      ]
    }

    Returns:
    {
      "predictions": [
        {
          "irrigation": {"label": "YES", "probability": 0.92},
          "fan":        {"label": "NO",  "probability": 0.13}
        },
        ...
      ]
    }
    """
    data = request.get_json(silent=True)

    if not data or "rows" not in data:
        return jsonify({"error": "Invalid payload. Expected JSON with key 'rows'."}), 400

    rows = data["rows"]
    if not isinstance(rows, list) or len(rows) == 0:
        return jsonify({"error": "'rows' must be a non-empty list."}), 400

    try:
        input_df = build_df_from_rows(rows)

        preds = model.predict(input_df)  # shape: (n, 2)

        probas = None
        try:
            probas = model.predict_proba(input_df)  # list of arrays, one per target
        except Exception:
            probas = None

        predictions = []
        for i in range(len(input_df)):
            irrigation_pred = int(preds[i][0])
            fan_pred        = int(preds[i][1])

            irrigation_prob = None
            fan_prob        = None

            if probas is not None:
                if probas[0] is not None:
                    irrigation_prob = float(probas[0][i][1])
                if len(probas) > 1 and probas[1] is not None:
                    fan_prob = float(probas[1][i][1])

            predictions.append({
                "irrigation": {
                    "label":       "YES" if irrigation_pred == 1 else "NO",
                    "probability": irrigation_prob,
                },
                "fan": {
                    "label":       "YES" if fan_pred == 1 else "NO",
                    "probability": fan_prob,
                },
            })

        return jsonify({"predictions": predictions})

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