"""Smart irrigation data exploration and preprocessing pipeline.

This script performs:
1) Full EDA for the IoT dataset.
2) Feature engineering for time-based and derived features.
3) Leakage-aware preprocessing and train/test splitting.
4) Persistence of a reusable sklearn preprocessing pipeline.

Run:
    python main.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import io
import math

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------
# Configuration
# ----------------------------
DATA_PATH = Path("data/IoTProcessed_Data.csv")
TARGET = "Watering_plant_pump_ON"
DATE_COLUMN = "date"

OUTPUT_DIR = Path("outputs")
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = OUTPUT_DIR / "eda_report.txt"
PIPELINE_PATH = OUTPUT_DIR / "preprocess_pipeline.pkl"

# Columns that represent actuators and may leak control logic.
# We keep the target and drop all other actuator columns.
ACTUATOR_KEYWORDS = ("actuator", "pump_ON", "pump_OFF", "Fan_", "Watering_")


@dataclass
class PreparedData:
    """Container for processed train/test splits and metadata."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    pipeline: Pipeline
    constant_columns: List[str]
    dropped_actuator_columns: List[str]


def ensure_output_dirs() -> None:
    """Create output directories if they do not exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    """Load CSV dataset into a pandas DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)
    return df


def basic_eda(df: pd.DataFrame) -> Dict[str, pd.DataFrame | pd.Series | List[str]]:
    """Run core EDA checks and save summary/plots."""
    results: Dict[str, pd.DataFrame | pd.Series | List[str]] = {}

    # Core tables
    results["head"] = df.head()
    results["describe"] = df.describe(include="all").transpose()
    results["missing_values"] = df.isna().sum().sort_values(ascending=False)

    # Class distribution
    if TARGET not in df.columns:
        raise KeyError(f"Target column '{TARGET}' not found in dataset")
    results["class_distribution"] = df[TARGET].value_counts(dropna=False)

    # Constant / near-useless columns (strictly constant)
    nunique = df.nunique(dropna=False)
    constant_columns = nunique[nunique <= 1].index.tolist()
    results["constant_columns"] = constant_columns

    # Save text report for reproducibility
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write("=== DATASET HEAD ===\n")
        f.write(results["head"].to_string())
        f.write("\n\n=== DATASET INFO ===\n")
        info_buffer = io.StringIO()
        df.info(buf=info_buffer)
        f.write(info_buffer.getvalue())
        f.write("\n\n=== DESCRIBE (ALL COLUMNS) ===\n")
        f.write(results["describe"].to_string())
        f.write("\n\n=== MISSING VALUES ===\n")
        f.write(results["missing_values"].to_string())
        f.write("\n\n=== TARGET CLASS DISTRIBUTION ===\n")
        f.write(results["class_distribution"].to_string())
        f.write("\n\n=== CONSTANT COLUMNS ===\n")
        f.write(str(constant_columns if constant_columns else "None"))

    # Distribution plots for selected columns
    for col in ["tempreature", "humidity", "water_level"]:
        if col in df.columns:
            plt.figure(figsize=(8, 4))
            sns.histplot(df[col], kde=True)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.tight_layout()
            plt.savefig(FIGURE_DIR / f"distribution_{col}.png", dpi=150)
            plt.close()

    # Correlation heatmap (numeric only)
    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        corr = numeric_df.corr(numeric_only=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", center=0)
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "correlation_heatmap.png", dpi=150)
        plt.close()
        results["correlation"] = corr

    return results


def _make_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse datetime and create cyclic and calendar features."""
    out = df.copy()

    if DATE_COLUMN in out.columns:
        out[DATE_COLUMN] = pd.to_datetime(out[DATE_COLUMN], errors="coerce")
        out["hour"] = out[DATE_COLUMN].dt.hour
        out["day"] = out[DATE_COLUMN].dt.day
        out["month"] = out[DATE_COLUMN].dt.month
        out["dayofweek"] = out[DATE_COLUMN].dt.dayofweek

        # Cyclic hour encoding for smoother temporal pattern learning.
        out["hour_sin"] = out["hour"].apply(
            lambda x: pd.NA if pd.isna(x) else math.sin(2 * math.pi * x / 24)
        )
        out["hour_cos"] = out["hour"].apply(
            lambda x: pd.NA if pd.isna(x) else math.cos(2 * math.pi * x / 24)
        )

        # Drop raw date to avoid type issues and potential leakage from exact timestamps.
        out = out.drop(columns=[DATE_COLUMN])

    return out


def _make_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lightweight domain features from sensors."""
    out = df.copy()

    if {"tempreature", "humidity"}.issubset(out.columns):
        # Simple interaction term: hotter + drier conditions typically imply more irrigation need.
        out["temp_humidity_interaction"] = out["tempreature"] * out["humidity"]

    if {"N", "P", "K"}.issubset(out.columns):
        out["npk_sum"] = out[["N", "P", "K"]].sum(axis=1)
        out["npk_mean"] = out[["N", "P", "K"]].mean(axis=1)

    return out


def feature_engineering(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Apply feature engineering and leakage-aware column pruning.

    Returns:
        transformed DataFrame, removed actuator columns, removed constant columns
    """
    out = df.copy()

    out = _make_time_features(out)
    out = _make_domain_features(out)

    # Remove actuator columns except target to reduce leakage from control signals.
    actuator_cols_to_drop: List[str] = []
    for col in out.columns:
        if col == TARGET:
            continue
        if any(keyword in col for keyword in ACTUATOR_KEYWORDS):
            actuator_cols_to_drop.append(col)

    out = out.drop(columns=actuator_cols_to_drop, errors="ignore")

    # Remove constant columns after feature generation.
    constant_cols = out.nunique(dropna=False)
    constant_cols = constant_cols[constant_cols <= 1].index.tolist()
    # Never drop target here, even if pathological constant class.
    constant_cols = [c for c in constant_cols if c != TARGET]
    out = out.drop(columns=constant_cols, errors="ignore")

    return out, actuator_cols_to_drop, constant_cols


def build_preprocess_pipeline(X: pd.DataFrame) -> Pipeline:
    """Construct a reusable sklearn preprocessing pipeline."""
    numeric_features = X.select_dtypes(include="number").columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(steps=[("preprocess", preprocess)])
    return pipeline


def prepare_data(df: pd.DataFrame) -> PreparedData:
    """Run full preparation flow from features to train/test split."""
    transformed_df, dropped_actuators, dropped_constants = feature_engineering(df)

    if TARGET not in transformed_df.columns:
        raise KeyError(f"Target column '{TARGET}' missing after feature engineering")

    X = transformed_df.drop(columns=[TARGET])
    y = transformed_df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_preprocess_pipeline(X_train)
    pipeline.fit(X_train)

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        pipeline=pipeline,
        constant_columns=dropped_constants,
        dropped_actuator_columns=dropped_actuators,
    )


def save_outputs(prepared: PreparedData) -> None:
    """Persist fitted pipeline and split datasets to disk."""
    joblib.dump(prepared.pipeline, PIPELINE_PATH)

    prepared.X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
    prepared.X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
    prepared.y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    prepared.y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)


def print_summary(prepared: PreparedData, eda_results: Dict[str, pd.DataFrame | pd.Series | List[str]]) -> None:
    """Print concise summary of all transformations and outputs."""
    print("\n=== EDA SUMMARY ===")
    print("Missing values by column:")
    print(eda_results["missing_values"])
    print("\nTarget class distribution:")
    print(eda_results["class_distribution"])
    print("\nDetected constant columns (raw data):")
    print(eda_results["constant_columns"])

    print("\n=== FEATURE ENGINEERING SUMMARY ===")
    print(f"Dropped actuator columns (except target): {prepared.dropped_actuator_columns}")
    print(f"Dropped constant columns (post-FE): {prepared.constant_columns}")

    print("\n=== SPLIT SUMMARY ===")
    print(f"X_train shape: {prepared.X_train.shape}")
    print(f"X_test shape: {prepared.X_test.shape}")
    print(f"y_train shape: {prepared.y_train.shape}")
    print(f"y_test shape: {prepared.y_test.shape}")

    print("\nSaved artifacts:")
    print(f"- EDA report: {REPORT_PATH}")
    print(f"- Figures directory: {FIGURE_DIR}")
    print(f"- Preprocessing pipeline: {PIPELINE_PATH}")
    print("- Train/test splits: outputs/X_train.csv, outputs/X_test.csv, outputs/y_train.csv, outputs/y_test.csv")


def main() -> None:
    ensure_output_dirs()
    df = load_data(DATA_PATH)
    eda_results = basic_eda(df)
    prepared = prepare_data(df)
    save_outputs(prepared)
    print_summary(prepared, eda_results)


if __name__ == "__main__":
    main()
