# Smart Irrigation Decision Model

This project prepares IoT smart-agriculture data for binary classification of irrigation need (`Watering_plant_pump_ON` = 0/1).

## What `main.py` does

1. Loads `data/IoTProcessed_Data.csv`
2. Runs EDA:
   - `head`, `info`, `describe`
   - missing-value check
   - target class distribution
   - histograms for `tempreature`, `humidity`, `water_level`
   - numeric correlation heatmap
   - constant column detection
3. Performs feature engineering:
   - extracts time features from `date` (`hour`, `day`, `month`, `dayofweek`)
   - adds cyclic hour features (`hour_sin`, `hour_cos`)
   - adds domain features (`temp_humidity_interaction`, `npk_sum`, `npk_mean`)
   - drops actuator columns except target to reduce leakage
   - removes constant features
4. Builds preprocessing with sklearn `Pipeline`:
   - median imputation
   - standard scaling (numeric features)
5. Splits data (80/20, stratified)
6. Saves artifacts:
   - `outputs/preprocess_pipeline.pkl`
   - `outputs/X_train.csv`, `outputs/X_test.csv`, `outputs/y_train.csv`, `outputs/y_test.csv`
   - EDA report and plots under `outputs/`

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python main.py
```
