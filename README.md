# Project 35 - Simple Backend XAI Dashboard

This is a simple Streamlit prototype for **Project 35 - Designing an Interactive Explainable AI & Trust-Aware Visualization System for Cybersecurity Decision Support**.

## What it does

- Uses the saved XGBoost model backend.
- Loads the saved scaler.
- Loads the expected model feature list.
- Uses only the first 50 rows of the UNSW-NB15 training sample to keep processing light.
- Lets the user inject a random network log.
- Analyses the log using the backend model.
- Shows threat confidence, risk level and a simple explanation chart.
- Stores analyst evaluation feedback in `evaluation_log.csv`.

## Files included

- `app.py` - Streamlit dashboard
- `requirements.txt` - Python packages
- `xgboost_quantum_sec.pkl` - trained model
- `scaler_quantum_sec.pkl` - scaler
- `features_quantum_sec.pkl` - expected features
- `UNSW_NB15_training-set.csv` - dataset sample

## How to run in VSCode

Open this folder in VSCode, then run:

```bash
pip install -r requirements.txt
```

Then run:

```bash
streamlit run app.py
```

Or:

```bash
python -m streamlit run app.py
```

## Important note

The uploaded dataset sample only contains Normal rows in the first 50 rows. For demonstration, the dashboard includes optional suspicious presets such as port scan and HTTP data exfiltration style traffic. These presets are simple demo modifications and are not a replacement for a full labelled attack dataset.
