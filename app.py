from pathlib import Path
from datetime import datetime
import random

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# Project 35 - backend-connected prototype


BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "xgboost_quantum_sec.pkl"
SCALER_FILE = BASE_DIR / "scaler_quantum_sec.pkl"
FEATURES_FILE = BASE_DIR / "features_quantum_sec.pkl"
DATA_FILE = BASE_DIR / "UNSW_NB15_training-set.csv"
EVAL_FILE = BASE_DIR / "evaluation_log.csv"
N_ROWS_USED = 50

st.set_page_config(
    page_title="Project 35 XAI Prototype",
    page_icon="🛡️",
    layout="wide",
)

RISK_COLOURS = {
    "Low": "#16a34a",
    "Medium": "#ca8a04",
    "High": "#ea580c",
    "Critical": "#dc2626",
}

SUSPICIOUS_PRESETS = {
    "None - use dataset row as-is": {},
    "Port scan style traffic": {
        "proto": "tcp",
        "service": "-",
        "state": "SYN",
        "dur": 0.001,
        "spkts": 2,
        "dpkts": 0,
        "sbytes": 120,
        "dbytes": 0,
        "rate": 2000,
        "sttl": 254,
        "dttl": 0,
        "sload": 480000,
        "dload": 0,
        "ct_srv_src": 50,
        "ct_dst_ltm": 50,
        "ct_src_dport_ltm": 50,
        "ct_dst_sport_ltm": 50,
        "ct_dst_src_ltm": 50,
        "ct_srv_dst": 50,
    },
    "HTTP data exfiltration style traffic": {
        "proto": "tcp",
        "service": "http",
        "state": "FIN",
        "dur": 15,
        "spkts": 20,
        "dpkts": 120,
        "sbytes": 1200,
        "dbytes": 250000,
        "rate": 10,
        "sload": 500,
        "dload": 120000,
        "dmean": 1000,
        "trans_depth": 5,
        "response_body_len": 200000,
        "ct_srv_src": 20,
        "ct_dst_ltm": 20,
        "ct_dst_src_ltm": 20,
    },
}


def risk_from_probability(probability: float) -> str:
    percentage = probability * 100
    if percentage >= 85:
        return "Critical"
    if percentage >= 70:
        return "High"
    if percentage >= 40:
        return "Medium"
    return "Low"


def recommendation_from_risk(risk: str) -> str:
    if risk == "Critical":
        return "Escalate immediately and check related firewall/SIEM logs."
    if risk == "High":
        return "Review soon and compare with recent activity from the same source."
    if risk == "Medium":
        return "Monitor and collect more evidence before escalating."
    return "Likely benign, but keep the event for audit history."


@st.cache_resource
def load_backend():
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    expected_features = joblib.load(FEATURES_FILE)
    return model, scaler, expected_features


@st.cache_data
def load_dataset():
    data = pd.read_csv(DATA_FILE)
    return data.head(N_ROWS_USED).copy()


def apply_demo_preset(row: pd.Series, preset_name: str) -> pd.Series:
    updated = row.copy()
    for column, value in SUSPICIOUS_PRESETS[preset_name].items():
        if column in updated.index:
            updated[column] = value
    return updated


def prepare_model_input(row: pd.Series, expected_features: list[str]) -> pd.DataFrame:
    event_df = pd.DataFrame([row]).copy()

    # These columns are useful for display/evaluation, but they are not model inputs.
    for column in ["id", "attack_cat", "label"]:
        if column in event_df.columns:
            event_df = event_df.drop(columns=[column])

    categorical_columns = [column for column in ["proto", "state", "service"] if column in event_df.columns]
    event_df = pd.get_dummies(event_df, columns=categorical_columns)

    # Match exactly the 190 columns expected by the trained model.
    event_df = event_df.reindex(columns=expected_features, fill_value=0)
    event_df = event_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    return event_df


def predict_event(row: pd.Series, model, scaler, expected_features):
    model_input = prepare_model_input(row, expected_features)
    scaled_input = scaler.transform(model_input)
    probabilities = model.predict_proba(scaled_input)[0]
    threat_probability = float(probabilities[1])
    predicted_label = "Attack" if threat_probability >= 0.5 else "Benign"
    risk_band = risk_from_probability(threat_probability)

    return {
        "model_input": model_input,
        "scaled_input": scaled_input,
        "threat_probability": threat_probability,
        "predicted_label": predicted_label,
        "risk_band": risk_band,
        "recommendation": recommendation_from_risk(risk_band),
    }


def simple_feature_impact(model, scaled_input, expected_features, top_n=8) -> pd.DataFrame:
    """
    This is intentionally simple. It is not full SHAP yet.
    It uses the trained XGBoost feature importance and the scaled event values
    to give a rough explanation of which inputs mattered most.
    """
    if hasattr(model, "feature_importances_"):
        importances = np.array(model.feature_importances_)
    else:
        importances = np.ones(len(expected_features))

    scaled_values = np.abs(np.array(scaled_input[0]))
    impact_scores = importances * scaled_values

    explanation = pd.DataFrame({
        "feature": expected_features,
        "impact": impact_scores,
    })
    explanation = explanation.sort_values("impact", ascending=False).head(top_n)
    explanation["impact"] = explanation["impact"].round(4)
    return explanation


def make_alert_id() -> str:
    return datetime.now().strftime("A%H%M%S") + str(random.randint(10, 99))


def store_injected_alert(alert: dict):
    if "alerts" not in st.session_state:
        st.session_state.alerts = []
    st.session_state.alerts.insert(0, alert)
    st.session_state.selected_alert_id = alert["alert_id"]


def get_selected_alert():
    alerts = st.session_state.get("alerts", [])
    if not alerts:
        return None
    selected_id = st.session_state.get("selected_alert_id", alerts[0]["alert_id"])
    for alert in alerts:
        if alert["alert_id"] == selected_id:
            return alert
    return alerts[0]


def read_evaluation_log() -> pd.DataFrame:
    if not EVAL_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(EVAL_FILE)


def save_evaluation(entry: dict):
    new_entry = pd.DataFrame([entry])
    if EVAL_FILE.exists():
        existing = pd.read_csv(EVAL_FILE)
        saved = pd.concat([existing, new_entry], ignore_index=True)
    else:
        saved = new_entry
    saved.to_csv(EVAL_FILE, index=False)


st.markdown(
    """
    <style>
    .main-title {font-size: 34px; font-weight: 800; margin-bottom: 0px;}
    .sub-title {color: #6b7280; font-size: 16px; margin-top: 0px;}
    .small-note {color: #6b7280; font-size: 14px;}
    .risk-pill {padding: 5px 12px; border-radius: 999px; font-size: 13px; font-weight: 700; color: white;}
    .soft-box {padding: 16px; border: 1px solid #e5e7eb; border-radius: 14px; background: #ffffff;}
    </style>
    """,
    unsafe_allow_html=True,
)

model, scaler, expected_features = load_backend()
dataset = load_dataset()

if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "selected_alert_id" not in st.session_state:
    st.session_state.selected_alert_id = None

st.markdown('<p class="main-title">Project 35 - Simple Backend XAI Dashboard</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">A simple cybersecurity decision-support prototype.</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Project 35")
    st.write("Explainable AI & Trust-Aware Visualization")
    st.divider()
    st.success("Backend connected")
    st.caption(f"Dataset rows used: first {len(dataset)} rows")
    st.divider()
    if st.button("Clear session alerts"):
        st.session_state.alerts = []
        st.session_state.selected_alert_id = None
        st.rerun()
    if st.button("Clear saved evaluations"):
        if EVAL_FILE.exists():
            EVAL_FILE.unlink()
        st.success("Saved evaluations cleared.")

page = st.radio(
    "Choose page",
    ["Dashboard", "Inject Log", "Explain Alert", "Analyst Evaluation", "About"],
    horizontal=True,
)

# -----------------------------
# Dashboard
# -----------------------------
if page == "Dashboard":
    st.subheader("SOC Overview")
    st.write("This page shows the alerts that have been injected and analysed by the model during this session.")

    alerts = st.session_state.alerts
    total_alerts = len(alerts)
    high_alerts = len([a for a in alerts if a["risk_band"] in ["High", "Critical"]])
    avg_conf = round(np.mean([a["threat_probability"] * 100 for a in alerts]), 1) if alerts else 0
    eval_count = len(read_evaluation_log())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Injected alerts", total_alerts)
    col2.metric("High/Critical", high_alerts)
    col3.metric("Average threat confidence", f"{avg_conf}%")
    col4.metric("Saved evaluations", eval_count)

    st.divider()

    if not alerts:
        st.info("No alerts have been injected yet. Go to the Inject Log page and click the inject button.")
    else:
        table = pd.DataFrame([
            {
                "Alert ID": a["alert_id"],
                "Dataset row": a["dataset_row"],
                "Preset": a["preset"],
                "Protocol": a["proto"],
                "Service": a["service"],
                "State": a["state"],
                "Prediction": a["predicted_label"],
                "Risk": a["risk_band"],
                "Threat confidence (%)": round(a["threat_probability"] * 100, 2),
                "Dataset label": a["dataset_label"],
            }
            for a in alerts
        ])
        st.dataframe(table, use_container_width=True, hide_index=True)

        risk_counts = table["Risk"].value_counts().reindex(["Low", "Medium", "High", "Critical"], fill_value=0)
        st.markdown("### Alerts by risk")
        st.bar_chart(risk_counts)

# -----------------------------
# Inject Log
# -----------------------------
elif page == "Inject Log":
    st.subheader("Inject a Random Network Log")
    st.write(
        "This page randomly selects one of the rows from the dataset, prepares it for the trained model, "
        "then shows the prediction. The suspicious preset is optional and is only for demonstration because the uploaded sample rows are all labelled Normal."
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        preset = st.selectbox("Injection mode", list(SUSPICIOUS_PRESETS.keys()))
        st.caption("Use 'None' for the real row exactly as it appears in the dataset.")
    with col_b:
        random_seed = st.number_input("Optional random seed", min_value=0, max_value=9999, value=0)
        st.caption("Leave as 0 for normal random behaviour.")

    if st.button("Inject random network log", type="primary"):
        if random_seed > 0:
            sampled_row = dataset.sample(1, random_state=int(random_seed)).iloc[0]
        else:
            sampled_row = dataset.sample(1).iloc[0]

        analysed_row = apply_demo_preset(sampled_row, preset)
        prediction = predict_event(analysed_row, model, scaler, expected_features)
        explanation = simple_feature_impact(model, prediction["scaled_input"], expected_features)

        alert = {
            "alert_id": make_alert_id(),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_row": int(sampled_row.get("id", 0)),
            "preset": preset,
            "proto": str(analysed_row.get("proto", "unknown")),
            "service": str(analysed_row.get("service", "unknown")),
            "state": str(analysed_row.get("state", "unknown")),
            "source_bytes": float(analysed_row.get("sbytes", 0)),
            "destination_bytes": float(analysed_row.get("dbytes", 0)),
            "rate": float(analysed_row.get("rate", 0)),
            "duration": float(analysed_row.get("dur", 0)),
            "dataset_label": "Attack" if int(sampled_row.get("label", 0)) == 1 else "Normal",
            "dataset_attack_cat": str(sampled_row.get("attack_cat", "Unknown")),
            "predicted_label": prediction["predicted_label"],
            "threat_probability": prediction["threat_probability"],
            "risk_band": prediction["risk_band"],
            "recommendation": prediction["recommendation"],
            "top_features": explanation.to_dict("records"),
            "raw_event": analysed_row.to_dict(),
        }
        store_injected_alert(alert)
        st.success("Network log injected and analysed by the backend model.")
        st.rerun()

    selected_alert = get_selected_alert()
    if selected_alert:
        st.divider()
        st.markdown("### Latest analysed alert")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Prediction", selected_alert["predicted_label"])
        col2.metric("Risk band", selected_alert["risk_band"])
        col3.metric("Threat confidence", f"{selected_alert['threat_probability'] * 100:.2f}%")
        col4.metric("Dataset row", selected_alert["dataset_row"])

        risk_colour = RISK_COLOURS.get(selected_alert["risk_band"], "#6b7280")
        st.markdown(
            f"<span class='risk-pill' style='background:{risk_colour}'>{selected_alert['risk_band']} risk</span>",
            unsafe_allow_html=True,
        )
        st.info(selected_alert["recommendation"])

        st.markdown("### Basic network details")
        details = pd.DataFrame([
            ["Protocol", selected_alert["proto"]],
            ["Service", selected_alert["service"]],
            ["Connection state", selected_alert["state"]],
            ["Source bytes", selected_alert["source_bytes"]],
            ["Destination bytes", selected_alert["destination_bytes"]],
            ["Rate", selected_alert["rate"]],
            ["Duration", selected_alert["duration"]],
            ["Original dataset label", selected_alert["dataset_label"]],
        ], columns=["Field", "Value"])
        st.dataframe(details, use_container_width=True, hide_index=True)

# -----------------------------
# Explain Alert
# -----------------------------
elif page == "Explain Alert":
    st.subheader("Explain the Selected Alert")
    st.write("This page gives a simple explanation from the backend model output. Full SHAP can be added later.")

    alerts = st.session_state.alerts
    if not alerts:
        st.info("Inject a network log first.")
    else:
        options = {f"{a['alert_id']} - Row {a['dataset_row']} - {a['risk_band']}": a["alert_id"] for a in alerts}
        selected_label = st.selectbox("Select alert", list(options.keys()))
        st.session_state.selected_alert_id = options[selected_label]
        alert = get_selected_alert()

        col1, col2, col3 = st.columns(3)
        col1.metric("Prediction", alert["predicted_label"])
        col2.metric("Threat confidence", f"{alert['threat_probability'] * 100:.2f}%")
        col3.metric("Risk", alert["risk_band"])

        st.markdown("### Plain-English explanation")
        if alert["predicted_label"] == "Attack":
            st.write(
                "The model believes this event looks suspicious compared with the training pattern. "
                "The features below had the strongest influence in the model's simple explanation view."
            )
        else:
            st.write(
                "The model currently sees this event as mostly benign. The features below still show which inputs were most important for the model check."
            )

        top_features = pd.DataFrame(alert["top_features"])
        st.markdown("### Simple feature impact estimate")
        st.caption("This is based on XGBoost feature importance and scaled input values. It is a simple placeholder for SHAP, not a full SHAP explanation yet.")
        st.bar_chart(top_features.set_index("feature"))
        st.dataframe(top_features, use_container_width=True, hide_index=True)

        with st.expander("View raw network row used by the model"):
            raw_df = pd.DataFrame([alert["raw_event"]])
            st.dataframe(raw_df, use_container_width=True, hide_index=True)

# -----------------------------
# Analyst Evaluation
# -----------------------------
elif page == "Analyst Evaluation":
    st.subheader("Analyst Evaluation")
    st.write("This page stores simple analyst feedback in a CSV file so the team can use it later in the report.")

    alerts = st.session_state.alerts
    if not alerts:
        st.info("Inject a network log first, then come back here to review it.")
    else:
        options = {f"{a['alert_id']} - Row {a['dataset_row']} - {a['risk_band']}": a["alert_id"] for a in alerts}
        selected_label = st.selectbox("Select alert to review", list(options.keys()))
        st.session_state.selected_alert_id = options[selected_label]
        alert = get_selected_alert()

        probability = alert["threat_probability"]
        uncertainty = (1 - abs(probability - 0.5) * 2) * 100
        trust_score = int(100 - uncertainty)
        if "style traffic" in alert["preset"]:
            trust_score = max(0, trust_score - 10)

        col1, col2, col3 = st.columns(3)
        col1.metric("Threat confidence", f"{probability * 100:.2f}%")
        col2.metric("Uncertainty", f"{uncertainty:.1f}%")
        col3.metric("Simple trust score", f"{trust_score}/100")

        with st.form("evaluation_form"):
            decision = st.radio(
                "Analyst decision",
                ["Escalate", "Monitor", "Dismiss", "Need more evidence"],
                horizontal=True,
            )
            clarity = st.slider("How clear is the explanation?", 1, 5, 3)
            trust_rating = st.slider("How much do you trust this prediction?", 1, 5, 3)
            note = st.text_area("Short analyst note", placeholder="Example: Check whether this IP/service is expected for this host.")
            submitted = st.form_submit_button("Save evaluation to CSV")

            if submitted:
                save_evaluation({
                    "saved_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "alert_id": alert["alert_id"],
                    "dataset_row": alert["dataset_row"],
                    "preset": alert["preset"],
                    "dataset_label": alert["dataset_label"],
                    "model_prediction": alert["predicted_label"],
                    "risk_band": alert["risk_band"],
                    "threat_confidence_percent": round(probability * 100, 2),
                    "trust_score": trust_score,
                    "analyst_decision": decision,
                    "explanation_clarity_1_to_5": clarity,
                    "analyst_trust_1_to_5": trust_rating,
                    "note": note,
                })
                st.success("Evaluation saved to evaluation_log.csv")

        st.divider()
        st.markdown("### Saved evaluation log")
        eval_log = read_evaluation_log()
        if eval_log.empty:
            st.caption("No saved evaluations yet.")
        else:
            st.dataframe(eval_log, use_container_width=True, hide_index=True)
            st.download_button(
                "Download evaluation CSV",
                data=eval_log.to_csv(index=False),
                file_name="evaluation_log.csv",
                mime="text/csv",
            )

# -----------------------------
# About
# -----------------------------
else:
    st.subheader("About this prototype")
    st.write("This version is intentionally simple.")

    st.markdown("### What is connected now")
    st.write("- Trained XGBoost backend model")
    st.write("- Scaler used before prediction")
    st.write("- Expected 190-feature input structure")
    st.write("- First 50 rows from the UNSW-NB15 training sample")
    st.write("- Stored analyst evaluation log as a local CSV file")

    st.markdown("### What is still simple / future work")
    st.write("- The explanation chart is a simple feature impact estimate, not full SHAP yet")
    st.write("- No login/accounts")
    st.write("- No live network traffic")
    st.write("- No external database")
    st.write("- The uploaded dataset sample only contains Normal rows, so suspicious presets are included for demonstration")

    st.markdown("### How it works")
    st.code(
        """1. Randomly select one row from the first 50 UNSW-NB15 rows.
2. Optionally apply a simple suspicious demo preset.
3. Convert categorical fields into one-hot encoded model features.
4. Reorder the input to match the trained feature list.
5. Scale the input using the saved scaler.
6. Predict threat probability using the XGBoost backend.
7. Store analyst feedback in evaluation_log.csv.""",
        language="text",
    )
