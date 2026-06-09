import streamlit as st
import pandas as pd
import joblib
import random
import time
from pathlib import Path

st.set_page_config(
    page_title="SECS Simulator",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_models():
    models = {}
    try:
        models['decision'] = joblib.load(str(Path(__file__).resolve().parents[1] / "Trained_Model" / "decision_model.pkl"))
        models['encoders'] = joblib.load(str(Path(__file__).resolve().parents[1] / "Trained_Model" / "label_encoders.pkl"))
    except Exception as e:
        st.error(f"Unable to load decision model: {e}")
    return models

@st.cache_data
def load_dataset():
    dataset_path = Path(__file__).resolve().parents[1] / "Dataset" / "traffic_decisions.csv"
    if dataset_path.exists():
        return pd.read_csv(dataset_path)
    return pd.DataFrame()

models = load_models()
dataset = load_dataset()

st.markdown("# 🚦 Emergency Traffic Simulator")
st.markdown("Use sample traffic scenarios to test corridor planning decisions and compare AI outputs.")

mode = st.sidebar.radio("Simulation Mode", ["Random Scenario", "Dataset Playback", "Manual Scenario"])

if mode == "Dataset Playback":
    if dataset.empty:
        st.error("Dataset not found. Make sure Dataset/traffic_decisions.csv exists.")
        st.stop()
    index = st.sidebar.number_input("Choose dataset row", min_value=0, max_value=len(dataset)-1, value=0, step=1)
    scenario = dataset.iloc[index].to_dict()
else:
    traffic_options = ["clear", "low", "medium", "high", "jam"]
    vehicle_options = ["ambulance", "firetruck", "car", "bus", "truck", "motorcycle"]
    time_options = ["morning", "afternoon", "evening", "night"]
    weather_options = ["clear", "rain", "fog"]

    if mode == "Random Scenario":
        if st.sidebar.button("Generate random case") or True:
            scenario = {
                "vehicle_type": random.choice(vehicle_options),
                "traffic_density": random.choice(traffic_options),
                "time_of_day": random.choice(time_options),
                "weather": random.choice(weather_options),
                "distance_m": random.randint(50, 1000),
                "confidence_pct": round(random.uniform(60, 99), 1),
                "vehicle_count": random.randint(0, 30),
            }
    else:
        scenario = {
            "vehicle_type": st.sidebar.selectbox("Vehicle type", vehicle_options, index=2),
            "traffic_density": st.sidebar.selectbox("Traffic density", traffic_options, index=2),
            "time_of_day": st.sidebar.selectbox("Time of day", time_options, index=1),
            "weather": st.sidebar.selectbox("Weather", weather_options, index=0),
            "distance_m": st.sidebar.slider("Distance from intersection (m)", 50, 1000, 420, 10),
            "confidence_pct": st.sidebar.slider("Detection confidence (%)", 60.0, 99.0, 92.0, 0.1),
            "vehicle_count": st.sidebar.number_input("Vehicle count at intersection", 0, 30, 8),
        }

st.markdown("---")

if not scenario:
    st.warning("No scenario selected yet.")
    st.stop()

st.subheader("Scenario details")
cols = st.columns(3)
with cols[0]:
    st.metric("Vehicle", scenario["vehicle_type"].upper())
    st.metric("Distance", f"{scenario['distance_m']} m")
with cols[1]:
    st.metric("Traffic", scenario["traffic_density"].upper())
    st.metric("Confidence", f"{scenario['confidence_pct']} %")
with cols[2]:
    st.metric("Weather", scenario["weather"].capitalize())
    st.metric("Vehicles waiting", scenario["vehicle_count"])

if st.button("Run AI prediction"):
    if 'decision' not in models or models['decision'] is None:
        st.error("Decision model is unavailable. Check Trained_Model/decision_model.pkl.")
    else:
        enc = models['encoders']
        X = pd.DataFrame([{
            'vehicle_type_enc': enc['vehicle_type'].transform([scenario['vehicle_type']])[0],
            'traffic_density_enc': enc['traffic_density'].transform([scenario['traffic_density']])[0],
            'time_of_day_enc': enc['time_of_day'].transform([scenario['time_of_day']])[0],
            'weather_enc': enc['weather'].transform([scenario['weather']])[0],
            'distance_m': scenario['distance_m'],
            'confidence_pct': scenario['confidence_pct'],
            'vehicle_count': scenario['vehicle_count'],
        }])
        try:
            prediction = models['decision'].predict(X)[0]
            probabilities = models['decision'].predict_proba(X)[0]
            decision = enc['decision'].inverse_transform([prediction])[0]
            prob_map = {c: round(float(p) * 100, 1) for c, p in zip(enc['decision'].classes_, probabilities)}
            st.success(f"AI Decision: {decision}")
            st.write("### Decision probabilities")
            for cls, prob in prob_map.items():
                st.write(f"- **{cls}**: {prob}%")
            if decision == "OPEN_CORRIDOR":
                st.info("Emergency corridor should open to prioritize the vehicle.")
            elif decision == "CAUTION":
                st.warning("Monitor the situation and prepare to adjust traffic lights.")
            else:
                st.success("No corridor action needed; continue normal flow.")

            st.markdown("---")
            st.write("## Simulation log")
            timeline = [
                "Detect approaching vehicle via CCTV and sensors.",
                "Analyze vehicle type, distance, confidence, and traffic patterns.",
                "Evaluate emergency priority rules and ethical constraints.",
                "Update signal timing and open corridor if required.",
                "Monitor vehicle clearance and restore normal traffic.",
            ]
            for idx, entry in enumerate(timeline, start=1):
                st.write(f"**Step {idx}.** {entry}")
            if decision == "OPEN_CORRIDOR":
                st.success("Result: Emergency corridor activated.")
            elif decision == "CAUTION":
                st.warning("Result: Caution state — hold decision until higher confidence or closer range.")
            else:
                st.info("Result: Normal traffic cycle preserved.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.markdown("---")
with st.expander("About this simulator"):
    st.write(
        "This simulator uses the trained decision model to evaluate traffic scenarios and visualize how the emergency corridor planner would behave. "
        "Use dataset playback to test real examples or manual mode to craft your own scenarios."
    )
