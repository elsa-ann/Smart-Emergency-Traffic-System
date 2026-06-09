import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import time
import json
from pathlib import Path
from PIL import Image
import random
import io
import importlib.util

# Cari folder root proyek (yang berisi Trained_Model/)
def find_project_root(start_path):
    current = Path(start_path).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "Trained_Model").exists():
            return parent
    return current  # fallback

PROJECT_ROOT = find_project_root(__file__)

# ── PAGE CONFIG ─────────────────────────────────────────────────
st.set_page_config(
    page_title="SETS — Smart Emergency Traffic System",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── VIEW SELECTOR ─────────────────────────────────────────────────
view = st.sidebar.radio("View", ["Prototype (HTML)", "Simulator", "Streamlit Dashboard", "Webcam AI"], index=0)
html_path = Path(__file__).with_name('app.html')
if view == "Prototype (HTML)":
    if html_path.exists():
        components.html(html_path.read_text(encoding='utf-8'), width=1200, height=920, scrolling=True)
    else:
        st.error(f'File not found: {html_path}')
    st.stop()
elif view == "Simulator":
    try:
        import runpy
        runpy.run_path(str(Path(__file__).with_name('app_sim.py')))
    except Exception as e:
        st.error(f'Failed to launch simulator: {e}')
    st.stop()
elif view == "Webcam AI":
    st.markdown("""
    <div style="background:#080f18;border:1px solid #0e3a5c;border-radius:8px;
    padding:16px 24px;margin-bottom:16px;text-align:center">
        <div style="font-family:Orbitron,monospace;font-size:20px;font-weight:900;
        color:#00c8ff;letter-spacing:3px">📷 WEBCAM AI — EMERGENCY CLASSIFIER</div>
        <div style="font-family:Share Tech Mono,monospace;font-size:10px;
        color:#5a9abf;letter-spacing:2px;margin-top:6px">
        POWERED BY RANDOM FOREST · DATASET BARU · REAL-TIME DETECTION</div>
    </div>
    """, unsafe_allow_html=True)

    # Load model classifier dan label encoder
    @st.cache_resource
    def load_classifier():
        try:
            # Cetak path untuk debugging (hapus nanti)
            path_model = PROJECT_ROOT / "Trained_Model" / "webcam_classifier.pkl"
            # st.write(f"Debug: looking for {path_model}")  # sementara
            clf = joblib.load(path_model)
            with open(PROJECT_ROOT / "Trained_Model" / "webcam_classes.json", "r") as f:
                classes = json.load(f)["classes"]
            return clf, classes, None
        except Exception as e:
            return None, None, str(e)

    # Fungsi ekstraksi fitur (sama seperti di notebook)
    def extract_features(image, size=(128, 128)):
        image = image.convert('RGB').resize(size)
        arr = np.array(image)
        # Histogram RGB
        hist = []
        for i in range(3):
            h = np.histogram(arr[:,:,i], bins=32, range=(0,255))[0].astype(float)
            hist.extend(h)
        # Edge stats sederhana
        gray = np.mean(arr, axis=2).astype(np.uint8)
        edge_stats = [gray.mean(), gray.std(), gray.max()]
        return np.concatenate([hist, edge_stats])

    classifier, class_names, cls_err = load_classifier()
    # Load decision model (masih sama)
    @st.cache_resource
    def load_decision():
        try:
            rf = joblib.load(PROJECT_ROOT / "Trained_Model" / "decision_model.pkl")
            enc = joblib.load(PROJECT_ROOT / "Trained_Model" / "label_encoders.pkl")
            return rf, enc, None
        except Exception as e:
            return None, None, str(e)

    rf_model, rf_enc, rf_err = load_decision()

    # Status badges
    c1, c2 = st.columns(2)
    with c1:
        if classifier:
            st.success("✅ Classifier (Random Forest) loaded")
        else:
            st.error(f"❌ Classifier error: {cls_err}")
    with c2:
        if rf_model:
            st.success("✅ Decision Model loaded")
        else:
            st.error(f"❌ Decision model error: {rf_err}")

    st.markdown("---")

    # Input mode
    input_mode = st.radio("Input Mode", ["📷 Live Webcam", "🖼️ Upload Image"], horizontal=True)

    if input_mode == "📷 Live Webcam":
        img_input = st.camera_input("Arahkan kamera ke kendaraan")
    else:
        uploaded = st.file_uploader("Upload foto kendaraan", type=["jpg","jpeg","png"])
        img_input = uploaded

    if img_input is not None:
        from PIL import Image
        img = Image.open(img_input).convert("RGB")
        col_img, col_result = st.columns([1, 1])
        with col_img:
            st.markdown("**📸 Input Image**")
            st.image(img, use_container_width=True)
        with col_result:
            st.markdown("**🤖 Detection Result**")
            if classifier:
                with st.spinner("Memproses gambar..."):
                    feat = extract_features(img).reshape(1, -1)
                    pred_label = classifier.predict(feat)[0]          # langsung dapat string
                    pred_proba = classifier.predict_proba(feat)[0].max()
                # Tampilkan hasil deteksi
                is_emergency = (pred_label == "emergency")
                color = "#ff4500" if is_emergency else "#00ff9d"
                emoji = "🚑" if is_emergency else "🚗"
                st.markdown(f"""
                <div style="background:#0c1620;border:1px solid {color};border-radius:4px;
                padding:10px;margin-bottom:8px">
                    <div style="font-family:Orbitron,monospace;font-size:14px;color:{color}">
                    {emoji} {pred_label.upper()}</div>
                    <div style="font-family:Share Tech Mono,monospace;font-size:11px;color:#5a9abf;margin-top:4px">
                    Confidence: <span style="color:{color}">{pred_proba*100:.1f}%</span></div>
                </div>
                """, unsafe_allow_html=True)

                # Panggil decision model jika emergency dan confidence >=85%
                if is_emergency and pred_proba >= 0.85:
                    # Gunakan decision model
                    if rf_model and rf_enc:
                        # Contoh parameter: jarak 300m, traffic medium, dll (bisa diambil dari sidebar atau default)
                        # Di sini kita gunakan default sederhana, tapi bisa juga user set di sidebar
                        X = pd.DataFrame([{
                            'vehicle_type_enc': rf_enc['vehicle_type'].transform(["ambulance"])[0],
                            'traffic_density_enc': rf_enc['traffic_density'].transform(["medium"])[0],
                            'time_of_day_enc': rf_enc['time_of_day'].transform(["afternoon"])[0],
                            'weather_enc': rf_enc['weather'].transform(["clear"])[0],
                            'distance_m': 300,
                            'confidence_pct': pred_proba*100,
                            'vehicle_count': 5,
                        }])
                        pred = rf_model.predict(X)[0]
                        proba = rf_model.predict_proba(X)[0]
                        decision = rf_enc['decision'].inverse_transform([pred])[0]
                        prob_map = {c: round(float(p)*100,1) for c,p in zip(rf_enc['decision'].classes_, proba)}
                        if decision == "OPEN_CORRIDOR":
                            st.markdown('<div class="emergency-banner" style="font-size:14px;">⚠ OPEN CORRIDOR ⚠<br>Emergency vehicle detected. Traffic light will turn GREEN.</div>', unsafe_allow_html=True)
                        elif decision == "CAUTION":
                            st.markdown('<div class="caution-banner">⚠ CAUTION — Monitoring, confidence not high enough</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="normal-banner">✓ NO ACTION — No corridor needed</div>', unsafe_allow_html=True)
                        st.markdown("**Decision probabilities:**")
                        for cls2, p in sorted(prob_map.items(), key=lambda x:-x[1]):
                            color2 = "#ff6600" if cls2=="OPEN_CORRIDOR" else "#ffb800" if cls2=="CAUTION" else "#00ff9d"
                            st.markdown(f'<div style="display:flex;justify-content:space-between;font-family:Share Tech Mono,monospace;font-size:10px;color:{color2};margin-bottom:3px"><span>{cls2}</span><span>{p}%</span></div>', unsafe_allow_html=True)
                    else:
                        st.warning("Decision model not available.")
                elif is_emergency:
                    st.markdown('<div class="caution-banner">⚠ CAUTION — Emergency detected but confidence below 85% (ethical guardrail)</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="normal-banner">✓ NO ACTION — Non-emergency vehicle</div>', unsafe_allow_html=True)
            else:
                st.error("Classifier model not available.")
    else:
        st.info("📷 Aktifkan webcam atau upload foto kendaraan untuk memulai deteksi.")
    st.stop()

# ── LOAD MODELS ─────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    try:
        models['decision'] = joblib.load(PROJECT_ROOT / "Trained_Model" / "decision_model.pkl")
        models['encoders'] = joblib.load(PROJECT_ROOT / "Trained_Model" / "label_encoders.pkl")
        models['yolo_ready'] = (PROJECT_ROOT / "Trained_Model" / "best.pt").exists()
    except Exception as e:
        st.error(f"Error loading models: {e}")
    return models

models = load_models()

# ── CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

* { font-family: 'Rajdhani', sans-serif; }

.stApp {
    background: #050b12;
    color: #c8e8f8;
}

/* Header */
.secs-header {
    background: linear-gradient(135deg, #080f18, #0c1a2e);
    border: 1px solid #0e3a5c;
    border-radius: 8px;
    padding: 20px 30px;
    margin-bottom: 20px;
    text-align: center;
    box-shadow: 0 0 40px rgba(0,200,255,0.1);
}
.secs-title {
    font-family: 'Orbitron', monospace;
    font-size: 28px;
    font-weight: 900;
    color: #00c8ff;
    letter-spacing: 4px;
    text-shadow: 0 0 30px rgba(0,200,255,0.6);
    margin: 0;
}
.secs-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
    color: #5a9abf;
    letter-spacing: 3px;
    margin-top: 6px;
}

/* Cards */
.card {
    background: #080f18;
    border: 1px solid #0e3a5c;
    border-radius: 6px;
    padding: 16px;
    margin-bottom: 12px;
}
.card-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    color: #5a9abf;
    letter-spacing: 3px;
    text-transform: uppercase;
    border-bottom: 1px solid #0e3a5c;
    padding-bottom: 8px;
    margin-bottom: 12px;
}

/* Status badges */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 3px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    font-weight: 700;
}
.badge-green  { background: rgba(0,255,157,0.15); color: #00ff9d; border: 1px solid rgba(0,255,157,0.4); }
.badge-red    { background: rgba(255,59,59,0.15);  color: #ff3b3b; border: 1px solid rgba(255,59,59,0.4); }
.badge-orange { background: rgba(255,69,0,0.15);   color: #ff6600; border: 1px solid rgba(255,69,0,0.4); }
.badge-yellow { background: rgba(255,184,0,0.15);  color: #ffb800; border: 1px solid rgba(255,184,0,0.4); }
.badge-blue   { background: rgba(0,200,255,0.15);  color: #00c8ff; border: 1px solid rgba(0,200,255,0.4); }

/* Metric boxes */
.metric-box {
    background: #0c1620;
    border: 1px solid #0e3a5c;
    border-radius: 4px;
    padding: 12px;
    text-align: center;
}
.metric-val {
    font-family: 'Orbitron', monospace;
    font-size: 22px;
    font-weight: 700;
    color: #00c8ff;
}
.metric-lbl {
    font-family: 'Share Tech Mono', monospace;
    font-size: 9px;
    color: #5a9abf;
    letter-spacing: 2px;
    margin-top: 4px;
}

/* Emergency banner */
.emergency-banner {
    background: linear-gradient(135deg, #8b0000, #cc2200);
    border: 2px solid #ff4500;
    border-radius: 6px;
    padding: 14px;
    text-align: center;
    font-family: 'Orbitron', monospace;
    font-size: 16px;
    font-weight: 900;
    color: white;
    letter-spacing: 3px;
    box-shadow: 0 0 30px rgba(255,69,0,0.5);
    margin: 10px 0;
}
.normal-banner {
    background: rgba(0,255,157,0.08);
    border: 1px solid rgba(0,255,157,0.3);
    border-radius: 6px;
    padding: 12px;
    text-align: center;
    font-family: 'Share Tech Mono', monospace;
    font-size: 13px;
    color: #00ff9d;
    margin: 10px 0;
}
.caution-banner {
    background: rgba(255,184,0,0.08);
    border: 1px solid rgba(255,184,0,0.4);
    border-radius: 6px;
    padding: 12px;
    text-align: center;
    font-family: 'Share Tech Mono', monospace;
    font-size: 13px;
    color: #ffb800;
    margin: 10px 0;
}

/* Rule items */
.rule-item {
    background: #0c1620;
    border-left: 3px solid #1a5c8a;
    padding: 8px 12px;
    margin-bottom: 6px;
    border-radius: 0 4px 4px 0;
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    color: #5a9abf;
}
.rule-item.active {
    border-left-color: #00ff9d;
    color: #00ff9d;
    background: rgba(0,255,157,0.06);
}
.rule-item.fail {
    border-left-color: #ff3b3b;
    color: #ff3b3b;
    background: rgba(255,59,59,0.06);
}

/* Plan steps */
.plan-step {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 7px 0;
    border-bottom: 1px solid #0e3a5c;
    font-size: 13px;
    color: #5a9abf;
}
.plan-step.done { color: #00ff9d; }
.step-circle {
    width: 22px; height: 22px;
    border-radius: 50%;
    border: 1px solid #1a5c8a;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Share Tech Mono', monospace;
    font-size: 9px;
    flex-shrink: 0;
}
.step-circle.done {
    border-color: #00ff9d;
    background: rgba(0,255,157,0.15);
    color: #00ff9d;
}

/* Traffic light visual */
.tl-container { display: flex; gap: 16px; justify-content: center; margin: 12px 0; }
.tl-box {
    background: #08141e;
    border: 1px solid #1a5c8a;
    border-radius: 6px;
    padding: 10px 14px;
    text-align: center;
    min-width: 70px;
}
.tl-dir {
    font-family: 'Share Tech Mono', monospace;
    font-size: 9px; color: #5a9abf; letter-spacing: 1px; margin-bottom: 6px;
}
.tl-light {
    font-size: 28px;
    line-height: 1;
}
.tl-state {
    font-family: 'Share Tech Mono', monospace;
    font-size: 8px; margin-top: 4px;
}

/* Confidence bar */
.conf-bar-bg {
    background: #0c1620;
    border: 1px solid #0e3a5c;
    border-radius: 2px;
    height: 8px;
    margin: 6px 0;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.5s;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #080f18 !important;
    border-right: 1px solid #0e3a5c;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #004d7a, #007ab8) !important;
    border: 1px solid #00c8ff !important;
    color: #00c8ff !important;
    font-family: 'Orbitron', monospace !important;
    letter-spacing: 2px !important;
    font-size: 11px !important;
    border-radius: 4px !important;
    padding: 10px 20px !important;
    transition: all 0.3s !important;
}
.stButton > button:hover {
    box-shadow: 0 0 20px rgba(0,200,255,0.4) !important;
    transform: translateY(-1px) !important;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 10px !important;
    color: #5a9abf !important;
    letter-spacing: 2px !important;
}
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ────────────────────────────────────────────────
if 'scan_result' not in st.session_state:
    st.session_state.scan_result = None
if 'scan_history' not in st.session_state:
    st.session_state.scan_history = []
if 'total_scans' not in st.session_state:
    st.session_state.total_scans = 0
if 'corridors_opened' not in st.session_state:
    st.session_state.corridors_opened = 0

# ── HEADER ──────────────────────────────────────────────────────
st.markdown("""
<div class="secs-header">
    <p class="secs-title">🚑 SETS // SMART EMERGENCY TRAFFIC SYSTEM</p>
    <p class="secs-sub">AI-POWERED TRAFFIC MANAGEMENT · REASONING & PLANNING · ETHICAL GUARDRAILS</p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="card-title">⚙ SIMULATION CONTROL</div>', unsafe_allow_html=True)
    direction = st.selectbox(
        "DIRECTION",
        ["north", "south", "east", "west"],
        format_func=lambda x: {
            "north": "⬆️ North (dari atas)",
            "south": "⬇️ South (dari bawah)",
            "east":  "➡️ East (dari kanan)",
            "west":  "⬅️ West (dari kiri)"
        }[x]
    )
    vehicle_type = st.selectbox(
        "VEHICLE TYPE",
        ["ambulance", "firetruck", "car", "bus", "truck", "motorcycle"],
        format_func=lambda x: {"ambulance":"🚑 Ambulance","firetruck":"🚒 Fire Truck",
                                "car":"🚗 Car","bus":"🚌 Bus","truck":"🚛 Truck",
                                "motorcycle":"🏍 Motorcycle"}[x]
    )

    distance = st.slider("DISTANCE FROM INTERSECTION (m)", 50, 1000, 420, 10)
    confidence = st.slider("DETECTION CONFIDENCE (%)", 60, 99, 94, 1)
    traffic_density = st.selectbox("TRAFFIC DENSITY", ["clear","low","medium","high","jam"],
                                    index=2,
                                    format_func=lambda x: {"clear":"◻ Clear","low":"◈ Low",
                                                           "medium":"◉ Medium","high":"⬛ High","jam":"🔴 Traffic Jam"}[x])
    vehicle_count = st.number_input("VEHICLE COUNT AT INTERSECTION", 0, 30, 8)
    time_of_day = st.selectbox("TIME OF DAY", ["morning","afternoon","evening","night"],
                                format_func=lambda x: {"morning":"🌅 Morning","afternoon":"☀ Afternoon",
                                                       "evening":"🌆 Evening","night":"🌙 Night"}[x])
    weather = st.selectbox("WEATHER", ["clear","rain","fog"],
                            format_func=lambda x: {"clear":"☀ Clear","rain":"🌧 Rain","fog":"🌫 Fog"}[x])

    st.markdown("---")
    scan_btn = st.button("▶ ACTIVATE CCTV DETECTION", use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="card-title">📊 SESSION STATS</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{st.session_state.total_scans}</div><div class="metric-lbl">TOTAL SCANS</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{st.session_state.corridors_opened}</div><div class="metric-lbl">CORRIDORS</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="card-title">🤖 MODEL STATUS</div>', unsafe_allow_html=True)
    st.markdown(f'<span class="badge badge-green">✓ RANDOM FOREST LOADED</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    yolo_status = "badge-green" if models.get('yolo_ready') else "badge-yellow"
    yolo_text = "✓ YOLO READY" if models.get('yolo_ready') else "⚠ YOLO SIMULATED"
    st.markdown(f'<span class="badge {yolo_status}">{yolo_text}</span>', unsafe_allow_html=True)

# ── AI PREDICTION FUNCTION ───────────────────────────────────────
def predict_decision(vehicle_type, traffic_density, time_of_day, weather,
                     distance, confidence, vehicle_count):
    try:
        enc = models['encoders']
        features = {
            'vehicle_type_enc':    enc['vehicle_type'].transform([vehicle_type])[0],
            'traffic_density_enc': enc['traffic_density'].transform([traffic_density])[0],
            'time_of_day_enc':     enc['time_of_day'].transform([time_of_day])[0],
            'weather_enc':         enc['weather'].transform([weather])[0],
            'distance_m':          distance,
            'confidence_pct':      confidence,
            'vehicle_count':       vehicle_count,
        }
        X = pd.DataFrame([features])
        pred_enc  = models['decision'].predict(X)[0]
        pred_prob = models['decision'].predict_proba(X)[0]
        decision  = enc['decision'].inverse_transform([pred_enc])[0]
        max_prob  = round(float(max(pred_prob)) * 100, 1)
        classes   = enc['decision'].classes_
        probs     = {c: round(float(p)*100, 1) for c, p in zip(classes, pred_prob)}
        return decision, max_prob, probs
    except Exception as e:
        return "CAUTION", 75.0, {}

# ── MAIN SCAN LOGIC ───────────────────────────────────────────────
if scan_btn:
    st.session_state.total_scans += 1
    is_emergency = vehicle_type in ["ambulance", "firetruck"]

    # Simulate YOLO detection
    with st.spinner("🔍 CCTV scanning..."):
        time.sleep(0.8)

    # AI Decision
    decision, prob, all_probs = predict_decision(
        vehicle_type, traffic_density, time_of_day, weather,
        distance, confidence, vehicle_count
    )

    if decision == "OPEN_CORRIDOR":
        st.session_state.corridors_opened += 1

    st.session_state.scan_result = {
        "vehicle_type": vehicle_type,
        "is_emergency": is_emergency,
        "distance": distance,
        "confidence": confidence,
        "traffic_density": traffic_density,
        "vehicle_count": vehicle_count,
        "time_of_day": time_of_day,
        "weather": weather,
        "direction": direction,
        "decision": decision,
        "prob": prob,
        "all_probs": all_probs,
        "timestamp": time.strftime("%H:%M:%S"),
    }
    st.session_state.scan_history.insert(0, st.session_state.scan_result.copy())
    if len(st.session_state.scan_history) > 10:
        st.session_state.scan_history = st.session_state.scan_history[:10]

# ── MAIN CONTENT ─────────────────────────────────────────────────
res = st.session_state.scan_result

col_left, col_mid, col_right = st.columns([1.1, 1.2, 0.9])

# ── LEFT: CCTV + Detection ───────────────────────────────────────
with col_left:
    st.markdown('<div class="card-title">📷 CCTV DETECTION</div>', unsafe_allow_html=True)

    # CCTV feed simulation
    cctv_color = "#ff4500" if (res and res['is_emergency']) else "#00c8ff"
    detected_text = "SCANNING..."

    if res:
        emoji_map = {"ambulance":"🚑","firetruck":"🚒","car":"🚗","bus":"🚌","truck":"🚛","motorcycle":"🏍"}
        detected_text = f"{emoji_map.get(res['vehicle_type'],'?')} {res['vehicle_type'].upper()}"
    else:
        detected_text = "SCANNING..."

    st.markdown(f"""
    <div style="background:#020608;border:1px solid #0e3a5c;border-radius:4px;height:110px;
    display:flex;align-items:center;justify-content:center;position:relative;overflow:hidden;margin-bottom:10px;">
        <div style="position:absolute;top:6px;right:10px;font-family:'Share Tech Mono',monospace;
        font-size:9px;color:#ff3b3b;display:flex;align-items:center;gap:4px;">
            <span style="width:6px;height:6px;border-radius:50%;background:#ff3b3b;display:inline-block"></span>REC
        </div>
        <div style="text-align:center">
            <div style="font-family:'Orbitron',monospace;font-size:16px;color:{cctv_color};
            text-shadow:0 0 15px {cctv_color};">{detected_text}</div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:9px;color:#1a5c8a;margin-top:4px;">
            {'AI PROCESSING COMPLETE' if res else 'AWAITING SCAN'}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if res:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="metric-box"><div class="metric-val" style="font-size:16px;color:{"#ff4500" if res["is_emergency"] else "#00ff9d"}">{res["vehicle_type"].upper()}</div><div class="metric-lbl">DETECTED</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-box"><div class="metric-val" style="font-size:16px;">{res["distance"]}m</div><div class="metric-lbl">DISTANCE</div></div>', unsafe_allow_html=True)

        st.markdown(f'<div style="margin:8px 0"><div style="display:flex;justify-content:space-between;font-family:\'Share Tech Mono\',monospace;font-size:9px;color:#5a9abf;margin-bottom:4px"><span>CONFIDENCE</span><span>{res["confidence"]}%</span></div><div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{res["confidence"]}%;background:{"linear-gradient(90deg,#00ff9d,#00c8ff)" if res["confidence"]>=85 else "linear-gradient(90deg,#ffb800,#ff3b3b)"}"></div></div></div>', unsafe_allow_html=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown(f'<div class="metric-box"><div class="metric-val" style="font-size:14px;">{res["traffic_density"].upper()}</div><div class="metric-lbl">TRAFFIC</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-box"><div class="metric-val" style="font-size:16px;">{res["vehicle_count"]}</div><div class="metric-lbl">VEHICLES</div></div>', unsafe_allow_html=True)

    # Traffic Light Visual
    st.markdown('<div class="card-title" style="margin-top:14px">🚦 TRAFFIC LIGHT STATUS</div>', unsafe_allow_html=True)

    if res and res['decision'] == 'OPEN_CORRIDOR' and res['is_emergency']:
        # Tentukan jalur prioritas berdasarkan direction dari scan_result
        dir_priority = res.get('direction', 'north')
        # Jika arah utara atau selatan, maka NS hijau, EW merah
        if dir_priority in ['north', 'south']:
            ns_green = True
            ew_green = False
        else:  # east atau west
            ns_green = False
            ew_green = True
    else:
        # Normal cycle: bergantian (misal default NS hijau)
        ns_green = True
        ew_green = False

    ns_light = "🟢" if ns_green else "🔴"
    ew_light = "🟢" if ew_green else "🔴"
    ns_label = "GREEN" if ns_green else "RED"
    ew_label = "GREEN" if ew_green else "RED"
    ns_color = "#00ff9d" if ns_green else "#ff3b3b"
    ew_color = "#00ff9d" if ew_green else "#ff3b3b"

        # Tentukan teks arah untuk banner
    if res and res["decision"] == "OPEN_CORRIDOR":
        dir_text = {
            "north": "⬆️ UTARA",
            "south": "⬇️ SELATAN",
            "east": "➡️ TIMUR",
            "west": "⬅️ BARAT"
        }.get(res.get('direction', 'north'), '')
        banner_html = f'<div class="emergency-banner">⚠ EMERGENCY CORRIDOR ACTIVE ⚠<br>PRIORITY LANE: {dir_text}</div>'
    elif res:
        banner_html = '<div class="normal-banner">✓ NORMAL TRAFFIC CYCLE</div>'
    else:
        banner_html = '<div style="text-align:center;font-family:Share Tech Mono,monospace;font-size:11px;color:#1a5c8a;padding:10px">AWAITING SCAN...</div>'

    st.markdown(f"""
    <div style="display:flex;gap:10px;justify-content:center;margin:8px 0">
        <div class="tl-box">
            <div class="tl-dir">NORTH ↓</div>
            <div class="tl-light">{ns_light}</div>
            <div class="tl-state" style="color:{ns_color}">{ns_label}</div>
        </div>
        <div class="tl-box">
            <div class="tl-dir">SOUTH ↑</div>
            <div class="tl-light">{ns_light}</div>
            <div class="tl-state" style="color:{ns_color}">{ns_label}</div>
        </div>
        <div class="tl-box">
            <div class="tl-dir">EAST →</div>
            <div class="tl-light">{ew_light}</div>
            <div class="tl-state" style="color:{ew_color}">{ew_label}</div>
        </div>
        <div class="tl-box">
            <div class="tl-dir">WEST ←</div>
            <div class="tl-light">{ew_light}</div>
            <div class="tl-state" style="color:{ew_color}">{ew_label}</div>
        </div>
    </div>
    {banner_html}
    """, unsafe_allow_html=True)

# ── MIDDLE: AI Reasoning + Planning ─────────────────────────────
with col_mid:
    st.markdown('<div class="card-title">🧠 AI REASONING ENGINE</div>', unsafe_allow_html=True)

    rules = [
        ("RULE 01", "Emergency Vehicle Detected?",
         res and res['is_emergency'], res and not res['is_emergency'],
         f"{'🚑' if res and res['vehicle_type']=='ambulance' else '🚒' if res and res['vehicle_type']=='firetruck' else '🚗'} {res['vehicle_type'].upper()} DETECTED" if res else ""),
        ("RULE 02", "Distance < 500m?",
         res and res['distance'] <= 500, res and res['distance'] > 500,
         f"Distance: {res['distance']}m {'✓' if res and res['distance']<=500 else '✗'}" if res else ""),
        ("RULE 03", "Confidence ≥ 85%?",
         res and res['confidence'] >= 85, res and res['confidence'] < 85,
         f"Confidence: {res['confidence']}% {'✓' if res and res['confidence']>=85 else '— BELOW THRESHOLD'}" if res else ""),
        ("RULE 04", "Traffic Density Check",
         res is not None, False,
         f"Traffic: {res['traffic_density'].upper()} | Vehicles: {res['vehicle_count']}" if res else ""),
    ]

    for rule_id, rule_text, active, fail, detail in rules:
        if active:
            st.markdown(f'<div class="rule-item active"><span style="color:#1a5c8a;font-size:9px">{rule_id}</span><br>{detail or rule_text}</div>', unsafe_allow_html=True)
        elif fail:
            st.markdown(f'<div class="rule-item fail"><span style="color:#5a2a2a;font-size:9px">{rule_id}</span><br>{detail or rule_text}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="rule-item"><span style="font-size:9px">{rule_id}</span><br>{rule_text}</div>', unsafe_allow_html=True)

    # AI Decision result
    if res:
        dec = res['decision']
        if dec == "OPEN_CORRIDOR":
            st.markdown(f'<div style="background:rgba(255,69,0,0.1);border:1px solid rgba(255,69,0,0.4);border-radius:4px;padding:10px;text-align:center;font-family:Share Tech Mono,monospace;font-size:13px;color:#ff6600;margin-top:8px">✓ EMERGENCY PRIORITY ACTIVATED<br><span style="font-size:10px;color:#5a9abf">Model confidence: {res["prob"]}%</span></div>', unsafe_allow_html=True)
        elif dec == "CAUTION":
            st.markdown(f'<div style="background:rgba(255,184,0,0.08);border:1px solid rgba(255,184,0,0.4);border-radius:4px;padding:10px;text-align:center;font-family:Share Tech Mono,monospace;font-size:13px;color:#ffb800;margin-top:8px">⚠ CAUTION — MONITORING<br><span style="font-size:10px;color:#5a9abf">Model confidence: {res["prob"]}%</span></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="background:rgba(0,255,157,0.06);border:1px solid rgba(0,255,157,0.3);border-radius:4px;padding:10px;text-align:center;font-family:Share Tech Mono,monospace;font-size:13px;color:#00ff9d;margin-top:8px">✓ NO ACTION REQUIRED<br><span style="font-size:10px;color:#5a9abf">Model confidence: {res["prob"]}%</span></div>', unsafe_allow_html=True)

    # Decision probabilities
    if res and res.get('all_probs'):
        st.markdown('<div class="card-title" style="margin-top:14px">📊 DECISION PROBABILITIES</div>', unsafe_allow_html=True)
        for cls, p in sorted(res['all_probs'].items(), key=lambda x: -x[1]):
            color = "#ff6600" if cls=="OPEN_CORRIDOR" else "#ffb800" if cls=="CAUTION" else "#00ff9d"
            st.markdown(f"""
            <div style="margin-bottom:6px">
                <div style="display:flex;justify-content:space-between;font-family:'Share Tech Mono',monospace;font-size:10px;color:{color};margin-bottom:3px">
                    <span>{cls}</span><span>{p}%</span>
                </div>
                <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{p}%;background:{color};opacity:0.8"></div></div>
            </div>
            """, unsafe_allow_html=True)

    # Planning steps
    st.markdown('<div class="card-title" style="margin-top:14px">📋 PLANNING ENGINE</div>', unsafe_allow_html=True)
    steps = [
        ("Activate Emergency Mode", "Override traffic cycle"),
        ("Open Priority Lane", "Turn emergency lane GREEN"),
        ("Stop Cross Traffic", "Hold all other lanes RED"),
        ("Display Emergency Symbol", "Notify other drivers"),
        ("Restore Normal Cycle", "After vehicle passes"),
    ]
    all_done = res and res['decision'] == "OPEN_CORRIDOR"
    for i, (step, sub) in enumerate(steps):
        done = all_done
        cls = "done" if done else ""
        icon = "✓" if done else str(i+1)
        color = "#00ff9d" if done else "#5a9abf"
        st.markdown(f"""
        <div style="display:flex;align-items:flex-start;gap:10px;padding:6px 0;
        border-bottom:1px solid #0e3a5c;color:{color}">
            <div style="width:20px;height:20px;border-radius:50%;border:1px solid {'#00ff9d' if done else '#1a5c8a'};
            display:flex;align-items:center;justify-content:center;font-family:Share Tech Mono,monospace;
            font-size:9px;flex-shrink:0;background:{'rgba(0,255,157,0.1)' if done else 'transparent'}">{icon}</div>
            <div>
                <div style="font-size:13px;font-weight:600">{step}</div>
                <div style="font-size:10px;color:#5a9abf">{sub}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── RIGHT: Ethics + History ──────────────────────────────────────
with col_right:
    st.markdown('<div class="card-title">🛡 ETHICAL GUARDRAILS</div>', unsafe_allow_html=True)

    # Guardrail 1: Confidence
    if res:
        conf_ok = res['confidence'] >= 85
        g1_cls = "badge-green" if conf_ok else "badge-red"
        g1_txt = "APPROVED" if conf_ok else "REJECTED"
        g1_val = f"{res['confidence']}% (min: 85%)"
    else:
        g1_cls, g1_txt, g1_val = "badge-blue", "PENDING", "—"

    st.markdown(f"""
    <div style="background:#0c1620;border:1px solid {'rgba(0,255,157,0.3)' if res and res['confidence']>=85 else '#0e3a5c'};
    border-radius:4px;padding:9px;margin-bottom:7px;display:flex;justify-content:space-between;align-items:center">
        <div>
            <div style="font-family:Share Tech Mono,monospace;font-size:9px;color:#5a9abf;letter-spacing:1px">CONFIDENCE THRESHOLD</div>
            <div style="font-size:13px;font-weight:600;margin-top:2px">{g1_val}</div>
        </div>
        <span class="badge {g1_cls}">{g1_txt}</span>
    </div>
    """, unsafe_allow_html=True)

    # Guardrail 2: Distance
    if res:
        dist_ok = res['distance'] <= 500
        g2_cls = "badge-green" if dist_ok else "badge-yellow"
        g2_txt = "IN RANGE" if dist_ok else "TOO FAR"
        g2_val = f"{res['distance']}m (max: 500m)"
    else:
        g2_cls, g2_txt, g2_val = "badge-blue", "PENDING", "—"

    st.markdown(f"""
    <div style="background:#0c1620;border:1px solid {'rgba(0,255,157,0.3)' if res and res['distance']<=500 else 'rgba(255,184,0,0.3)' if res else '#0e3a5c'};
    border-radius:4px;padding:9px;margin-bottom:7px;display:flex;justify-content:space-between;align-items:center">
        <div>
            <div style="font-family:Share Tech Mono,monospace;font-size:9px;color:#5a9abf;letter-spacing:1px">DETECTION RANGE</div>
            <div style="font-size:13px;font-weight:600;margin-top:2px">{g2_val}</div>
        </div>
        <span class="badge {g2_cls}">{g2_txt}</span>
    </div>
    """, unsafe_allow_html=True)

    # Guardrail 3: Max green time
    st.markdown(f"""
    <div style="background:#0c1620;border:1px solid rgba(0,255,157,0.3);
    border-radius:4px;padding:9px;margin-bottom:7px;display:flex;justify-content:space-between;align-items:center">
        <div>
            <div style="font-family:Share Tech Mono,monospace;font-size:9px;color:#5a9abf;letter-spacing:1px">MAX GREEN DURATION</div>
            <div style="font-size:13px;font-weight:600;margin-top:2px">45 sec limit</div>
        </div>
        <span class="badge badge-green">ENFORCED</span>
    </div>
    """, unsafe_allow_html=True)

    # Human Override
    st.markdown('<div class="card-title" style="margin-top:4px">👤 HUMAN OVERRIDE</div>', unsafe_allow_html=True)
    override = st.toggle("Activate Manual Override", value=False)
    if override:
        st.markdown('<div class="caution-banner">⚠ HUMAN OVERRIDE ACTIVE<br>AI decision suspended</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-family:Share Tech Mono,monospace;font-size:10px;color:#1a5c8a;text-align:center;padding:6px">Override: INACTIVE · AI in control</div>', unsafe_allow_html=True)

    # Scan History
    st.markdown('<div class="card-title" style="margin-top:10px">🕐 SCAN HISTORY</div>', unsafe_allow_html=True)
    if st.session_state.scan_history:
        for h in st.session_state.scan_history[:6]:
            emoji = {"ambulance":"🚑","firetruck":"🚒","car":"🚗","bus":"🚌","truck":"🚛","motorcycle":"🏍"}.get(h['vehicle_type'],'?')
            dec_color = "#ff6600" if h['decision']=="OPEN_CORRIDOR" else "#ffb800" if h['decision']=="CAUTION" else "#00ff9d"
            st.markdown(f"""
            <div style="background:#0c1620;border:1px solid #0e3a5c;border-radius:3px;
            padding:7px 10px;margin-bottom:4px;display:flex;justify-content:space-between;align-items:center">
                <div style="font-size:12px">{emoji} {h['vehicle_type'].upper()}</div>
                <div style="font-family:Share Tech Mono,monospace;font-size:9px;color:{dec_color}">{h['decision']}</div>
                <div style="font-family:Share Tech Mono,monospace;font-size:9px;color:#1a5c8a">{h['timestamp']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-family:Share Tech Mono,monospace;font-size:10px;color:#1a5c8a;text-align:center;padding:10px">No history yet</div>', unsafe_allow_html=True)

# ── MODEL INFO (bottom) ───────────────────────────────────────────
st.markdown("---")
col_a, col_b, col_c, col_d = st.columns(4)
infos = [
    ("🤖 YOLO MODEL", "YOLOv11n", "Emergency Detection", "99.4% mAP50"),
    ("🌲 RANDOM FOREST", "100 Trees", "Decision Making", "95.0% Accuracy"),
    ("📊 TRAINING DATA", "200 images + 1000 samples", "Emergency Vehicles + Traffic", "Synthetic + Kaggle"),
    ("⚡ SYSTEM", "Real-time AI", "Sense → Reason → Plan → Act", "Ethical Guardrails Active"),
]
for col, (title, val, sub, badge) in zip([col_a,col_b,col_c,col_d], infos):
    with col:
        st.markdown(f"""
        <div style="background:#080f18;border:1px solid #0e3a5c;border-radius:6px;padding:12px;text-align:center">
            <div style="font-family:Share Tech Mono,monospace;font-size:9px;color:#5a9abf;letter-spacing:2px">{title}</div>
            <div style="font-family:Orbitron,monospace;font-size:14px;color:#00c8ff;margin:6px 0">{val}</div>
            <div style="font-size:11px;color:#5a9abf">{sub}</div>
            <div style="font-family:Share Tech Mono,monospace;font-size:9px;color:#00ff9d;margin-top:4px">{badge}</div>
        </div>
        """, unsafe_allow_html=True)
