# Simulator Mode (app_sim.py) — Usage Guide

## Overview
`app_sim.py` adalah mode Streamlit untuk menguji AI decision model menggunakan dataset nyata atau skenario manual.

Berbeda dengan `app.html` (prototype):
- `app.html` → visual simulasi jalan raya penuh (kendaraan, lampu, CCTV)
- `app_sim.py` → test decision model dengan data CCTV-like (dataset playback)

## How to Run

### Via Streamlit Dashboard
1. `streamlit run Web_Prototype/app.py`
2. Sidebar → pilih `Simulator`
3. Muncul page `app_sim.py`

### Langsung
```bash
streamlit run Web_Prototype/app_sim.py
```

## Modes

### 1. Dataset Playback (default)
- Pilih row dari `Dataset/traffic_decisions.csv`
- Slider untuk pilih scenario
- Tampilkan decision history

**Use case**: Validation & testing terhadap real data

### 2. Random Scenario
- Generate random kendaraan, traffic, weather
- Klik tombol "Generate random case"
- Test model dengan berbagai kombinasi

**Use case**: Stress testing & edge case exploration

### 3. Manual Scenario
- Input custom:
  - Vehicle type
  - Traffic density
  - Time of day
  - Weather
  - Distance
  - Confidence
  - Vehicle count
- Jalankan prediction manual

**Use case**: Specific hypothesis testing

## Output

Untuk setiap scenario, `app_sim.py` menampilkan:

```
┌─ SCENARIO DETAILS ────────────────
│ Vehicle:        ambulance
│ Distance:       300m
│ Traffic:        medium
│ Confidence:     92%
│ Vehicles:       8
└──────────────────────────────────

┌─ AI DECISION ─────────────────────
│ Decision:       OPEN_CORRIDOR ✓
│ Confidence:     87.5%
│
│ Probabilities:
│ - OPEN_CORRIDOR:   87.5%
│ - CAUTION:         10.2%
│ - NO_ACTION:        2.3%
└──────────────────────────────────

┌─ SIMULATION LOG ──────────────────
│ Step 1. Detect approaching vehicle via CCTV
│ Step 2. Analyze vehicle type, distance, confidence
│ Step 3. Evaluate emergency priority rules
│ Step 4. Update signal timing and open corridor
│ Step 5. Monitor vehicle clearance and restore
│
│ Result: Emergency corridor activated.
└──────────────────────────────────
```

## Code Structure

```python
# Load models
models = load_models()  # decision_model.pkl + encoders

# Load dataset
dataset = load_dataset()  # traffic_decisions.csv

# Mode selector
if mode == "Dataset Playback":
    scenario = dataset.iloc[index].to_dict()
elif mode == "Random Scenario":
    scenario = generate_random()
else:  # Manual
    scenario = user_input()

# Predict
if st.button("Run AI prediction"):
    decision, prob, all_probs = predict_decision(scenario, models)
    display_result(decision, prob, all_probs)
```

## Tips

1. **Dataset Playback**: Lihat decision history & pattern
2. **Random Mode**: Lihat bagaimana model handle edge cases (night + jam + rain + far distance)
3. **Manual Mode**: Test specific hypothesis (e.g., "ambulance + 85% confidence + 500m = ?")

## Integration

Jika ingin menambahkan mode baru (e.g., live CCTV):

```python
mode = st.sidebar.radio("Simulation Mode", ["Random", "Dataset Playback", "Manual", "Live CCTV"])

if mode == "Live CCTV":
    camera_input = st.camera_input("Capture vehicle")
    if camera_input:
        # Extract vehicle info from image (YOLO)
        # Run decision model
        # Show result
```
