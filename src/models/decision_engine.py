"""Decision engine for emergency corridor planning."""
import joblib
import pandas as pd
from pathlib import Path

class DecisionEngine:
    def __init__(self, model_path="Trained_Model/decision_model.pkl", 
                 encoder_path="Trained_Model/label_encoders.pkl"):
        """Initialize decision model and encoders."""
        self.model_path = Path(model_path)
        self.encoder_path = Path(encoder_path)
        
        try:
            self.model = joblib.load(self.model_path)
            self.encoders = joblib.load(self.encoder_path)
        except Exception as e:
            print(f"Error loading models: {e}")
            self.model = None
            self.encoders = None
    
    def predict(self, vehicle_type, traffic_density, time_of_day, weather,
                distance_m, confidence_pct, vehicle_count):
        """Predict corridor action: OPEN_CORRIDOR, CAUTION, or NO_ACTION."""
        if self.model is None or self.encoders is None:
            return "CAUTION", 0.0
        
        try:
            enc = self.encoders
            features = {
                'vehicle_type_enc': enc['vehicle_type'].transform([vehicle_type])[0],
                'traffic_density_enc': enc['traffic_density'].transform([traffic_density])[0],
                'time_of_day_enc': enc['time_of_day'].transform([time_of_day])[0],
                'weather_enc': enc['weather'].transform([weather])[0],
                'distance_m': distance_m,
                'confidence_pct': confidence_pct,
                'vehicle_count': vehicle_count,
            }
            X = pd.DataFrame([features])
            pred_enc = self.model.predict(X)[0]
            pred_prob = self.model.predict_proba(X)[0]
            decision = enc['decision'].inverse_transform([pred_enc])[0]
            max_prob = round(float(max(pred_prob)) * 100, 1)
            return decision, max_prob
        except Exception as e:
            print(f"Prediction error: {e}")
            return "CAUTION", 0.0
