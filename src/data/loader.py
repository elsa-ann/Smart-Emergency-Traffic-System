"""Data loading and preprocessing."""
import pandas as pd
from pathlib import Path

def load_traffic_decisions(path="Dataset/traffic_decisions.csv"):
    """Load traffic decisions dataset."""
    df = pd.read_csv(path)
    return df

def load_emergency_vehicles(path="Dataset/emergency-vehicles/train.csv"):
    """Load emergency vehicle labels."""
    df = pd.read_csv(path)
    return df
