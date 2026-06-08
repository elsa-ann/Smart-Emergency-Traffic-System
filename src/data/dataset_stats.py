"""
Simple dataset stats helper. Usage:
python scripts/dataset_stats.py path/to/traffic_decisions.csv [path/to/emergency_train.csv]
"""
import sys
import pandas as pd
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/dataset_stats.py traffic_decisions.csv [emergency_train.csv]')
        return
    td = pd.read_csv(sys.argv[1])
    print('\n== Traffic decisions summary ==')
    print(td.describe(include='all'))
    print('\nVehicle type counts:')
    print(td['vehicle_type'].value_counts())
    print('\nDecision counts:')
    print(td['decision'].value_counts())
    if len(sys.argv) > 2:
        ev = pd.read_csv(sys.argv[2])
        print('\nEmergency train positive count:')
        print((ev['emergency_or_not'] == 1).sum(), '/', len(ev))

if __name__ == '__main__':
    main()
