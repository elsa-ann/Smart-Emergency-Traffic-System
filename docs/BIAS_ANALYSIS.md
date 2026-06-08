# Dataset Bias Analysis — SECS

This document provides a step-by-step bias analysis procedure and a summary of known dataset characteristics.

## Purpose
Identify potential biases (class, temporal, geographic) and provide mitigation recommendations.

## How to run analysis locally
1. Install dependencies: `pip install pandas matplotlib seaborn`
2. Run the provided analysis script `scripts/dataset_stats.py` (or inspect CSVs manually).

### Quick commands
```bash
python scripts/dataset_stats.py Dataset/traffic_decisions.csv Dataset/emergency-vehicles/train.csv
```

## Recommended Checks
- Class balance (vehicle_type distribution)
- Decision balance (OPEN_CORRIDOR / CAUTION / NO_ACTION)
- Confidence distribution (confidence_pct histogram)
- Temporal distribution (time_of_day)
- Traffic density distribution
- Geographic coverage (source metadata if available)

## Example Findings (placeholder)
> Note: automated stat run failed in this environment (missing `pandas`). Run the script above locally to produce exact numbers.

- Class balance: *fill after running script*
- Decision balance: *fill after running script*
- Confidence: *fill after running script*

## Risks Identified (typical)
- Night-time underrepresentation → lower detection accuracy at night
- Geographic concentration (Bandung centric) → reduced generalization to other cities
- Few samples of `firetruck` vs `ambulance` → model may favor ambulance patterns

## Mitigation Recommendations
1. Collect additional night-time images (minimum +200 samples)
2. Collect from at least two other cities to improve geographic coverage
3. Use synthetic augmentation (brightness, blur, rain) for edge cases
4. Re-balance training set or use class-weighting for decision model
5. Re-run bias analysis after each data collection round

## Deliverables produced
- `scripts/dataset_stats.py` (helper) — see `scripts/` folder to run automatic stats
- `Documentation/BIAS_ANALYSIS.md` (this file) — update with concrete numbers after running script

