# Ethical Guardrails for SECS (Smart Emergency Corridor System)

This document lists the safety and ethical constraints enforced by the system.

## Overview
SECS uses vision and decision models to open an emergency corridor. Guardrails ensure the system acts reliably, transparently, and safely.

## Hard Constraints
- Minimum confidence to activate corridor: **85%**
- Maximum activation distance: **500 meters** (detections beyond this require human confirmation)
- Maximum emergency green duration: **45 seconds** (to avoid starving cross-traffic)
- Human override always allowed and logged
- System must maintain an audit trail for every AI decision (input snapshot, model outputs, timestamp, operator decisions)

## Operational Rules
1. Only `ambulance` and `firetruck` are considered emergency by default. Other vehicle types require higher evidence (multiple frames or cross-sensor confirmation).
2. If confidence < 85% but distance < 100m, escalate to CAUTION state (notify operator), do not auto-open corridor.
3. If traffic density is `jam` and vehicle_count > 20, do not automatically force open corridor without operator confirmation.
4. Maximum of one corridor activation per direction within a 2-minute window to avoid oscillation.

## Ethical Considerations
- Fairness: Ensure dataset diversity to avoid geographic or temporal bias affecting detection accuracy.
- Accountability: Log model version and encoder versions with each decision for traceability.
- Transparency: Provide on-screen explanation of the top contributing signals for the decision (confidence, distance, vehicle_type, traffic_density).

## Monitoring & Auditing
- Save decision logs in `logs/decisions.csv` with fields: timestamp, model_version, vehicle_type, distance_m, confidence_pct, decision, operator_override, notes.
- Periodic evaluation (weekly) on held-out validation set to monitor drift.

## Human-in-the-loop
- Manual override interface must require authentication, reason for override, and storing the operator id.
- All overrides generate an automatic follow-up report for review.

## Emergency Handling
- If multiple sensors disagree, follow majority rule but escalate to human when disagreement persists for > 5 seconds.
- For any false positive leading to unsafe outcomes (e.g., causing collision), the incident must be recorded and trigger model retraining review.

## Contact & Ownership
- Maintainer: Project team
- For security/ethics issues: create an issue in GitHub and tag `ethics` and `safety`.
