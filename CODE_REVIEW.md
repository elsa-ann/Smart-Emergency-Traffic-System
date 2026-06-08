# Code Review Checklist — SECS

Use this checklist when performing a clean code review.

## Style & Readability
- [ ] Clear, descriptive function and variable names
- [ ] Docstrings for all public functions and classes
- [ ] No large functions (>200 lines) — break into helpers
- [ ] No commented-out dead code
- [ ] Consistent formatting (use `black` / `flake8` rules)

## Design & Architecture
- [ ] Single Responsibility per function/module
- [ ] No tight coupling between modules
- [ ] Clear module boundaries (`src/models`, `src/web`, `src/data`)
- [ ] Configs separated from code (`configs/*.yaml`)

## Safety & Guardrails
- [ ] Ethical constraints implemented (see `Documentation/GUARDRAILS.md`)
- [ ] Audit trail for decisions implemented
- [ ] Human override implemented and logged

## Tests & Validation
- [ ] Unit tests for core logic (decision rules, encoders)
- [ ] Integration test for Streamlit workflows (basic smoke test)
- [ ] Data validation tests for dataset CSVs (expected columns, types)

## Documentation
- [ ] README updated with run instructions
- [ ] CONTRIBUTING.md present
- [ ] Architecture diagram and API docs present

## Security
- [ ] No secrets in repository
- [ ] Dependencies pinned in `requirements.txt`

## Final
- [ ] All issues fixed and PR opened for review
