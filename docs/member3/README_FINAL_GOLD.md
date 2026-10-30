# Member 3 Final Gold Pack

Human-reviewed local-policy Gold set prepared from:
`PDPL_Gold_Review_150_Approved.xlsx`

- Policies: 10
- PDPL requirements: 15
- Decisions: 150
- Covered: 63
- Partial: 47
- Missing: 40

## Files
- `data/local_policies/gold/pdpl_local_policies_gold.csv`
- `data/local_policies/gold/gold_summary.json`
- `src/gap_analyzer/evaluate_v3_gold.py`

## Run final V3 evaluation
```powershell
python ".\src\gap_analyzer\evaluate_v3_gold.py"
```

This re-scores the already generated V3 out-of-fold predictions against the
human-reviewed Gold labels. It does not retrain or tune the model.
