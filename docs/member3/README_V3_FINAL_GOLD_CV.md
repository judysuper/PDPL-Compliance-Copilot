
# V3 Final Gold Cross-Validation

This is the correct final V3 evaluation after human-reviewed Gold labels replaced
the old provisional labels.

It retrains the V3 meta-classifier inside a nested Leave-One-Policy-Out evaluation:
- 10 policies total
- each outer fold holds out one entire policy
- hyperparameters are selected using only the other 9 policies
- the held-out policy is never used for training or tuning in that fold

Run:
```powershell
python ".\src\gap_analyzer\gap_analyzer_v3_gold_cv.py"
```
