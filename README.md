# Reviewer verification materials — AIIG-D-26-00130 (50-species revision)

```bash
pip install -r requirements.txt
python verify_claims.py
```

No GPU, weights or network access are needed. `CLAIMS.md` lists what is checked. `data/splits/` holds the frozen
specimen partition and its provenance, `data/splits/audit_overlap_39.csv` the pre-registered external cohort,
`results/<arm>/` the predictions and summaries of both arms, `tables/` and `figures/` the manuscript assets, and
`scripts/` the code that produced them from the predictions.
