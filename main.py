import ast
from pathlib import Path
import pandas as pd

ROOT = Path("raw_data")
OUTPUT = Path("output")

for f in sorted(ROOT.rglob("*.csv")):
    df = pd.read_csv(f)
    if "scores" not in df.columns:
        print(f"skip (no scores column): {f}")
        continue

    # unpack scores: one column per service, named <service>_scores
    scores = pd.DataFrame([ast.literal_eval(s) if isinstance(s, str) else {} for s in df["scores"]])
    scores = scores[sorted(scores.columns)].add_suffix("_scores")

    # keep only expected_service + the score columns
    out = pd.concat([df[["expected_service"]].reset_index(drop=True), scores], axis=1)

    # save to output/<same folder>/<same name>_cleaned.csv
    dest = OUTPUT / f.relative_to(ROOT).parent / f"{f.stem}_cleaned.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"{dest}: {len(out)} rows, {scores.shape[1]} services")