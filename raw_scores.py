import os
import json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(BASE_DIR, "results_raw_scores")   # where the run saved its json files
OUT_DIR = os.path.join(BASE_DIR, "raw_scores")               # where the new csv files go
KS = [3, 5, 8, 10, 12, 14]

frames = []
for k in KS:
    folder = os.path.join(RESULTS_DIR, f"k{k}")
    if not os.path.isdir(folder):
        print(f"skipping k={k}: folder not found: {folder}")
        continue
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".json") or "_efficiency" in name:
            continue
        with open(os.path.join(folder, name)) as f:
            frames.append(pd.DataFrame(json.load(f)))

df = pd.concat(frames, ignore_index=True)

scores = pd.DataFrame(df["scores"].tolist()).add_suffix("_score")

raw = pd.concat([df[["model_name", "total_service_count", "expected_service"]], scores], axis=1)
raw = raw.rename(columns={"total_service_count": "k"})

for (model, k), part in raw.groupby(["model_name", "k"]):
    part = part.dropna(axis=1, how="all")        # drop specialists that are not in this K
    out_dir = os.path.join(OUT_DIR, f"k{k}")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, model.replace("/", "_") + ".csv")
    part.to_csv(path, index=False)
    print(f"saved {path}  ({len(part)} rows, {part.shape[1]} columns)")