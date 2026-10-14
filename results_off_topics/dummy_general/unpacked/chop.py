import itertools
from pathlib import Path

import numpy as np
import pandas as pd


def chop_by_k(table, k):
    """Every combination of k specialists: routing_method, query, combo_id, specialists, score_1..score_k, gap."""

    score_cols = [c for c in table.columns if c.endswith("_scores")]

    parts = []
    for combo_id, combo in enumerate(itertools.combinations(score_cols, k)):
        part = table[["routing_method", "query"]].copy()
        part["combo_id"] = combo_id
        part["specialists"] = "|".join(c[: -len("_scores")] for c in combo)
        for i, col in enumerate(combo, start=1):
            part[f"score_{i}"] = table[col].values

        S = table[list(combo)].values
        part["gap"] = S.max(axis=1) - np.median(S, axis=1)

        parts.append(part)

    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    folder = Path(__file__).parent
    chopped_dir = folder / "chopped"
    chopped_dir.mkdir(exist_ok=True)

    for csv in sorted(folder.glob("*.csv")):
        table = pd.read_csv(csv)
        n = len([c for c in table.columns if c.endswith("_scores")])

        for k in range(3, n + 1):
            chopped = chop_by_k(table, k)
            chopped.to_csv(chopped_dir / f"{csv.stem}_k{k}.csv", index=False)
            print(f"{csv.stem} k={k}: {chopped['combo_id'].nunique()} combinations, {len(chopped)} rows")