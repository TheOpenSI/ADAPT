import ast
from pathlib import Path

import pandas as pd


def load_table(train_general_csv):
    """General rows only: routing_method, query, then one column per specialist."""

    data = pd.read_csv(train_general_csv, usecols=["routing_method", "query", "scores", "expected_service"])
    data = data[data["expected_service"] == "general"].reset_index(drop=True)

    # unpack scores
    scores = data["scores"].apply(ast.literal_eval).apply(pd.Series)
    scores.columns = [f"{c}_scores" for c in scores.columns]

    return pd.concat([data[["routing_method", "query"]], scores], axis=1)


if __name__ == "__main__":
    folder = Path(__file__).parent      
    out_dir = folder / "unpacked"
    out_dir.mkdir(exist_ok=True)

    for csv in sorted(folder.glob("*.csv")):
        table = load_table(csv)
        table.to_csv(out_dir / csv.name, index=False)
        print(f"{csv.name}: {len(table)} rows, {table.shape[1] - 2} specialists")