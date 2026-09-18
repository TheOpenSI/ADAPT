import os
import sys 
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")


import pandas as pd
from sklearn.metrics import roc_curve


for k in [5, 8, 10]:

    csv_path = (
        f"results/validation_service_{k}/"
        "cross-encoder_Qwen-Qwen3-Reranker-4B.csv"
        )

    df = pd.read_csv(csv_path)

    y_true = (
        df["expected_service"] != "general"
    ).astype(int)

    y_score = df["top_zscore"]

    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_score
    )

    youden_j = tpr - fpr

    best_idx = youden_j.argmax()

    best_threshold = thresholds[best_idx]

    print("\n=== ROC THRESHOLD ANALYSIS ===")
    print(f"Rows: {len(df)}")
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"Best TPR: {tpr[best_idx]:.4f}")        # was: {tpr.4f}
    print(f"Best FPR: {fpr[best_idx]:.4f}")        # was: {fpr.4f}
    print(f"Best Youden J: {youden_j[best_idx]:.4f}")   # was: {youden_j.4f}

    print("\nTop Z-score Distribution")
    print(df["top_zscore"].describe())