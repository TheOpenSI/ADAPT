from datasets import load_dataset
import pandas as pd
dataset = load_dataset(
    "mteb/amazon_massive_intent",
    split="validation"
)

df = pd.DataFrame(dataset)

df = df[df["lang"] == "en"]

print(
    df["label"]
    .value_counts()
    .sort_index()
)