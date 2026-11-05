import pandas as pd

df = pd.read_csv("general_4.csv")
df["expected_service"] = "general"
df.to_csv("../general_dataset/general_4.csv", index=False)
print(df["expected_service"].value_counts())