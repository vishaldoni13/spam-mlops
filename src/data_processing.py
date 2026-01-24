import pandas as pd

df = pd.read_csv("data/processed/ingested.csv")

# Convert labels to int
df["label"] = df["label"].astype(int)

df.to_csv("data/processed/clean.csv", index=False)

