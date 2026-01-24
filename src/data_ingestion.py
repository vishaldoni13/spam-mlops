import pandas as pd

# Step 1: Read raw CSV data
df = pd.read_csv("data/raw/spam.csv")

# Step 2: Save a copy (acts like ingestion stage)
df.to_csv("data/processed/ingested.csv", index=False)

