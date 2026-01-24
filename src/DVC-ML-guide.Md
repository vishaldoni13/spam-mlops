📘 End-to-End Spam Classification with DVC (Production-Grade, Step by Step)
This document is a complete, corrected, and production-aligned guide to building an end-to-end ML pipeline using DVC for a Spam Identification problem.
It reflects the latest working state of our project, including: - Correct feature engineering - Proper train/test split - Correct evaluation logic - A clean, valid DVC DAG - Real-world MLOps explanations (what, why, how)
This README is written as if you are onboarding a new MLOps engineer onto the project.

🧩 Problem Statement
We want to build a Spam Classification ML system.
Input: Text message
Output: Spam (1) or Not Spam (0)
The focus is not just accuracy, but: - Reproducibility - Versioning - Automation - CI/CD readiness

🏗️ Final Project Structure
spam-mlops/
├── data/
│   ├── raw/
│   │   └── spam.csv                # Immutable raw data
│   └── processed/
│       ├── ingested.csv            # Ingestion checkpoint
│       ├── clean.csv               # Cleaned data
│       ├── features.pkl            # TF-IDF features
│       ├── train.pkl               # Training split
│       └── test.pkl                # Test split
│
├── src/
│   ├── data_ingestion.py
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── train_test_split.py
│   ├── train.py
│   └── evaluation.py
│
├── models/
│   ├── vectorizer.pkl              # Text → numeric transformer
│   └── model.pkl                   # Trained ML model
│
├── dvclive/
│   └── metrics.json                # Evaluation metrics
│
├── dvc.yaml                        # Pipeline definition
├── dvc.lock                        # Reproducibility contract
└── README.md

🔁 High-Level Pipeline Flow
Raw Data
   ↓
Data Ingestion
   ↓
Data Processing
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Model Evaluation
Each arrow represents a DVC stage boundary.

🧠 Why We Use DVC Here
DVC provides: - Git-like versioning for data & models - Reproducible ML pipelines - Hash-based dependency tracking - CI/CD-friendly execution
❗ DVC tracks files, not in-memory variables.

1️⃣ Data Ingestion Stage
📄 src/data_ingestion.py
import pandas as pd

# Read raw data (source of truth)
df = pd.read_csv("data/raw/spam.csv")

# Write ingestion checkpoint
# This creates a pipeline boundary
df.to_csv("data/processed/ingested.csv", index=False)
🧠 Explanation
No transformation happens here
The file may look identical to raw data — this is expected
Purpose:
Validate data can be read
Decouple downstream stages from raw source

2️⃣ Data Processing Stage
📄 src/data_processing.py
import pandas as pd

df = pd.read_csv("data/processed/ingested.csv")

# Ensure labels are numeric
df["label"] = df["label"].astype(int)

# Save cleaned data
df.to_csv("data/processed/clean.csv", index=False)
🧠 Explanation
This stage is responsible for: - Type enforcement - Validation - Cleaning
In real projects, this is where: - Missing values - Schema checks - Deduplication
would live.

3️⃣ Feature Engineering Stage (CRITICAL)
📄 src/feature_engineering.py
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle
import os

# Load cleaned data
df = pd.read_csv("data/processed/clean.csv")

# Convert text to numeric features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])

y = df["label"]

# Ensure output directories exist
os.makedirs("models", exist_ok=True)

# Save vectorizer for inference
with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Save features for training stage
with open("data/processed/features.pkl", "wb") as f:
    pickle.dump((X, y), f)
🧠 Explanation (Line by Line)
ML models cannot read text
TF-IDF converts text → numbers
Vectorizer must be saved for inference
Features must be written to disk for DVC tracking
This stage is where ML truly begins.

4️⃣ Train/Test Split Stage
📄 src/train_test_split.py
import pickle
from sklearn.model_selection import train_test_split

X, y = pickle.load(open("data/processed/features.pkl", "rb"))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

with open("data/processed/train.pkl", "wb") as f:
    pickle.dump((X_train, y_train), f)

with open("data/processed/test.pkl", "wb") as f:
    pickle.dump((X_test, y_test), f)
🧠 Why This Is a Separate Stage
Changing test size should NOT recompute features
This separation enables efficient experiments

5️⃣ Model Training Stage
📄 src/train.py
import pickle
from sklearn.linear_model import LogisticRegression
import os

X_train, y_train = pickle.load(
    open("data/processed/train.pkl", "rb")
)

model = LogisticRegression()
model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
pickle.dump(model, open("models/model.pkl", "wb"))
🧠 Explanation
Training uses ONLY training data
Output is a versioned model artifact

6️⃣ Model Evaluation Stage
📄 src/evaluation.py
import pickle
from sklearn.metrics import accuracy_score
from dvclive import Live

model = pickle.load(open("models/model.pkl", "rb"))
X_test, y_test = pickle.load(
    open("data/processed/test.pkl", "rb")
)

preds = model.predict(X_test)

with Live() as live:
    live.log_metric("accuracy", accuracy_score(y_test, preds))
🧠 Explanation
Evaluation uses unseen test data
Metrics are logged via DVCLive
DVC versions metrics automatically

7️⃣ DVC Pipeline Definition (dvc.yaml)
stages:
  ingest:
    cmd: python src/data_ingestion.py
    deps:
      - src/data_ingestion.py
      - data/raw/spam.csv
    outs:
      - data/processed/ingested.csv

  process:
    cmd: python src/data_processing.py
    deps:
      - src/data_processing.py
      - data/processed/ingested.csv
    outs:
      - data/processed/clean.csv

  features:
    cmd: python src/feature_engineering.py
    deps:
      - src/feature_engineering.py
      - data/processed/clean.csv
    outs:
      - data/processed/features.pkl
      - models/vectorizer.pkl

  split:
    cmd: python src/train_test_split.py
    deps:
      - src/train_test_split.py
      - data/processed/features.pkl
    outs:
      - data/processed/train.pkl
      - data/processed/test.pkl

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/processed/train.pkl
    outs:
      - models/model.pkl

  evaluate:
    cmd: python src/evaluation.py
    deps:
      - src/evaluation.py
      - models/model.pkl
      - data/processed/test.pkl
    metrics:
      - dvclive/metrics.json

8️⃣ DVC DAG (Validated)
dvc dag
Confirms: - Correct dependencies - Proper branching - CI-safe execution order

9️⃣ How to Run the Project (Correct Way)
dvc repro
DO NOT run individual scripts in production.

🔐 Reproducibility Guarantee
dvc.lock captures exact hashes
Same inputs → same outputs
Works across machines and CI systems

✅ Key Takeaways
DVC enforces correct ML architecture
Each stage has a clear contract
Pipelines are reproducible and debuggable
This setup is production-ready

🚀 Next Steps
params.yaml for hyperparameters
dvc exp run for experiments
GitHub Actions CI
MLflow integration
Deployment with FastAPI + Docker

👨‍🏫 You now have a real-world MLOps-ready DVC project.
