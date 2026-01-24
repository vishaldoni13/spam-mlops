from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle
import os

# 1️⃣ Load cleaned data produced by the previous stage
df = pd.read_csv("data/processed/clean.csv")

# 2️⃣ Initialize TF-IDF vectorizer
# This converts text into numerical features
vectorizer = TfidfVectorizer()

# 3️⃣ Learn vocabulary + transform text into vectors
X = vectorizer.fit_transform(df["text"])

# 4️⃣ Extract labels
y = df["label"]

# 5️⃣ Ensure output directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# 6️⃣ Save the trained vectorizer
# Needed later during inference to transform new text
with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# 7️⃣ Save features + labels for training stage
# This is the missing piece that caused your error
with open("data/processed/features.pkl", "wb") as f:
    pickle.dump((X, y), f)

