import pickle
from sklearn.model_selection import train_test_split
import os

# Load full feature set
X, y = pickle.load(open("data/processed/features.pkl", "rb"))

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Ensure output directory exists
os.makedirs("data/processed", exist_ok=True)

# Save splits
with open("data/processed/train.pkl", "wb") as f:
    pickle.dump((X_train, y_train), f)

with open("data/processed/test.pkl", "wb") as f:
    pickle.dump((X_test, y_test), f)

