import pickle
from sklearn.metrics import accuracy_score
from dvclive import Live

# Load trained model
model = pickle.load(open("models/model.pkl", "rb"))

# Load test data (NOW IT EXISTS)
X_test, y_test = pickle.load(
    open("data/processed/test.pkl", "rb")
)

# Predict
preds = model.predict(X_test)

# Log metrics with DVC
with Live(save_dvc_exp=True) as live:
    live.log_metric("accuracy", accuracy_score(y_test, preds))

