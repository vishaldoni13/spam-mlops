#from sklearn.linear_model import LogisticRegression
#import pickle
#
#X, y = pickle.load(open("data/processed/features.pkl", "rb"))
#
#model = LogisticRegression()
#model.fit(X, y)

#pickle.dump(model, open("models/model.pkl", "wb"))


import pickle
from sklearn.linear_model import LogisticRegression
import os

# Load training data
X_train, y_train = pickle.load(
    open("data/processed/train.pkl", "rb")
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
os.makedirs("models", exist_ok=True)
pickle.dump(model, open("models/model.pkl", "wb"))

