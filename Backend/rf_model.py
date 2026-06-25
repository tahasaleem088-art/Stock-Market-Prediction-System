import pandas as pd
import numpy as np
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("data/AAPL.csv")

# Encode target
df["direction"] = df["direction"].map({
    "Increasing": 1,
    "Decreasing": 0
})

# Features
features = [
    "AAPL.Open",
    "AAPL.High",
    "AAPL.Low",
    "AAPL.Close",
    "AAPL.Volume",
    "mavg",
    "dn",
    "up"
]

X = df[features]
y = df["direction"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ---------------- IMPROVED RANDOM FOREST ----------------
model = RandomForestClassifier(
    n_estimators=300,      # more trees = better stability
    max_depth=12,          # prevents overfitting
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n🟢 IMPROVED RANDOM FOREST RESULTS")
print("Accuracy:", acc)
print("F1 Score:", f1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ---------------- CONFUSION MATRIX ----------------
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# RF predictions are already 1-D class labels, so no flattening or >0.5 step needed
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix (rows = actual, cols = predicted):")
print(cm)

# Pull out the four counts so you can read them in your report / viva
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives  (SELL predicted SELL): {tn}")
print(f"False Positives (SELL predicted BUY) : {fp}")
print(f"False Negatives (BUY predicted SELL) : {fn}")
print(f"True Positives  (BUY predicted BUY)  : {tp}")

# Plot and save it as an image for the report
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["SELL (0)", "BUY (1)"]
)
disp.plot(cmap="Oranges", values_format="d")
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig("rf_confusion_matrix.png", dpi=150)
plt.show()
import joblib

joblib.dump(model, "rf_model.pkl")
plt.figure(figsize=(20,10))

plot_tree(
    model.estimators_[0],
    filled=True,
    feature_names=[f"Feature_{i}" for i in range(1, 9)],
    class_names=["SELL", "BUY"],
    rounded=True
)

plt.show()