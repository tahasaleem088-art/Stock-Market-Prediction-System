import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load dataset
df = pd.read_csv("data/AAPL.csv")

# Convert target
df["direction"] = df["direction"].map({
    "Increasing": 1,
    "Decreasing": 0
})

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

data = df[features].values
target = df["direction"].values

# Create sequences
SEQ_LEN = 10

X = []
y = []

for i in range(SEQ_LEN, len(data)):
    X.append(data[i-SEQ_LEN:i])
    y.append(target[i])

X = np.array(X)
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ---------------- LSTM MODEL ----------------
model = Sequential()

model.add(LSTM(50, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(1, activation="sigmoid"))

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1)

# Predict
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

# ---------------- EVALUATION ----------------
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n🟢 LSTM RESULTS")
print("Accuracy:", acc)
print("F1 Score:", f1)

# ---------------- CONFUSION MATRIX ----------------
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# y_pred from Keras is shape (n, 1); flatten it to 1-D so it lines up with y_test
y_pred = y_pred.ravel()

# Build the matrix: rows = actual, cols = predicted
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
disp.plot(cmap="Blues", values_format="d")
plt.title("LSTM Confusion Matrix")
plt.tight_layout()
plt.savefig("lstm_confusion_matrix.png", dpi=150)
plt.show()