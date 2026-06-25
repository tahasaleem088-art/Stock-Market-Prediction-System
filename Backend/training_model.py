import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/AAPL.csv")

# Select features (inputs)
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

# Target column
y = df["direction"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print("Train set:", X_train.shape)
print("Test set:", X_test.shape)