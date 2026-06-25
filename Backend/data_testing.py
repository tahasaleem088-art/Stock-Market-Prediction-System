import pandas as pd

# Path to dataset
file_path = "data/AAPL.csv"

# Load CSV file
df = pd.read_csv(file_path)

# Show basic info
print(" Dataset loaded successfully!\n")

print("\nColumns in dataset:")
print(df.columns)

print("\nDataset shape (rows, columns):")
print(df.shape)