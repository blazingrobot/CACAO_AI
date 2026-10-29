import pandas as pd

file_path = "datasets/cocoa_fermentation_training_dataset.csv"

print("Loading dataset...")

df = pd.read_csv(file_path)

print("\n==============================")
print("DATASET INFORMATION")
print("==============================")

print("\nColumns:")
print(df.columns.tolist())

print("\nNumber of rows and columns:")
print(df.shape)

print("\nFirst 10 rows:")
print(df.head(10).to_string())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\n==============================")
print("DONE")
print("==============================")