import pandas as pd

df = pd.read_csv("data/ai4i2020.csv")

print("=== SHAPE ===")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n=== COLUMN NAMES ===")
for col in df.columns:
    print(f"  {col}")

print("\n=== FIRST 3 ROWS ===")
print(df.head(3).to_string())

print("\n=== DATA TYPES ===")
print(df.dtypes)

print("\n=== FAILURE TYPE COUNTS ===")
failure_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]
for col in failure_cols:
    print(f"  {col}: {df[col].sum()} cases")
print(f"  No Failure: {(df['Machine failure'] == 0).sum()} cases")

print("\n=== SENSOR VALUE RANGES ===")
sensor_cols = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]
for col in sensor_cols:
    print(f"  {col}: min={df[col].min():.1f}, max={df[col].max():.1f}, mean={df[col].mean():.1f}")