import pandas as pd

# -------------------------
# Load the dataset
# -------------------------
df = pd.read_csv("dataset/Housing.csv")

# -------------------------
# Basic Exploration
# -------------------------
print("First 5 Rows")
print(df.head())

print("\nShape of Dataset")
print(df.shape)

print("\nColumn Names")
print(df.columns)

print("\nDataset Information")
df.info()

print("\nStatistical Summary")
print(df.describe())

# -------------------------
# Separate Features and Target
# -------------------------
X = df.drop("price", axis=1)
y = df["price"]

print("\nFeatures (X)")
print(X.head())

print("\nTarget (y)")
print(y.head())

# -------------------------
# Check Unique Values
# -------------------------
print("\nMainroad Values")
print(df["mainroad"].unique())

print("\nGuestroom Values")
print(df["guestroom"].unique())

print("\nBasement Values")
print(df["basement"].unique())

print("\nAirconditioning Values")
print(df["airconditioning"].unique())

print("\nPreferred Area Values")
print(df["prefarea"].unique())

print("\nFurnishing Status Values")
print(df["furnishingstatus"].unique())

binary_columns = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea"
]

for column in binary_columns:
    df[column] = df[column].map({"yes": 1, "no": 0})
df = pd.get_dummies(df, columns=["furnishingstatus"],dtype=int)

print(df.head())