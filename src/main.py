import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load Dataset

df = pd.read_csv("dataset/Housing.csv")

# Display Dataset Information

print(" FIRST 5 ROWS ")
print(df.head())

print("\n SHAPE ")
print(df.shape)

print("\n COLUMNS ")
print(df.columns)

print("\n INFO ")
df.info()

print("\n DESCRIPTION ")
print(df.describe())

# Encode Binary Columns (Yes / No)

binary_columns = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea"
]

for column in binary_columns:
    df[column] = df[column].map({
        "yes": 1,
        "no": 0
    })

# One Hot Encoding

df = pd.get_dummies(
    df,
    columns=["furnishingstatus"],
    dtype=int
)

print("\n DATA AFTER ENCODING ")
print(df.head())

# Separate Features and Target

X = df.drop("price", axis=1)
y = df["price"]

print("\n FEATURES ")
print(X.head())

print("\n TARGET ")
print(y.head())

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n TRAINING DATA ")
print(X_train.shape)

print("\n TESTING DATA ")
print(X_test.shape)

print("\n TRAINING TARGET ")
print(y_train.shape)

print("\n TESTING TARGET")
print(y_test.shape)

# Create Linear Regression Model

model = LinearRegression()

# Train the Model

model.fit(X_train, y_train)

print("\n Model Trained Successfully!")
