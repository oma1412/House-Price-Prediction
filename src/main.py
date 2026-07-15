import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =====================================================
# Load Dataset
# =====================================================
df = pd.read_csv("dataset/Housing.csv")

# =====================================================
# Dataset Information
# =====================================================
print("========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== SHAPE ==========")
print(df.shape)

print("\n========== COLUMNS ==========")
print(df.columns)

print("\n========== INFO ==========")
df.info()

print("\n========== DESCRIPTION ==========")
print(df.describe())

# =====================================================
# Encode Yes/No Columns
# =====================================================
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

# =====================================================
# One-Hot Encoding
# =====================================================
df = pd.get_dummies(
    df,
    columns=["furnishingstatus"],
    dtype=int
)

print("\n========== DATA AFTER ENCODING ==========")
print(df.head())

# =====================================================
# Features and Target
# =====================================================
X = df.drop("price", axis=1)
y = df["price"]

print("\n========== FEATURES ==========")
print(X.head())

print("\n========== TARGET ==========")
print(y.head())

# =====================================================
# Train-Test Split
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n========== TRAINING FEATURES ==========")
print(X_train.shape)

print("\n========== TESTING FEATURES ==========")
print(X_test.shape)

print("\n========== TRAINING TARGET ==========")
print(y_train.shape)

print("\n========== TESTING TARGET ==========")
print(y_test.shape)

# =====================================================
# Linear Regression Model
# =====================================================
model = LinearRegression()

# =====================================================
# Train Model
# =====================================================
model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# =====================================================
# Make Predictions
# =====================================================
predictions = model.predict(X_test)

print("\n========== FIRST 10 PREDICTIONS ==========")
print(predictions[:10])

print("\n========== ACTUAL PRICES ==========")
print(y_test.head(10).values)

# =====================================================
# Model Evaluation
# =====================================================
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n========== MODEL EVALUATION ==========")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R2 Score:", r2)