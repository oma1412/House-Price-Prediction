import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load Dataset
df = pd.read_csv("dataset/Housing.csv")


# Encode Yes/No Columns
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


# One-Hot Encoding
df = pd.get_dummies(
    df,
    columns=["furnishingstatus"],
    dtype=int
)


# Separate Features and Target
X = df.drop("price", axis=1)
y = df["price"]


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create and Train Model
model = LinearRegression()
model.fit(X_train, y_train)


# Make Predictions on Test Data
predictions = model.predict(X_test)


# Evaluate Model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)


# Display Model Performance
print("\n========== MODEL PERFORMANCE ==========")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R2 Score: {r2:.4f}")


# Interactive House Price Prediction
print("\n========== HOUSE PRICE PREDICTION ==========")

area = float(input("Enter area: "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))
stories = int(input("Enter number of stories: "))

mainroad = int(input("Main road available? (1 = Yes, 0 = No): "))
guestroom = int(input("Guest room available? (1 = Yes, 0 = No): "))
basement = int(input("Basement available? (1 = Yes, 0 = No): "))
hotwaterheating = int(input("Hot water heating available? (1 = Yes, 0 = No): "))
airconditioning = int(input("Air conditioning available? (1 = Yes, 0 = No): "))

parking = int(input("Enter number of parking spaces: "))

prefarea = int(input("Preferred area? (1 = Yes, 0 = No): "))

furnishing = input(
    "Enter furnishing status (furnished/semi-furnished/unfurnished): "
).lower()


# Create Input Data
new_house = {
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "stories": stories,
    "mainroad": mainroad,
    "guestroom": guestroom,
    "basement": basement,
    "hotwaterheating": hotwaterheating,
    "airconditioning": airconditioning,
    "parking": parking,
    "prefarea": prefarea,
    "furnishingstatus_furnished": 0,
    "furnishingstatus_semi-furnished": 0,
    "furnishingstatus_unfurnished": 0
}


# Set Furnishing Status
if furnishing == "furnished":
    new_house["furnishingstatus_furnished"] = 1

elif furnishing == "semi-furnished":
    new_house["furnishingstatus_semi-furnished"] = 1

elif furnishing == "unfurnished":
    new_house["furnishingstatus_unfurnished"] = 1


# Convert Input to DataFrame
new_house_df = pd.DataFrame([new_house])

# Ensure Same Column Order as Training Data
new_house_df = new_house_df[X.columns]


# Predict House Price
predicted_price = model.predict(new_house_df)


# Display Result
print("\n======================================")
print(f"Predicted House Price: ₹{predicted_price[0]:,.2f}")
print("======================================")