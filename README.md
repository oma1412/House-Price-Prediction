# House Price Prediction

This is a machine learning project that predicts house prices based on features such as area, number of bedrooms, bathrooms, parking, air conditioning, and furnishing status.

I built this project to understand the basic workflow of a machine learning project, including data preprocessing, train-test splitting, model training, prediction, and evaluation.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Git and GitHub

## Dataset

The project uses a housing dataset containing different features of houses along with their prices.

The `price` column is the target variable that the model tries to predict.

## Approach

First, I loaded and explored the dataset using Pandas. The categorical `yes/no` columns were converted into `1/0`, and One-Hot Encoding was used for the `furnishingstatus` column.

The data was then divided into training and testing sets using an 80-20 split.

I used a Linear Regression model and trained it on the training data. The trained model was then tested on unseen test data.

## Model Performance

The model achieved the following results:

| Metric | Result |
|---|---:|
| MAE | 970,043.40 |
| MSE | 1,754,318,687,330.66 |
| R² Score | 0.6529 |

The R² score shows that the model explains approximately 65% of the variation in house prices in the test data.

## House Price Prediction

The program also allows the user to enter details of a house, such as:

- Area
- Bedrooms
- Bathrooms
- Stories
- Parking
- Air conditioning
- Basement
- Furnishing status

The trained model then predicts an estimated price for the house.

## Project Structure

```text
House-Price-Prediction/
│
├── dataset/
│   └── Housing.csv
│
├── src/
│   └── main.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the project from the root directory:

```bash
python src/main.py
```

Then enter the house details when prompted to get the predicted price.

## What I Learned

Through this project, I learned the basics of:

- Data preprocessing using Pandas
- Encoding categorical data
- Splitting data into training and testing sets
- Training a Linear Regression model
- Making predictions on unseen data
- Evaluating a regression model using MAE, MSE, and R² Score
- Using Git and GitHub to track project development