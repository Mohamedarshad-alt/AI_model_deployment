from sklearn.linear_model import LinearRegression
import joblib

# Training data
# Each row contains:
# [house area in square feet, number of bedrooms]

X = [
    [1000, 2],
    [1200, 2],
    [1500, 3],
    [1800, 3],
    [2000, 4],
]

# House prices in lakhs
y = [
    40,
    48,
    60,
    72,
    80,
]

# Create the machine learning model
model = LinearRegression()

# Train the model using the training data
model.fit(X, y)

# Save the trained model
joblib.dump(model, "model.pkl")

print("Model trained successfully!")
print("Model saved as model.pkl")