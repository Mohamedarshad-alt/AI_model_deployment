from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Create FastAPI application
app = FastAPI()

# Load the trained AI model
model = joblib.load("model.pkl")


# Define the data we expect from the user
class HouseData(BaseModel):
    area: float
    bedrooms: int


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "AI House Price Prediction API is running"
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: HouseData):

    # Send the user's data to the AI model
    prediction = model.predict([
        [data.area, data.bedrooms]
    ])

    # Return the prediction
    return {
        "area": data.area,
        "bedrooms": data.bedrooms,
        "predicted_price_lakhs": round(float(prediction[0]), 2)
    }