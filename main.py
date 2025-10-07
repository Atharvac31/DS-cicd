import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from typing import List

# Initialize the FastAPI app
app = FastAPI(
    title="ML Model Deployment API",
    description="An API to make predictions using a pre-trained model.",
    version="1.0.0"
)

# --- Pydantic Model for Input Validation ---
# The user MUST update this class to match the features their model expects.
# I'm using generic feature names as placeholders.
class InputFeatures(BaseModel):
    """
    Defines the structure and validation for the input data.
    Each field corresponds to a feature your model was trained on.
    """
    feature1: float = Field(..., example=5.1, description="Example feature 1 (e.g., sepal length)")
    feature2: float = Field(..., example=3.5, description="Example feature 2 (e.g., sepal width)")
    feature3: float = Field(..., example=1.4, description="Example feature 3 (e.g., petal length)")
    feature4: float = Field(..., example=0.2, description="Example feature 4 (e.g., petal width)")

    # The 'Config' class provides examples for the FastAPI documentation.
    class Config:
        json_schema_extra = {
            "example": {
                "feature1": 5.1,
                "feature2": 3.5,
                "feature3": 1.4,
                "feature4": 0.2
            }
        }


# --- Load The Model ---
# Load your pre-trained model from the .pkl file.
# Ensure 'best_sentiment_model.pkl' is in the same directory as this script.
try:
    model = joblib.load('best_sentiment_model.pkl')
except FileNotFoundError:
    model = None
    print("Error: 'best_sentiment_model.pkl' not found. Please place your model file in the correct directory.")
except Exception as e:
    model = None
    print(f"An error occurred while loading the model: {e}")


# --- API Endpoints ---
@app.get("/", tags=["General"])
def read_root():
    """
    A simple root endpoint to check if the API is running.
    """
    return {"message": "Welcome to the Model Prediction API!"}


@app.post("/predict", tags=["Prediction"])
def predict(input_features: InputFeatures):
    """
    Endpoint to make a prediction.

    Takes a JSON object with model features and returns a prediction.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Cannot make predictions.")

    try:
        # Convert the Pydantic model to a dictionary, then to a DataFrame
        # The model expects a 2D array-like input, so we create a DataFrame with a single row.
        features_dict = input_features.model_dump()
        input_df = pd.DataFrame([features_dict])

        # Ensure the column order matches the order used during model training.
        # This is a good practice, though often not strictly necessary if the dict keys match.
        # required_columns = ['feature1', 'feature2', 'feature3', 'feature4']
        # input_df = input_df[required_columns]

        # Make a prediction
        prediction = model.predict(input_df)

        # The prediction is often a numpy array, so we convert it to a list
        # and extract the first element.
        prediction_result = prediction.tolist()[0]

        return {
            "prediction": prediction_result,
            "input_features": features_dict
        }
    except Exception as e:
        # Catch potential errors during prediction (e.g., data format issues)
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


# This block allows you to run the app directly with `python main.py` for local testing.
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
