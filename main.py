import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import joblib
import pandas as pd

# Initialize the FastAPI app
app = FastAPI(
    title="ML Model Deployment API",
    description="An API to make predictions using a pre-trained model.",
    version="1.0.0"
)

# --- Pydantic Model for Input Validation (Updated for Pydantic V2) ---
class InputFeatures(BaseModel):
    """
    Defines the structure and validation for the input data.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "feature1": 5.1,
                "feature2": 3.5,
                "feature3": 1.4,
                "feature4": 0.2
            }
        }
    )

    feature1: float = Field(description="Example feature 1 (e.g., sepal length)")
    feature2: float = Field(description="Example feature 2 (e.g., sepal width)")
    feature3: float = Field(description="Example feature 3 (e.g., petal length)")
    feature4: float = Field(description="Example feature 4 (e.g., petal width)")


# --- Load The Model ---
# Load your pre-trained model from the .pkl file.
# Ensure 'best_sentiment_model.pkl' is in the same directory as this script.
try:
    model = joblib.load('best_sentiment_model.pkl')
except FileNotFoundError:
    model = None
    print("Error: 'best_sentiment_model.pkl' not found. The API will not be able to make predictions.")
except Exception as e:
    model = None
    print(f"An error occurred while loading the model: {e}")


# --- API Endpoints ---
@app.get("/", tags=["General"])
def read_root():
    """ A simple root endpoint to check if the API is running. """
    return {"message": "Welcome to the Model Prediction API!"}


@app.post("/predict", tags=["Prediction"])
def predict(input_features: InputFeatures):
    """ Endpoint to make a prediction. """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Cannot make predictions.")

    try:
        features_dict = input_features.model_dump()
        input_df = pd.DataFrame([features_dict])

        prediction = model.predict(input_df)
        prediction_result = prediction.tolist()[0]

        return {
            "prediction": prediction_result,
            "input_features": features_dict
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


# This block allows you to run the app directly with `python main.py` for local testing.
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)

