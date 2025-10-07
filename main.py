import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
import joblib
import pandas as pd
import os

# --- Model Loading ---
model = None
model_load_error = None
model_path = 'best_sentiment_model.pkl'

if not os.path.exists(model_path):
    model_load_error = f"Model file not found at path: {model_path}"
else:
    try:
        model = joblib.load(model_path)
    except Exception as e:
        model_load_error = f"Error loading model with joblib: {str(e)}"

# Initialize the FastAPI app
app = FastAPI(
    title="ML Model Deployment API",
    description="An API to make predictions using a pre-trained model.",
    version="1.0.0"
)

# --- Pydantic Model for Input Validation ---
# UPDATED to match the actual features your model expects.
class InputFeatures(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Category": "Food",
                "City": "New York",
                "Calories": 550.0,
                "Price": 25.99,
                "Offer_Type": "Discount",
                "Is_Premium": True
            }
        }
    )
    Category: str
    City: str
    Calories: float
    Price: float
    Offer_Type: str
    Is_Premium: bool

# --- API Endpoints ---
@app.get("/", tags=["General"])
def read_root():
    return {"message": "Welcome to the Model Prediction API!"}

@app.post("/predict", tags=["Prediction"])
def predict(input_features: InputFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail=model_load_error)

    try:
        features_dict = input_features.model_dump()
        input_df = pd.DataFrame([features_dict])
        
        # Make a prediction
        prediction = model.predict(input_df)
        prediction_result = prediction.tolist()[0]
        
        return {"prediction": prediction_result, "input_features": features_dict}
    except Exception as e:
        # We add this check to provide a more specific error message for missing columns.
        if "columns are missing" in str(e):
             raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during prediction: {str(e)}")

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)

