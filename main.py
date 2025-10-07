import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
import joblib
import pandas as pd
import os

# --- Model Loading ---
# We try to load the model at startup and store any errors.
model = None
model_load_error = None

# Check if the model file exists before trying to load it.
model_path = 'best_sentiment_model.pkl'
if not os.path.exists(model_path):
    model_load_error = f"Model file not found at path: {model_path}"
else:
    try:
        model = joblib.load(model_path)
    except Exception as e:
        # This is the crucial part: we capture the actual error.
        model_load_error = f"Error loading model with joblib: {str(e)}"

# Initialize the FastAPI app
app = FastAPI(
    title="ML Model Deployment API",
    description="An API to make predictions using a pre-trained model.",
    version="1.0.0"
)

# --- Pydantic Model for Input Validation ---
class InputFeatures(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "feature1": 5.1, "feature2": 3.5,
                "feature3": 1.4, "feature4": 0.2
            }
        }
    )
    feature1: float
    feature2: float
    feature3: float
    feature4: float

# --- API Endpoints ---
@app.get("/", tags=["General"])
def read_root():
    return {"message": "Welcome to the Model Prediction API!"}

@app.post("/predict", tags=["Prediction"])
def predict(input_features: InputFeatures):
    if model is None:
        # Now we return the specific error message we captured earlier.
        raise HTTPException(status_code=503, detail=model_load_error)

    try:
        features_dict = input_features.model_dump()
        input_df = pd.DataFrame([features_dict])
        prediction = model.predict(input_df)
        prediction_result = prediction.tolist()[0]
        return {"prediction": prediction_result, "input_features": features_dict}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)

