from fastapi.testclient import TestClient
from main import app  # Import your FastAPI app instance

# Create a TestClient for your app
client = TestClient(app)

def test_read_root_endpoint():
    """ Tests the root '/' endpoint. """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Model Prediction API!"}

def test_predict_endpoint():
    """
    Tests the '/predict' endpoint with a valid sample payload.
    """
    # This is a sample input that matches your `InputFeatures` Pydantic model
    sample_payload = {
        "feature1": 5.1,
        "feature2": 3.5,
        "feature3": 1.4,
        "feature4": 0.2
    }

    # Make a POST request to the '/predict' endpoint with the sample data
    response = client.post("/predict", json=sample_payload)

    # --- THIS IS THE KEY DEBUGGING CHANGE ---
    # If the test fails, print the detailed error from the API response body.
    if response.status_code != 200:
        print(f"API Error Response: {response.json()}")

    # Check if the HTTP status code is 200 (OK)
    assert response.status_code == 200