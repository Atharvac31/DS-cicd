from fastapi.testclient import TestClient
# We import the 'app' instance from your main.py file
from main import app

# Create a TestClient to make simulated requests to your API
client = TestClient(app)

def test_read_root_endpoint():
    """
    Tests if the root endpoint ('/') is working correctly.
    """
    # Make a GET request to the root URL
    response = client.get("/")
    # Check if the HTTP status code is 200 (OK)
    assert response.status_code == 200
    # Check if the response JSON matches the expected welcome message
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
    
    # Check if the HTTP status code is 200 (OK)
    assert response.status_code == 200
    
    # Check if the response is a JSON object that contains the 'prediction' key
    response_data = response.json()
    assert "prediction" in response_data

