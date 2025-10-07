from fastapi.testclient import TestClient
from main import app

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
    # UPDATED to use the correct feature names and sample data that match main.py
    sample_payload = {
        "Category": "Food",
        "City": "New York",
        "Calories": 550.0,
        "Price": 25.99,
        "Offer_Type": "Discount",
        "Is_Premium": True
    }

    response = client.post("/predict", json=sample_payload)

    if response.status_code != 200:
        print(f"API Error Response: {response.json()}")

    assert response.status_code == 200

