import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from api.main import app

client = TestClient(app)

SAMPLE_TRANSACTION = {
    "trans_date_trans_time": "2024-01-15 14:32:00",
    "cc_num": 4532015112830366,
    "merchant": "fraud_Rippin, Kub and Mann",
    "category": "shopping_net",
    "amt": 149.62,
    "gender": "M",
    "city": "Anytown",
    "state": "TX",
    "zip": 75001,
    "lat": 33.9659,
    "long": -80.9355,
    "city_pop": 149180,
    "job": "Software Engineer",
    "dob": "1990-03-15",
    "merch_lat": 33.986391,
    "merch_long": -81.200714
}


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def mock_model_service():
    with patch('api.main.model_service') as mock:
        mock.predict.return_value = (0.005, "legitimate", "Low Risk")
        yield mock

@pytest.fixture
def mock_db():
    with patch('api.main.SessionLocal') as mock:
        mock.return_value = MagicMock()
        yield mock


# ─── Health Tests ────────────────────────────────────────────────────────────

def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200

def test_health_returns_running():
    response = client.get("/health")
    assert response.json()["status"] == "running"

def test_health_has_database_status():
    response = client.get("/health")
    assert "database" in response.json()


# ─── Predict Tests ───────────────────────────────────────────────────────────

def test_predict_returns_200(mock_db, mock_model_service):
    response = client.post("/predict", json=SAMPLE_TRANSACTION)
    assert response.status_code == 200

def test_predict_response_has_required_fields(mock_db, mock_model_service):
    response = client.post("/predict", json=SAMPLE_TRANSACTION)
    data = response.json()
    assert "fraud_probability" in data
    assert "prediction" in data
    assert "risk_level" in data

def test_predict_legitimate_transaction(mock_db, mock_model_service):
    response = client.post("/predict", json=SAMPLE_TRANSACTION)
    data = response.json()
    assert data["prediction"] == "legitimate"
    assert data["risk_level"] == "Low Risk"

def test_predict_fraud_transaction(mock_db, mock_model_service):
    with patch('api.main.model_service') as mock:
        mock.predict.return_value = (0.95, "fraud", "High Risk")
        response = client.post("/predict", json=SAMPLE_TRANSACTION)
        data = response.json()
        assert data["prediction"] == "fraud"
        assert data["risk_level"] == "High Risk"

def test_predict_missing_field_returns_422():
    incomplete = SAMPLE_TRANSACTION.copy()
    del incomplete["amt"]
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422

def test_predict_invalid_data_returns_422():
    invalid = SAMPLE_TRANSACTION.copy()
    invalid["amt"] = "not_a_number"
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422

def test_predict_probability_between_0_and_1(mock_db, mock_model_service):
    response = client.post("/predict", json=SAMPLE_TRANSACTION)
    probability = response.json()["fraud_probability"]
    assert 0 <= probability <= 1

def test_predict_prediction_is_valid_label(mock_db, mock_model_service):
    response = client.post("/predict", json=SAMPLE_TRANSACTION)
    prediction = response.json()["prediction"]
    assert prediction in ["fraud", "legitimate"]

def test_predict_risk_is_valid_label(mock_db, mock_model_service):
    response = client.post("/predict", json=SAMPLE_TRANSACTION)
    risk = response.json()["risk_level"]
    assert risk in ["Low Risk", "Medium Risk", "High Risk"]


# ─── Batch Tests ─────────────────────────────────────────────────────────────

def test_batch_returns_200(mock_db, mock_model_service):
    response = client.post("/predict-batch", json={"transactions": [SAMPLE_TRANSACTION]})
    assert response.status_code == 200

def test_batch_returns_list_of_predictions(mock_db, mock_model_service):
    response = client.post("/predict-batch", json={"transactions": [SAMPLE_TRANSACTION, SAMPLE_TRANSACTION]})
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2

def test_batch_empty_transactions_returns_200(mock_db, mock_model_service):
    response = client.post("/predict-batch", json={"transactions": []})
    assert response.status_code == 200

def test_batch_single_transaction(mock_db, mock_model_service):
    response = client.post("/predict-batch", json={"transactions": [SAMPLE_TRANSACTION]})
    data = response.json()
    assert len(data["predictions"]) == 1

def test_batch_each_prediction_has_required_fields(mock_db, mock_model_service):
    response = client.post("/predict-batch", json={"transactions": [SAMPLE_TRANSACTION]})
    prediction = response.json()["predictions"][0]
    assert "fraud_probability" in prediction
    assert "prediction" in prediction
    assert "risk_level" in prediction