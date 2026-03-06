from fastapi import FastAPI
from api.schemas import Transaction, PredictionResponse, BatchRequest
from api.model_service import FraudModelService

app = FastAPI(
    title="Fraud Detection API",
    version="1.0",
    description="Real-time credit card fraud detection using CatBoost"
)

model_service = FraudModelService()

@app.get('/health')
def health():
    return {'status' : 'running'}

@app.post('/predict', response_model = PredictionResponse)
def predict(tnx: Transaction):
    data = tnx.model_dump()
    probability, prediction, risk = model_service.predict(data)
    return {
        "fraud_probability": probability,
        "prediction": prediction,
        "risk_level": risk
    }

@app.post('/predict-batch', response_model = PredictionResponse)
def predict_batch(batch: BatchRequest)
    results = []

    for tx in batch.transactions:

        data = tx.model_dump()

        probability, prediction, risk = model_service.predict(data)

        results.append({
            "fraud_probability": probability,
            "prediction": prediction,
            "risk_level": risk
        })

    return {"predictions": results}
