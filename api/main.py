from fastapi import FastAPI
from api.schemas import Transaction, PredictionResponse, BatchRequest, BatchPredictionResponse
from api.database import SessionLocal, PredictionLog, create_tables
from api.model_services import FraudModelService
from sqlalchemy import text
import logging
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fraud Detection API",
    version="1.0",
    description="Real-time credit card fraud detection using CatBoost"
)

model_service = FraudModelService()

try:
    create_tables()
    logger.info("Database tables created successfully")
except Exception as e:
    logger.warning(f"Database connection failed: {e}")

@app.get('/health')
def health():
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        db_status = "connected"
    except Exception:
        db_status = "disconnected"

    return {
        'status': 'running',
        'model': 'loaded',
        'version': '1.0',
        'database': db_status,
        'endpoints': ['/predict', '/predict-batch', '/predictions/history']
    }

@app.post('/predict', response_model = PredictionResponse)
def predict(tnx: Transaction):
    data = tnx.model_dump()
    probability, prediction, risk = model_service.predict(data)

    log = PredictionLog(
        cc_num=str(tnx.cc_num),
        amt=tnx.amt,
        merchant=tnx.merchant,
        category=tnx.category,
        fraud_probability=probability,
        prediction=prediction,
        risk_level=risk
    )
    try:
        db = SessionLocal()
        db.add(log)
        db.commit()
    except Exception as e:
        logger.error(f"Database logging failed: {e}")
        db.rollback()
    finally:
        db.close()

    return {
        "fraud_probability": probability,
        "prediction": prediction,
        "risk_level": risk
    }

@app.post('/predict-batch', response_model = BatchPredictionResponse)
def predict_batch(batch: BatchRequest):
    results = []

    db = SessionLocal()
    for tx in batch.transactions:
        data = tx.model_dump()
        probability, prediction, risk = model_service.predict(data)
        log = PredictionLog(
            cc_num=str(tx.cc_num),
            amt=tx.amt,
            merchant=tx.merchant,
            category=tx.category,
            fraud_probability=probability,
            prediction=prediction,
            risk_level=risk
        )
        db.add(log)
        results.append({
            "fraud_probability": probability,
            "prediction": prediction,
            "risk_level": risk
        })
    try:
        db.commit()
    except Exception as e:
        logger.error(f"Database logging failed: {e}")
        db.rollback()
    finally:
        db.close()

    return {"predictions": results}

@app.get('/predictions/history')
def get_history(limit: int = 50):
    db = SessionLocal()
    logs = db.query(PredictionLog).order_by(
        PredictionLog.created_at.desc()
    ).limit(limit).all()
    db.close()
    return logs
