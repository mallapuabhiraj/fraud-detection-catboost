from catboost import CatBoostClassifier, Pool
import pandas as pd
from api.config import FRAUD_THRESHOLD, CAT_COLS
from src.preprocess import Preprocess
from sklearn.pipeline import Pipeline
from joblib import load
import logging
logger = logging.getLogger(__name__)

class FraudModelService:
    def __init__(self):
        self.model = CatBoostClassifier()
        self.model.load_model('models/cb_model.cbm')
        logger.info('Model loaded successfully')
        self.preprocess_pipe = load(
            'models/preprocess_pipe.joblib'
        )
        logger.info('Pipeline loaded successfully')

    def predict(self, data: dict):
        logger.info('Received predict request')
        
        try:
            df = pd.DataFrame([data])
            df_proc = self.preprocess_pipe.transform(df)
            pool = Pool(df_proc, cat_features=CAT_COLS)
            probability = float(round(self.model.predict_proba(pool)[0][1], 3))
            prediction = "fraud" if probability >= FRAUD_THRESHOLD else "legitimate"
            risk = "Critical Risk" if probability >= 0.99 else "High Risk" if probability >= 0.9 else "Elevated Risk" if probability >= FRAUD_THRESHOLD else "Low Risk"
            
            return probability, prediction, risk
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
