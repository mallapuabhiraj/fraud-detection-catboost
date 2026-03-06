from catboost import CatBoostClassifier, Pool
import pandas as pd
import joblib
from api.config import FRAUD_THRESHOLD, CAT_COLS

class FraudModelService:
    def __init__(self):
        self.model = CatBoostClassifier()
        self.model.load_model('models/cb_model.cbm')
        self.preprocess_pipe = joblib.load('models/preprocess_pipe.joblib')
    def predict(self, data: dict):
        df = pd.DataFrame([data])
        df_preprocessed = self.preprocess_pipe.transform(df)
        pool = Pool(df_preprocessed, cat_features=CAT_COLS)
        proba = self.model.predict_proba(df_preprocessed)
        pred = proba > FRAUD_THRESHOLD
