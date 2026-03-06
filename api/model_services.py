from catboost import CatBoostClassifier, Pool
import pandas as pd
from api.config import FRAUD_THRESHOLD, CAT_COLS
from src.preprocess import Preprocess

class FraudModelService:
    def __init__(self):
        self.model = CatBoostClassifier()
        self.model.load_model('models/cb_model.cbm')
        print('Model loaded Successfully')
        self.preprocess_pipe = Pipeline([('preprocess', Preprocess())])
        print('Pipeline loaded Successfully')

    def predict(self, data: dict):
        print('Recieved Predict Request')
        # Converting the data to a Pandas DataFrame
        df = pd.DataFrame([data])
        
        # Performing Data Preprocessing using preprocess_pipe
        df_proc = self.preprocess_pipe.transform(df)

        # Creating pool as CatBoost require categorical features
        pool = Pool(df_proc, cat_features=CAT_COLS)

        # Taking Probability and rounding it
        probability = self.model.predict_proba(pool)[0][1]
        probability = float(round(probability, 3))

        # Making our final prediction using FRAUD_THRESHOLD
        prediction = "fraud" if probability >= FRAUD_THRESHOLD else "legitimate"

        # Defining risk using probability
        if probability >= 0.9:
            risk = "High Risk"
        elif probability >= FRAUD_THRESHOLD:
            risk = "Medium Risk"
        else:
            risk = "Low Risk"
        
        return probability, prediction, risk
