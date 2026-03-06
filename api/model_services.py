from catboost import CatBoostClassifier
Class FraudModelService:
    model = CatBoostClassifier()
    model.load('models/cb_model')
    preprocess_pipe = joblib.load('models/preprocess_pipe.joblib')
