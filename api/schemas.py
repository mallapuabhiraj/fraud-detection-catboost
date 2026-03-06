from pydantic import BaseModel
from typing import List

class Transaction(BaseModel):

    trans_date_trans_time: str
    cc_num: int
    merchant: str
    category: str
    amt: float
    gender: str
    city: str
    state: str
    zip: int
    lat: float
    long: float
    city_pop: int
    job: str
    dob: str
    merch_lat: float
    merch_long: float


class PredictionResponse(BaseModel):
    fraud_probability: float
    prediction: str
    risk_level: str


class BatchRequest(BaseModel):
    transactions: List[Transaction]
