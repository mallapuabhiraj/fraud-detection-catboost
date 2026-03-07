from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "predictions"

    id                = Column(Integer, primary_key=True, index=True)
    cc_num            = Column(String)
    amt               = Column(Float)
    merchant          = Column(String)
    category          = Column(String)
    fraud_probability = Column(Float)
    prediction        = Column(String)
    risk_level        = Column(String)
    created_at        = Column(DateTime, default=datetime.utcnow)

def create_tables():
    Base.metadata.create_all(bind=engine)