# api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import yaml

# Загрузка конфига
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

FEATURES = config["features"]

# Загрузка модели
try:
    model = joblib.load("models/best_xgboost_model.pkl")
except FileNotFoundError:
    raise RuntimeError("Модель не найдена. Проверьте путь к файлу")

app = FastAPI()


# Структура входных данных
class PredictionRequest(BaseModel):
    risky_tx_count: int
    avg_risk_factor: float
    total_collateral_eth: float
    risk_factor_above_threshold_daily_count: int
    unique_lending_protocol_count: int
    incoming_tx_count: int
    borrow_amount_sum_eth: float
    repay_amount_sum_eth: float
    wallet_age: float
    total_gas_paid_eth: float


@app.get("/")
def read_root():
    return {"status": "работает", "model_loaded": True}


@app.post("/predict")
def predict(request: PredictionRequest):
    """Предсказание вероятности дефолта"""
    input_data = pd.DataFrame([request.dict()])

    try:
        probability = model.predict_proba(input_data[FEATURES])[:, 1][0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка модели: {e}")

    return {
        "defaulter": bool(probability > 0.5),
        "probability": round(float(probability), 4)
    }
