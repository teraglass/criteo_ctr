from fastapi import APIRouter
import os
import lightgbm as lgb
import pandas as pd
import pickle

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok"}

@router.post("/predict")
def predict(data: dict):
    model_path = os.getenv("MODEL_PATH", "ml/model.lgb")
    model = lgb.Booster(model_file=model_path)
    df = pd.DataFrame([data])
    pred = model.predict(df)
    return {"prediction": pred.tolist()}
