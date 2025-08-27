# api/src/main.py
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .schemas import HealthResponse, RegressionRequest, RegressionResponse
from .loaders import load_all_models
from .infer import to_np, last_step, inverse_y_if_possible

load_dotenv()
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
DEFAULT_TAG = "A"
DEFAULT_FRAMEWORK = "lgbm"

app = FastAPI(title="NVDA Forecast API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS = load_all_models(MODEL_DIR)
Y_SCALER = MODELS.get("y_scaler", None)

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse()

def _reg_model():
    mdl = MODELS["lgbm"]["A"]["reg"]
    if mdl is None:
        raise HTTPException(status_code=404, detail="Regression model not loaded (lgbm/A).")
    return mdl

@app.post("/predict/regression", response_model=RegressionResponse)
def predict_regression(req: RegressionRequest):
    # Ignore incoming tag/framework — we serve only lgbm/A
    x = to_np(req.X)                       # [T,F] or [1,F]
    x_last = last_step(x)                  # [1,F] for LightGBM
    reg = _reg_model()
    y_scaled = float(reg.predict(x_last)[0])
    y_pred, scaled_flag = inverse_y_if_possible(y_scaled, Y_SCALER)
    return RegressionResponse(
        tag=DEFAULT_TAG,
        framework=DEFAULT_FRAMEWORK,
        y_pred=y_pred,
        scaled=scaled_flag,
        note=None if not scaled_flag else "Returned in scaled space; y_scaler.pkl missing",
    )
