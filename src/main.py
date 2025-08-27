import os
from pathlib import Path
from typing import Dict, Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .schemas import (
    HealthResponse,
    RegressionRequest, RegressionResponse,
    ClassificationRequest, ClassificationResponse,
)
from .loaders import load_all_models
from .infer import to_np, last_step, inverse_y_if_possible, prepare_seq_for_keras

# ----- Env -----
load_dotenv()
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
THRESH = float(os.getenv("THRESHOLD_UP", "0.5"))
DEFAULT_TAG = os.getenv("DEFAULT_TAG", "B")
# We only support LightGBM pickles here
DEFAULT_FRAMEWORK = "lgbm"

# Optional comma-separated CORS origins
def _parse_cors(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [o.strip().rstrip("/") for o in s.split(",") if o.strip()]

CORS_ALLOW_ORIGINS = _parse_cors(os.getenv("CORS_ALLOW_ORIGINS"))

# ----- App -----
app = FastAPI(title="NVDA Forecast API", version="1.0.0")

# CORS
allow_origins = CORS_ALLOW_ORIGINS or [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    # Add your deployed frontend origin(s) here if needed:
    # "https://<username>.github.io",
    # "https://<username>.github.io/<repo>",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins if allow_origins else ["*"],  # permissive for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models at import time (Render/uvicorn will import module once)
MODELS = load_all_models(MODEL_DIR)
Y_SCALER = MODELS.get("y_scaler", None)

def _get_model(tag: str, kind: str):
    """
    kind: "reg" or "cls"
    """
    mdl = MODELS.get("lgbm", {}).get(tag, {}).get(kind)
    if mdl is None:
        raise HTTPException(status_code=404, detail=f"Model not found for tag={tag}, kind={kind}")
    return mdl

# ----- Routes -----
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()

@app.post("/predict/regression", response_model=RegressionResponse)
def predict_regression(req: RegressionRequest) -> RegressionResponse:
    # Only LightGBM path (use last row)
    X = to_np(req.X)              # [T, F]
    X_last = last_step(X)         # (1, F)
    reg = _get_model(req.tag, "reg")
    y_scaled = float(reg.predict(X_last)[0])
    y_pred, scaled_flag = inverse_y_if_possible(y_scaled, Y_SCALER)
    return RegressionResponse(
        tag=req.tag, framework="lgbm",
        y_pred=y_pred, scaled=scaled_flag,
        note=None if not scaled_flag else "Returned in scaled space; y_scaler.pkl not found"
    )

@app.post("/predict/classification", response_model=ClassificationResponse)
def predict_classification(req: ClassificationRequest) -> ClassificationResponse:
    # Optional: only works if cls pickle is present
    X = to_np(req.X)
    X_last = last_step(X)
    cls = _get_model(req.tag, "cls")
    # sklearn predict_proba -> [:, 1] for positive class
    p_up = float(cls.predict_proba(X_last)[:, 1][0])
    label = int(p_up >= THRESH)
    return ClassificationResponse(
        tag=req.tag, framework="lgbm",
        p_up=p_up, label=label, threshold=THRESH
    )
