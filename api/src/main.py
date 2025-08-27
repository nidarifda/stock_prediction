# api/src/main.py
import os
import logging
from functools import lru_cache
from pathlib import Path
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .schemas import HealthResponse, RegressionRequest, RegressionResponse
from .loaders import load_all_models
from .infer import to_np, last_step, inverse_y_if_possible

# --------------------
# Setup & Settings
# --------------------
load_dotenv()

# Basic logging (feel free to adapt/route to JSON logger if you prefer)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("nvda-api")

HERE = Path(__file__).resolve().parent            # .../api/src
REPO_ROOT = HERE.parents[1]                       # .../api
# If your repo structure is repo_root/api/src/main.py, then:
#   - HERE = api/src
#   - REPO_ROOT = api
# If your repo structure is repo_root/<api>/src, adjust if needed.

DEFAULT_TAG = os.getenv("DEFAULT_TAG", "A")
DEFAULT_FRAMEWORK = os.getenv("DEFAULT_FRAMEWORK", "lgbm")

# CORS origins (override with ALLOW_ORIGINS env var: comma-separated)
DEFAULT_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
ENV_ORIGINS = os.getenv("ALLOW_ORIGINS")
ALLOW_ORIGINS = (
    [o.strip() for o in ENV_ORIGINS.split(",") if o.strip()]
    if ENV_ORIGINS
    else DEFAULT_ORIGINS + ["*"]  # keep "*" for convenience; remove for production hardening
)

app = FastAPI(title="NVDA Forecast API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _resolve_model_dir() -> Path:
    """
    Resolve the model directory with the following precedence:
    1) MODEL_DIR env var
    2) ./models next to this file (api/src/models)
    3) ../models (api/models)
    4) repo-root level 'models' (if your repo uses a different layout)
    """
    env_dir = os.getenv("MODEL_DIR")
    if env_dir:
        p = Path(env_dir)
        # If relative, interpret relative to repo root for convenience
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        return p

    # Heuristic fallbacks
    candidates = [
        HERE / "models",
        REPO_ROOT / "models",
        REPO_ROOT.parent / "models",  # repo_root if api/ is a subfolder
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    # Final default (even if it doesn't exist yet — loaders will error with listing)
    return (HERE / "models").resolve()

# --------------------
# Lazy Model Loading
# --------------------
@lru_cache(maxsize=1)
def get_models_and_scaler():
    model_dir = _resolve_model_dir()
    log.info("Loading models from: %s", model_dir)
    models = load_all_models(model_dir)
    y_scaler = models.get("y_scaler", None)
    loaded = {
        "model_dir": str(model_dir),
        "has_y_scaler": y_scaler is not None,
        "keys": list(models.keys()),
    }
    log.info("Models loaded summary: %s", loaded)
    return models, y_scaler, model_dir

def _reg_model():
    models, _, _ = get_models_and_scaler()
    try:
        mdl = models["lgbm"]["A"]["reg"]
    except Exception:
        mdl = None
    if mdl is None:
        raise HTTPException(status_code=404, detail="Regression model not loaded (lgbm/A).")
    return mdl

# --------------------
# Routes
# --------------------
@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness probe: server process is up."""
    return HealthResponse()

@app.get("/ready")
def ready():
    """
    Readiness probe: confirms model availability and where we loaded them from.
    Returns 200 if models are loaded; 503 with details if not.
    """
    try:
        models, y_scaler, model_dir = get_models_and_scaler()
        ok = models and models.get("lgbm", {}).get("A", {}).get("reg") is not None
        if not ok:
            raise RuntimeError("lgbm/A/reg not present in loaded models")
        return {
            "ok": True,
            "model_dir": str(model_dir),
            "has_y_scaler": y_scaler is not None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        log.exception("Readiness check failed")
        # 503 helps orchestrators distinguish from liveness
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/predict/regression", response_model=RegressionResponse)
def predict_regression(req: RegressionRequest):
    """
    Single-step regression predict:
    - Accepts either [T,F] or [1,F]
    - Uses last timestep features for tree model
    - Attempts inverse scaling if y_scaler is present
    """
    try:
        reg = _reg_model()
        _, y_scaler, _ = get_models_and_scaler()

        x = to_np(req.X)            # [T,F] or [1,F]
        x_last = last_step(x)       # [1,F] for LightGBM
        y_scaled = float(reg.predict(x_last)[0])

        y_pred, scaled_flag = inverse_y_if_possible(y_scaled, y_scaler)
        return RegressionResponse(
            tag=DEFAULT_TAG,
            framework=DEFAULT_FRAMEWORK,
            y_pred=y_pred,
            scaled=scaled_flag,
            note=None if not scaled_flag else "Returned in scaled space; y_scaler.pkl missing",
        )
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

@app.post("/reload")
def reload_models():
    """
    Clear the lazy-load cache and reload models. Useful after updating files on disk.
    """
    try:
        get_models_and_scaler.cache_clear()
        models, y_scaler, model_dir = get_models_and_scaler()
        ok = models and models.get("lgbm", {}).get("A", {}).get("reg") is not None
        return {
            "ok": bool(ok),
            "model_dir": str(model_dir),
            "has_y_scaler": y_scaler is not None,
            "reloaded": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        log.exception("Reload failed")
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")

@app.get("/")
def root():
    return {
        "name": "NVDA Forecast API",
        "version": app.version,
        "endpoints": ["/health", "/ready", "/predict/regression", "/reload"],
    }
