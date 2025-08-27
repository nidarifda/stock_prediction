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

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("nvda-api")

HERE = Path(__file__).resolve().parent            # .../api/src
REPO_ROOT = HERE.parents[1]                       # repo root for the 'api' package

DEFAULT_TAG = os.getenv("DEFAULT_TAG", "A")
DEFAULT_FRAMEWORK = os.getenv("DEFAULT_FRAMEWORK", "lgbm")

# --------------------
# CORS
# --------------------
DEFAULT_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
ENV_ORIGINS = os.getenv("ALLOW_ORIGINS")
ALLOW_ORIGINS = (
    [o.strip() for o in ENV_ORIGINS.split(",") if o.strip()]
    if ENV_ORIGINS
    else DEFAULT_ORIGINS + ["*"]  # include "*" by default for convenience
)
# Starlette/FastAPI: wildcard + credentials=True is not allowed
ALLOW_CREDENTIALS = "*" not in ALLOW_ORIGINS

app = FastAPI(title="NVDA Forecast API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------
# Model path resolution
# --------------------
def _resolve_model_dir() -> Path:
    """
    Resolve the model directory with the following precedence:
    1) MODEL_DIR env var (absolute or relative to REPO_ROOT)
    2) ./models next to this file (api/src/models)
    3) ../models (api/models)
    4) repo-root sibling 'models'
    """
    env_dir = os.getenv("MODEL_DIR")
    if env_dir:
        p = Path(env_dir)
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        return p

    candidates = [
        HERE / "models",
        REPO_ROOT / "models",
        REPO_ROOT.parent / "models",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()

    # Return a reasonable default; loaders will raise with a helpful listing
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
    log.info(
        "Models loaded summary: dir=%s, has_y_scaler=%s, keys=%s",
        model_dir, bool(y_scaler is not None), list(models.keys())
    )
    return models, y_scaler, model_dir

def _reg_model():
    models, _, _ = get_models_and_scaler()
    mdl = models.get("lgbm", {}).get("A", {}).get("reg")
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
        ok = models.get("lgbm", {}).get("A", {}).get("reg") is not None
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
        ok = models.get("lgbm", {}).get("A", {}).get("reg") is not None
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
