import pickle
from pathlib import Path
from typing import Any, Dict
import joblib

# File name patterns we look for inside MODEL_DIR
REG_NAMES = {
    "A":   "nvda_A_reg_lgb.pkl",
    "B":   "nvda_B_reg_lgb.pkl",
    "AFF": "nvda_AFF_reg_lgb.pkl",
}
CLS_NAMES = {
    "A":   "nvda_A_cls_lgb.pkl",
    "B":   "nvda_B_cls_lgb.pkl",
    "AFF": "nvda_AFF_cls_lgb.pkl",
}
SCALER_NAME = "y_scaler.pkl"

def _safe_load(path: Path):
    """Load with joblib first, then pickle fallback."""
    try:
        return joblib.load(path)
    except Exception:
        with path.open("rb") as f:
            return pickle.load(f)

def load_all_models(model_dir: Path) -> Dict[str, Any]:
    """
    Load LightGBM sklearn-wrapped models and optional y_scaler from MODEL_DIR.
    Structure:
      {
        "lgbm": {
          "A":   {"reg": ..., "cls": optional},
          "B":   {"reg": ..., "cls": optional},
          "AFF": {"reg": ..., "cls": optional},
        },
        "y_scaler": optional
      }
    """
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"MODEL_DIR does not exist: {model_dir.resolve()}")

    out: Dict[str, Any] = {"lgbm": {"A": {}, "B": {}, "AFF": {}}}

    # Load regressors (required)
    for tag, fname in REG_NAMES.items():
        p = model_dir / fname
        if not p.exists():
            raise FileNotFoundError(f"Missing regressor for {tag}: {p.name}")
        out["lgbm"][tag]["reg"] = _safe_load(p)

    # Load classifiers if present (optional)
    for tag, fname in CLS_NAMES.items():
        p = model_dir / fname
        if p.exists():
            out["lgbm"][tag]["cls"] = _safe_load(p)

    # Optional scaler
    p_scaler = model_dir / SCALER_NAME
    if p_scaler.exists():
        out["y_scaler"] = _safe_load(p_scaler)

    print("✅ Models loaded.")
    return out
