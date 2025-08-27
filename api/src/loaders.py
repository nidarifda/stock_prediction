# api/src/loaders.py
from pathlib import Path
import pickle

def load_all_models(model_dir: Path):
    model_dir = Path(model_dir)
    models = {"lgbm": {"A": {"reg": None}}, "y_scaler": None}

    reg_path = model_dir / "nvda_A_reg_lgb.pkl"
    if not reg_path.exists():
        raise FileNotFoundError(f"Missing model file: {reg_path}")

    with open(reg_path, "rb") as f:
        models["lgbm"]["A"]["reg"] = pickle.load(f)

    scaler_path = model_dir / "y_scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            models["y_scaler"] = pickle.load(f)

    print("✅ Loaded: lgbm/A/reg", " + y_scaler" if models["y_scaler"] is not None else "")
    return models
