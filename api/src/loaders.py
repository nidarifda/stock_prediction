# api/src/loaders.py
from pathlib import Path
import pickle
import traceback

def _list_dir(p: Path) -> str:
    try:
        if not p.exists():
            return f"<dir {p} does not exist>"
        entries = sorted([f.name for f in p.iterdir()])
        return "\n".join(entries) if entries else "<empty>"
    except Exception as e:
        return f"<error listing dir {p}: {e}>"

def load_all_models(model_dir: Path):
    """
    Load all required artifacts and return a nested dict:
      {
        "lgbm": {"A": {"reg": <model or None>}},
        "y_scaler": <scaler or None>
      }
    Raises FileNotFoundError with a helpful directory listing when critical files are missing.
    """
    model_dir = Path(model_dir)
    models = {"lgbm": {"A": {"reg": None}}, "y_scaler": None}

    # ---- Required: regression model for lgbm/A ----
    reg_path = model_dir / "nvda_A_reg_lgb.pkl"
    if not reg_path.exists():
        listing = _list_dir(model_dir)
        raise FileNotFoundError(
            f"Missing model file: {reg_path}\n"
            f"Resolved model_dir: {model_dir}\n"
            f"Directory content:\n{listing}"
        )

    try:
        with reg_path.open("rb") as f:
            models["lgbm"]["A"]["reg"] = pickle.load(f)
    except Exception as e:
        tb = traceback.format_exc()
        raise RuntimeError(f"Failed to load regression model from {reg_path}: {e}\n{tb}") from e

    # ---- Optional: y_scaler ----
    scaler_path = model_dir / "y_scaler.pkl"
    if scaler_path.exists():
        try:
            with scaler_path.open("rb") as f:
                models["y_scaler"] = pickle.load(f)
        except Exception as e:
            # Non-fatal: prediction can still proceed in scaled space
            # Raise if you want to enforce scaler presence.
            raise RuntimeError(f"Failed to load y_scaler from {scaler_path}: {e}") from e

    print(
        "✅ Loaded models:",
        f"reg={reg_path.name}",
        "+ y_scaler" if models["y_scaler"] is not None else "(no y_scaler found)",
        f"| dir={model_dir}",
    )
    return models
