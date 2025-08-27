# api/src/loaders.py
from pathlib import Path
import pickle

def load_all_models(model_dir: Path):
    model_dir = Path(model_dir)
    needed = ["nvda_A_reg_lgb.pkl"]  # add others if any
    models = {}
    for fname in needed:
        p = model_dir / fname
        if not p.exists():
            listing = "\n".join(str(x.name) for x in model_dir.glob("*"))
            raise FileNotFoundError(
                f"Missing model file: {p}\nDir contains:\n{listing}"
            )
        with p.open("rb") as f:
            models[fname] = pickle.load(f)
    return models
