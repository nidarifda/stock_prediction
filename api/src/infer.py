from typing import Iterable, Tuple
import numpy as np

def to_np(X: Iterable[Iterable[float]]) -> np.ndarray:
    """
    Ensure X is a 2D numpy array (T, F). Accepts lists/tuples.
    """
    arr = np.asarray(list(list(row) for row in X), dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("X must be a 2D array shaped [T, F].")
    return arr

def last_step(X: np.ndarray) -> np.ndarray:
    """
    Return the last timestep as shape (1, F) for tree models (LightGBM).
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D [T, F].")
    return X[-1:, :]  # shape (1, F)

def inverse_y_if_possible(y_scaled: float, scaler) -> Tuple[float, bool]:
    """
    If a y_scaler is available, inverse-transform a single value.
    Returns (value, scaled_flag). scaled_flag=False if inverse done,
    True if still in scaled space (no scaler available).
    """
    if scaler is None:
        return float(y_scaled), True
    val = np.asarray([[y_scaled]], dtype=np.float32)
    inv = scaler.inverse_transform(val).ravel()[0]
    return float(inv), False

# Present just for API compatibility; does nothing special
def prepare_seq_for_keras(X: np.ndarray) -> np.ndarray:
    """
    For neural nets we'd reshape to (B, T, F). We keep this noop adapter
    so main.py can import it without pulling TensorFlow.
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D [T, F].")
    return X[None, ...]  # (1, T, F)
