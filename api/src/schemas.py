from typing import List, Literal, Optional
from pydantic import BaseModel, Field

# ---- Health ----
class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"

# ---- Regression ----
class RegressionRequest(BaseModel):
    tag: Literal["A", "B", "AFF"] = Field(default="B")
    framework: Literal["lgbm"] = Field(default="lgbm")
    # 2D array [T, F] or [1, F]
    X: List[List[float]]

class RegressionResponse(BaseModel):
    tag: str
    framework: str
    y_pred: float
    # True means value is still in scaled space because y_scaler was not found
    scaled: bool = False
    note: Optional[str] = None

# ---- Classification (optional, only if cls models are present) ----
class ClassificationRequest(BaseModel):
    tag: Literal["A", "B", "AFF"] = Field(default="B")
    framework: Literal["lgbm"] = Field(default="lgbm")
    X: List[List[float]]

class ClassificationResponse(BaseModel):
    tag: str
    framework: str
    p_up: float
    label: int
    threshold: float
