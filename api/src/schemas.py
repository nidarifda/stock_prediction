# api/src/schemas.py
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

Tag = Literal["A"]
Framework = Literal["lgbm"]

class HealthResponse(BaseModel):
    status: str = "ok"

class RegressionRequest(BaseModel):
    X: List[List[float]]
    # kept but defaulted; they’re ignored server-side anyway
    tag: Tag = Field("A")
    framework: Framework = Field("lgbm")

class RegressionResponse(BaseModel):
    tag: Tag
    framework: Framework
    y_pred: float
    scaled: bool = False
    note: Optional[str] = None
