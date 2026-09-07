from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List

class DeepfakePredictionResponse(BaseModel):
    id: int
    analysis_id: int
    face_detection_id: int
    label: str
    confidence: Optional[float] = None
    model_status: str
    created_at: datetime

    class Config:
        from_attributes = True

class DeepfakeAnalysisResponse(BaseModel):
    id: int
    evidence_id: int
    preprocessing_analysis_id: int
    status: str
    model_name: str
    model_version: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    overall_label: str
    overall_confidence: Optional[float] = None
    error_message: Optional[str] = None
    predictions: List[DeepfakePredictionResponse] = []

    class Config:
        from_attributes = True
