from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class ElaAnalysisResponse(BaseModel):
    id: int
    evidence_id: int
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    visualization_path: Optional[str] = None
    mean_error: Optional[float] = None
    max_error: Optional[float] = None
    error_variance: Optional[float] = None
    jpeg_quality: int
    error_message: Optional[str] = None

    class Config:
        from_attributes = True

class FftAnalysisResponse(BaseModel):
    id: int
    evidence_id: int
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    visualization_path: Optional[str] = None
    average_magnitude: Optional[float] = None
    high_frequency_ratio: Optional[float] = None
    spectral_entropy: Optional[float] = None
    high_freq_radius_percent: float
    error_message: Optional[str] = None

    class Config:
        from_attributes = True
