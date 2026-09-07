from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List

class FaceDetectionResponse(BaseModel):
    id: int
    analysis_id: int
    frame_index: Optional[int] = None
    timestamp_seconds: Optional[float] = None
    face_index: int
    x: int
    y: int
    width: int
    height: int
    crop_path: Optional[str] = None

    class Config:
        from_attributes = True

class VideoMetadata(BaseModel):
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    sampled_frames: int

class MediaAnalysisResponse(BaseModel):
    id: int
    evidence_id: int
    analysis_type: str = Field(default="PREPROCESSING_FACE_DETECTION")
    status: str = Field(..., example="COMPLETED") # QUEUED | PROCESSING | COMPLETED | FAILED
    integrity_verified: bool
    started_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    faces_detected: int = 0
    faces: List[FaceDetectionResponse] = []
    video_metadata: Optional[VideoMetadata] = None

    class Config:
        from_attributes = True

class PreprocessTriggerResponse(BaseModel):
    analysis_id: int
    evidence_id: int
    status: str
    integrity_verified: bool
