from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Literal

class EvidenceBase(BaseModel):
    original_filename: str = Field(..., example="security_feed.mp4")
    media_category: str = Field(..., example="VIDEO") # IMAGE | VIDEO
    mime_type: str = Field(..., example="video/mp4")
    file_size: int = Field(..., example=245678)
    file_extension: str = Field(..., example=".mp4")
    sha256_hash: str = Field(..., example="a3f80000000000000000000000000000000000000000000000000000000091bc")
    status: str = Field(default="REGISTERED")

class EvidenceCreate(EvidenceBase):
    stored_filename: str
    storage_path: str

class EvidenceResponse(EvidenceBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

class AuditLogResponse(BaseModel):
    id: int
    evidence_id: int
    action: str = Field(..., example="EVIDENCE_REGISTERED")
    timestamp: datetime
    details: Optional[str] = Field(None, example="Evidence uploaded and SHA-256 integrity hash generated.")

    class Config:
        from_attributes = True

class EvidenceListResponse(BaseModel):
    items: List[EvidenceResponse]


class IntegrityResponse(BaseModel):
    evidence_id: int
    original_sha256: str
    current_sha256: Optional[str]
    matches: Optional[bool]
    status: Literal["VERIFIED", "MISMATCH", "MISSING"]
    checked_at: datetime
