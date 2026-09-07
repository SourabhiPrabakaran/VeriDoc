from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List

class ForensicReportResponse(BaseModel):
    id: int
    evidence_id: int
    report_uuid: str
    status: str
    generated_at: datetime
    file_path: str
    file_hash: Optional[str] = None
    error_message: Optional[str] = None

    class Config:
        from_attributes = True

class ForensicReportListResponse(BaseModel):
    items: List[ForensicReportResponse]
