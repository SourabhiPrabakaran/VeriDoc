from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.core.database import get_db
from app.models.evidence import Evidence, AuditLog
from app.schemas.evidence import EvidenceResponse, EvidenceListResponse, AuditLogResponse
from app.services.evidence_service import EvidenceService

router = APIRouter()

@router.post("/evidence/upload", response_model=EvidenceResponse, status_code=201, tags=["Evidence"])
def upload_evidence(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Ingest a new digital evidence file (Image/Video), perform validation checks,
    compute its cryptographic SHA-256 hash, write it to storage, and log audit events.
    """
    return EvidenceService.register_evidence(db, file)

@router.get("/evidence", response_model=EvidenceListResponse, tags=["Evidence"])
def list_evidence(db: Session = Depends(get_db)):
    """
    Retrieve all registered evidence metadata logs from the local SQLite index.
    """
    items = db.query(Evidence).order_by(Evidence.created_at.desc()).all()
    return {"items": items}

@router.get("/evidence/{evidence_id}", response_model=EvidenceResponse, tags=["Evidence"])
def get_evidence_details(evidence_id: int, db: Session = Depends(get_db)):
    """
    Retrieve metadata logs for a specific registered evidence file.
    """
    evidence = db.query(Evidence).filter(Evidence.id == evidence_id).first()
    if not evidence:
        raise HTTPException(status_code=404, detail=f"Evidence file with ID {evidence_id} not found.")
    return evidence

@router.get("/evidence/{evidence_id}/audit", response_model=List[AuditLogResponse], tags=["Evidence"])
def get_evidence_audit_logs(evidence_id: int, db: Session = Depends(get_db)):
    """
    Fetch the complete, timestamped custody log matching an evidence record.
    """
    # Verify that the evidence record exists first
    evidence = db.query(Evidence).filter(Evidence.id == evidence_id).first()
    if not evidence:
        raise HTTPException(status_code=404, detail=f"Evidence file with ID {evidence_id} not found.")

    logs = db.query(AuditLog).filter(AuditLog.evidence_id == evidence_id).order_by(AuditLog.timestamp.asc()).all()
    return logs
