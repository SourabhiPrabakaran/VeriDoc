import re
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List

from app.core.config import settings
from app.core.database import get_db
from app.models.report import ForensicReport
from app.schemas.report import ForensicReportResponse, ForensicReportListResponse
from app.services.report_service import ReportService
from app.services.hashing_service import HashingService

router = APIRouter()

@router.post("/evidence/{evidence_id}/report", response_model=ForensicReportResponse, status_code=201, tags=["Reports"])
def generate_report(evidence_id: int, db: Session = Depends(get_db)):
    """
    Trigger judicial-grade forensic PDF report generation for specific digital evidence.
    """
    return ReportService.generate_report(db, evidence_id)

@router.get("/evidence/{evidence_id}/reports", response_model=ForensicReportListResponse, tags=["Reports"])
def list_reports(evidence_id: int, db: Session = Depends(get_db)):
    """
    Retrieve all forensic report metadata records matching specific evidence.
    """
    items = db.query(ForensicReport).filter(ForensicReport.evidence_id == evidence_id).order_by(ForensicReport.generated_at.desc()).all()
    return {"items": items}

@router.get("/reports/{report_uuid}/download", tags=["Reports"])
def download_report_pdf(report_uuid: str, db: Session = Depends(get_db)):
    """
    Secure controlled route to stream forensic report PDF files.
    Validates report existence, prevents path traversal, and verifies PDF file integrity.
    """
    # Validate UUID format to prevent injection/traversal inputs
    if not re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", report_uuid, re.IGNORECASE):
        raise HTTPException(status_code=400, detail="Invalid report UUID format.")

    db_report = db.query(ForensicReport).filter(ForensicReport.report_uuid == report_uuid).first()
    if not db_report:
        raise HTTPException(status_code=404, detail="Forensic report record not found.")

    if db_report.status != "COMPLETED":
        raise HTTPException(status_code=400, detail=f"Forensic report is not ready. Status: {db_report.status}")

    # Build and verify path to prevent path traversal
    file_path = Path(db_report.file_path).resolve()
    reports_dir = settings.REPORTS_DIR.resolve()

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Report PDF file missing from storage vault.")

    # Enforce directory boundaries
    if reports_dir not in file_path.parents:
        raise HTTPException(status_code=403, detail="Access denied: invalid file path path boundaries.")

    # Cryptographic integrity verification of PDF file before streaming
    try:
        current_hash = HashingService.calculate_sha256(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate report hash: {str(e)}")

    if current_hash != db_report.file_hash:
        raise HTTPException(
            status_code=409,
            detail="Report file integrity mismatch. Stored file hash does not match registered metadata."
        )

    # Return PDF file stream
    return FileResponse(
        str(file_path),
        media_type="application/pdf",
        filename=f"veridoc_report_{db_report.evidence_id}_{report_uuid[:8]}.pdf"
    )
