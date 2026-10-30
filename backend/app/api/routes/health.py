from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.evidence import Evidence, AuditLog
from app.models.analysis import DeepfakeAnalysis, ElaAnalysis, FftAnalysis

router = APIRouter()

@router.get("/health", tags=["System"])
def health_check():
    """
    Query the active state of the forensic backend application.
    Returns: {"status": "ok", "service": "veridoc-backend"}
    """
    return {
        "status": "ok",
        "service": "veridoc-backend"
    }

@router.get("/dashboard/stats", tags=["System"])
def get_dashboard_stats(db: Session = Depends(get_db)):
    """
    Returns live database-backed statistics for the dashboard overview.
    """
    total_evidence = db.query(Evidence).count()
    completed_df = db.query(DeepfakeAnalysis).filter(DeepfakeAnalysis.status == "COMPLETED").count()
    completed_ela = db.query(ElaAnalysis).filter(ElaAnalysis.status == "COMPLETED").count()
    completed_fft = db.query(FftAnalysis).filter(FftAnalysis.status == "COMPLETED").count()
    total_analyses = completed_df + completed_ela + completed_fft

    high_risk_deepfake = db.query(DeepfakeAnalysis).filter(DeepfakeAnalysis.overall_label == "FAKE").count()
    high_risk_ela = db.query(ElaAnalysis).filter(ElaAnalysis.status == "COMPLETED", ElaAnalysis.error_variance > 100.0).count()
    high_risk_fft = db.query(FftAnalysis).filter(FftAnalysis.status == "COMPLETED", FftAnalysis.high_frequency_ratio > 0.5).count()
    high_risk_total = high_risk_deepfake + high_risk_ela + high_risk_fft

    integrity_warnings = db.query(AuditLog).filter(
        AuditLog.action.in_(["INTEGRITY_VERIFICATION_FAILED", "SHA256_MISMATCH"])
    ).count()

    return {
        "total_evidence": total_evidence,
        "analyses_completed": total_analyses,
        "high_risk_findings": high_risk_total,
        "integrity_warnings": integrity_warnings
    }
