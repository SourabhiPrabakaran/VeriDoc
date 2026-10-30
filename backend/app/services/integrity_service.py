from datetime import datetime, timezone
from pathlib import Path
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.evidence import Evidence, AuditLog
from app.services.hashing_service import HashingService


class IntegrityService:
    @staticmethod
    def verify(db: Session, evidence_id: int) -> dict:
        evidence = db.query(Evidence).filter(Evidence.id == evidence_id).first()
        if evidence is None:
            raise HTTPException(status_code=404, detail="Evidence record not found.")
        path = Path(evidence.storage_path)
        current = None
        if not path.is_file():
            status, matches = "MISSING", None
        else:
            try:
                current = HashingService.calculate_sha256(path)
            except OSError:
                raise HTTPException(status_code=500, detail="Unable to read stored evidence for verification.")
            matches = current == evidence.sha256_hash
            status = "VERIFIED" if matches else "MISMATCH"
        checked_at = datetime.now(timezone.utc)
        db.add(AuditLog(evidence_id=evidence_id, action="INTEGRITY_CHECK_" + status,
                        timestamp=checked_at, details=f"Fresh original-file verification: {status}."))
        db.commit()
        return dict(evidence_id=evidence_id, original_sha256=evidence.sha256_hash,
                    current_sha256=current, matches=matches, status=status, checked_at=checked_at)
