import os
import shutil
import uuid
from datetime import datetime
from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.evidence import Evidence, AuditLog
from app.services.hashing_service import HashingService
from app.utils.file_utils import validate_and_classify_file, generate_safe_stored_filename

class EvidenceService:
    @staticmethod
    def register_evidence(db: Session, upload_file: UploadFile) -> Evidence:
        original_name = upload_file.filename
        
        # 1 & 2: Validate file properties and extensions
        ext, category, mime = validate_and_classify_file(original_name, upload_file.content_type)
        
        # Ensure temporary directory exists
        settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        
        # 3: Save file stream to temporary storage
        temp_name = f"temp_{uuid.uuid4().hex}{ext}"
        temp_path = settings.TEMP_DIR / temp_name
        
        file_size = 0
        try:
            with open(temp_path, "wb") as buffer:
                # Read upload in 64KB blocks to optimize memory usage
                while chunk := upload_file.file.read(65536):
                    buffer.write(chunk)
                    file_size += len(chunk)
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise HTTPException(status_code=500, detail=f"Temp file write failed: {str(e)}")

        # Reject empty files
        if file_size == 0:
            if temp_path.exists():
                temp_path.unlink()
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # 4: Calculate SHA-256 hash of temporary file
        try:
            sha256_hash = HashingService.calculate_sha256(temp_path)
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise HTTPException(status_code=500, detail=f"Hash calculation failed: {str(e)}")

        # 5: Check duplicate hash in DB
        existing_evidence = db.query(Evidence).filter(Evidence.sha256_hash == sha256_hash).first()
        if existing_evidence:
            # Delete temporary file and raise 409 Conflict
            temp_path.unlink()
            raise HTTPException(
                status_code=409,
                detail=f"Evidence with this SHA-256 hash already exists. ID: {existing_evidence.id}"
            )

        # Ensure permanent storage directory exists
        settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)

        # 6: Move/rename from temp into permanent evidence storage
        stored_name = generate_safe_stored_filename(original_name)
        storage_path = settings.STORAGE_DIR / stored_name
        
        try:
            shutil.move(str(temp_path), str(storage_path))
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise HTTPException(status_code=500, detail=f"Failed to move file to storage: {str(e)}")

        # 7 & 8: Persist metadata in database & log audit trail
        db_evidence = Evidence(
            original_filename=original_name,
            stored_filename=stored_name,
            media_category=category,
            mime_type=mime,
            file_size=file_size,
            file_extension=ext,
            sha256_hash=sha256_hash,
            storage_path=str(storage_path),
            status="REGISTERED",
            created_at=datetime.utcnow()
        )

        try:
            db.add(db_evidence)
            db.flush() # Flushes session to fetch the database autoincremented ID

            # Create evidence audit log record referencing the real database ID
            db_audit = AuditLog(
                evidence_id=db_evidence.id,
                action="EVIDENCE_REGISTERED",
                timestamp=datetime.utcnow(),
                details="Evidence uploaded and SHA-256 integrity hash generated."
            )
            db.add(db_audit)
            db.commit()
            db.refresh(db_evidence)
            return db_evidence
        except Exception as e:
            db.rollback()
            # Security safeguard: delete the stored file so we don't leave orphaned files on DB failure
            if storage_path.exists():
                storage_path.unlink()
            raise HTTPException(
                status_code=500,
                detail=f"Evidence database registration failed: {str(e)}"
            )
