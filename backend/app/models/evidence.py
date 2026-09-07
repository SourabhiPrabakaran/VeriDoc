from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Index
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import Base

class Evidence(Base):
    __tablename__ = "evidence"

    id = Column(Integer, primary_key=True, autoincrement=True)
    original_filename = Column(String, nullable=False)
    stored_filename = Column(String, nullable=False)
    media_category = Column(String, nullable=False) # IMAGE | VIDEO
    mime_type = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    file_extension = Column(String, nullable=False) # e.g. .jpg, .mp4
    sha256_hash = Column(String(64), nullable=False, unique=True) # Exactly 64 hex chars, unique constraint
    storage_path = Column(String, nullable=False)
    status = Column(String, nullable=False, default="REGISTERED")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    audit_logs = relationship("AuditLog", back_populates="evidence", cascade="all, delete-orphan")

    # Add index for hash lookup
    __table_args__ = (
        Index("ix_evidence_sha256_hash", "sha256_hash", unique=True),
    )

class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    evidence_id = Column(Integer, ForeignKey("evidence.id", ondelete="CASCADE"), nullable=False)
    action = Column(String, nullable=False) # e.g. EVIDENCE_REGISTERED
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    details = Column(Text, nullable=True)

    # Relationships
    evidence = relationship("Evidence", back_populates="audit_logs")
