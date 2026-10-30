from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
from app.core.database import Base

class ForensicReport(Base):
    __tablename__ = "forensic_report"

    id = Column(Integer, primary_key=True, autoincrement=True)
    evidence_id = Column(Integer, ForeignKey("evidence.id", ondelete="CASCADE"), nullable=False)
    report_uuid = Column(String, unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    status = Column(String, nullable=False, default="GENERATING") # GENERATING | COMPLETED | FAILED
    generated_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    file_path = Column(String, nullable=False)
    file_hash = Column(String, nullable=True) # SHA-256 hash of PDF bytes
    error_message = Column(Text, nullable=True)

    # Relationships
    evidence = relationship("Evidence")
