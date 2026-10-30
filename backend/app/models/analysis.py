from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float, Text, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import Base

class MediaAnalysis(Base):
    __tablename__ = "media_analysis"

    id = Column(Integer, primary_key=True, autoincrement=True)
    evidence_id = Column(Integer, ForeignKey("evidence.id", ondelete="CASCADE"), nullable=False)
    analysis_type = Column(String, nullable=False, default="PREPROCESSING_FACE_DETECTION")
    status = Column(String, nullable=False, default="QUEUED") # QUEUED | PROCESSING | COMPLETED | FAILED
    integrity_verified = Column(Boolean, nullable=False, default=False)
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)

    # Relationships
    faces = relationship("FaceDetection", back_populates="analysis", cascade="all, delete-orphan")
    evidence = relationship("Evidence")

class FaceDetection(Base):
    __tablename__ = "face_detection"

    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(Integer, ForeignKey("media_analysis.id", ondelete="CASCADE"), nullable=False)
    frame_index = Column(Integer, nullable=True)
    timestamp_seconds = Column(Float, nullable=True)
    face_index = Column(Integer, nullable=False)
    x = Column(Integer, nullable=False)
    y = Column(Integer, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    crop_path = Column(String, nullable=True)

    # Relationships
    analysis = relationship("MediaAnalysis", back_populates="faces")

class DeepfakeAnalysis(Base):
    __tablename__ = "deepfake_analysis"

    id = Column(Integer, primary_key=True, autoincrement=True)
    evidence_id = Column(Integer, ForeignKey("evidence.id", ondelete="CASCADE"), nullable=False)
    preprocessing_analysis_id = Column(Integer, ForeignKey("media_analysis.id", ondelete="CASCADE"), nullable=False)
    status = Column(String, nullable=False, default="QUEUED") # QUEUED | PROCESSING | COMPLETED | FAILED
    model_name = Column(String, nullable=False) # e.g. MesoInception-4
    model_version = Column(String, nullable=False) # e.g. NOT_CONFIGURED
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    overall_label = Column(String, nullable=False) # REAL | FAKE | UNCERTAIN | NO_FACE_DETECTED | MODEL_NOT_CONFIGURED | MODEL_ERROR
    overall_confidence = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)

    # Relationships
    predictions = relationship("DeepfakePrediction", back_populates="analysis", cascade="all, delete-orphan")
    evidence = relationship("Evidence")
    preprocessing_analysis = relationship("MediaAnalysis")

class DeepfakePrediction(Base):
    __tablename__ = "deepfake_prediction"

    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(Integer, ForeignKey("deepfake_analysis.id", ondelete="CASCADE"), nullable=False)
    face_detection_id = Column(Integer, ForeignKey("face_detection.id", ondelete="CASCADE"), nullable=False)
    label = Column(String, nullable=False) # e.g. MODEL_NOT_CONFIGURED
    confidence = Column(Float, nullable=True)
    model_status = Column(String, nullable=False) # e.g. MODEL_NOT_CONFIGURED
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    analysis = relationship("DeepfakeAnalysis", back_populates="predictions")
    face_detection = relationship("FaceDetection")

class ElaAnalysis(Base):
    __tablename__ = "ela_analysis"

    id = Column(Integer, primary_key=True, autoincrement=True)
    evidence_id = Column(Integer, ForeignKey("evidence.id", ondelete="CASCADE"), nullable=False)
    status = Column(String, nullable=False, default="QUEUED") # QUEUED | PROCESSING | COMPLETED | FAILED
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    visualization_path = Column(String, nullable=True)
    mean_error = Column(Float, nullable=True)
    max_error = Column(Float, nullable=True)
    error_variance = Column(Float, nullable=True)
    jpeg_quality = Column(Integer, nullable=False, default=95)
    error_message = Column(Text, nullable=True)

    evidence = relationship("Evidence")

class FftAnalysis(Base):
    __tablename__ = "fft_analysis"

    id = Column(Integer, primary_key=True, autoincrement=True)
    evidence_id = Column(Integer, ForeignKey("evidence.id", ondelete="CASCADE"), nullable=False)
    status = Column(String, nullable=False, default="QUEUED") # QUEUED | PROCESSING | COMPLETED | FAILED
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    visualization_path = Column(String, nullable=True)
    average_magnitude = Column(Float, nullable=True)
    high_frequency_ratio = Column(Float, nullable=True)
    spectral_entropy = Column(Float, nullable=True)
    high_freq_radius_percent = Column(Float, nullable=False, default=15.0)
    error_message = Column(Text, nullable=True)

    evidence = relationship("Evidence")
