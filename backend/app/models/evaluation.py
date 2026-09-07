from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import Base

class EvaluationRun(Base):
    __tablename__ = "evaluation_run"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_name = Column(String, nullable=False)
    dataset_split = Column(String, nullable=False)
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    model_name = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    checkpoint_sha256 = Column(String, nullable=False)
    total_samples = Column(Integer, nullable=False, default=0)
    real_samples = Column(Integer, nullable=False, default=0)
    fake_samples = Column(Integer, nullable=False, default=0)
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    specificity = Column(Float, nullable=True)
    roc_auc = Column(Float, nullable=True)
    status = Column(String, nullable=False, default="PROCESSING") # PROCESSING | COMPLETED | FAILED

    # Relationships
    predictions = relationship("EvaluationPrediction", back_populates="run", cascade="all, delete-orphan")


class EvaluationPrediction(Base):
    __tablename__ = "evaluation_prediction"

    id = Column(Integer, primary_key=True, autoincrement=True)
    evaluation_run_id = Column(Integer, ForeignKey("evaluation_run.id", ondelete="CASCADE"), nullable=False)
    dataset_name = Column(String, nullable=False)
    source_video_id = Column(String, nullable=True)
    frame_number = Column(Integer, nullable=True)
    ground_truth = Column(String, nullable=False) # REAL | FAKE
    predicted_label = Column(String, nullable=False) # REAL | FAKE | MODEL_ERROR
    probability = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    inference_time_ms = Column(Float, nullable=False)
    crop_path = Column(String, nullable=False)

    # Relationships
    run = relationship("EvaluationRun", back_populates="predictions")
