from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings

from app.core.database import engine, Base

# Import all models to register Base metadata before table creation
from app.models.evidence import Evidence, AuditLog
from app.models.analysis import (
    MediaAnalysis,
    FaceDetection,
    DeepfakeAnalysis,
    DeepfakePrediction,
    ElaAnalysis,
    FftAnalysis
)
from app.models.report import ForensicReport

# Import API routes
from app.api.routes import health, evidence, analysis, reports

# Automatically initialize SQLite database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Veridoc Forensics API",
    description="Evidence Ingest, Cryptographic Hashing, Face Preprocessing, and Forensic Reporting Server.",
    version="1.3.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routers
app.include_router(health.router, prefix="/api")
app.include_router(evidence.router, prefix="/api")
app.include_router(analysis.router, prefix="/api")
app.include_router(reports.router, prefix="/api")