import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Veridoc Forensics API"
    API_V1_STR: str = "/api"
    
    # Path calculations using pathlib
    BACKEND_DIR: Path = Path(__file__).resolve().parent.parent.parent
    
    @property
    def DATA_DIR(self) -> Path:
        dir_path = self.BACKEND_DIR / "data"
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    @property
    def DATABASE_URL(self) -> str:
        db_path = self.DATA_DIR / "veridoc.db"
        return f"sqlite:///{db_path.as_posix()}"

    @property
    def STORAGE_DIR(self) -> Path:
        dir_path = self.BACKEND_DIR / "storage" / "evidence"
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    @property
    def TEMP_DIR(self) -> Path:
        dir_path = self.BACKEND_DIR / "storage" / "temp"
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    @property
    def ANALYSIS_DIR(self) -> Path:
        dir_path = self.BACKEND_DIR / "storage" / "analysis"
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    @property
    def FACE_CROPS_DIR(self) -> Path:
        dir_path = self.ANALYSIS_DIR / "face_crops"
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    @property
    def ELA_DIR(self) -> Path:
        dir_path = self.ANALYSIS_DIR / "ela"
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    @property
    def FFT_DIR(self) -> Path:
        dir_path = self.ANALYSIS_DIR / "fft"
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    @property
    def REPORTS_DIR(self) -> Path:
        # Directory for Forensic Reports
        dir_path = self.BACKEND_DIR / "storage" / "reports"
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    # Configurable frame sampling interval for video preprocessing
    VIDEO_SAMPLE_INTERVAL_SECONDS: float = 1.0

    # Configuration for MesoInception-4 Deepfake model
    DEEPFAKE_MODEL_PATH: Optional[str] = str(Path(__file__).resolve().parent.parent.parent / "data" / "mesoinception_4.pth")
    DEEPFAKE_MODEL_NAME: str = "MesoInception-4"
    DEEPFAKE_MODEL_VERSION: str = "1.0"
    DEEPFAKE_DEVICE: str = "cpu"

    # CORS configuration
    ALLOWED_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ]

    class Config:
        case_sensitive = True

settings = Settings()
