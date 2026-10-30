import re
import json
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List

from app.core.config import settings
from app.core.database import get_db
from app.models.evidence import Evidence
from app.models.analysis import MediaAnalysis, FaceDetection, DeepfakeAnalysis, DeepfakePrediction, ElaAnalysis, FftAnalysis
from app.schemas.analysis import MediaAnalysisResponse, PreprocessTriggerResponse
from app.schemas.deepfake import DeepfakeAnalysisResponse, DeepfakePredictionResponse
from app.schemas.signals import ElaAnalysisResponse, FftAnalysisResponse
from app.services.preprocessing_service import PreprocessingService
from app.services.deepfake_detection_service import DeepfakeDetectionService
from app.services.ela_service import ElaService
from app.services.fft_service import FftService
from app.services.gradcam_service import GradCAMService

router = APIRouter()

# --- PREPROCESSING & FACE DETECTION ENDPOINTS (PHASE 3) ---

@router.post("/evidence/{evidence_id}/preprocess", response_model=PreprocessTriggerResponse, tags=["Analysis"])
def start_preprocessing(evidence_id: int, db: Session = Depends(get_db)):
    """
    Trigger media stream verification, verification check, and face localization pipeline.
    Calculates SHA-256 and compares it against initial ingest records.
    """
    analysis = PreprocessingService.run_preprocessing(db, evidence_id)
    return {
        "analysis_id": analysis.id,
        "evidence_id": analysis.evidence_id,
        "status": analysis.status,
        "integrity_verified": analysis.integrity_verified
    }

@router.get("/evidence/{evidence_id}/preprocessing", response_model=MediaAnalysisResponse, tags=["Analysis"])
def get_preprocessing_details(evidence_id: int, db: Session = Depends(get_db)):
    """
    Retrieve the latest preprocessing run status and face bounding boxes index.
    """
    analysis = db.query(MediaAnalysis).filter(MediaAnalysis.evidence_id == evidence_id).order_by(MediaAnalysis.started_at.desc()).first()
    if not analysis:
        raise HTTPException(status_code=404, detail=f"No preprocessing logs found for evidence {evidence_id}.")

    faces = db.query(FaceDetection).filter(FaceDetection.analysis_id == analysis.id).all()

    video_meta = None
    returned_error_message = analysis.error_message
    
    if analysis.status == "COMPLETED" and analysis.error_message:
        try:
            meta_dict = json.loads(analysis.error_message)
            if "frame_count" in meta_dict:
                video_meta = meta_dict
                returned_error_message = None
        except Exception:
            pass

    return {
        "id": analysis.id,
        "evidence_id": analysis.evidence_id,
        "analysis_type": analysis.analysis_type,
        "status": analysis.status,
        "integrity_verified": analysis.integrity_verified,
        "started_at": analysis.started_at,
        "completed_at": analysis.completed_at,
        "error_message": returned_error_message,
        "faces_detected": len(faces),
        "faces": faces,
        "video_metadata": video_meta
    }

# --- DEEPFAKE DETECTION INTEGRATION ENDPOINTS (PHASE 4) ---

@router.post("/evidence/{evidence_id}/deepfake/analyze", response_model=DeepfakeAnalysisResponse, tags=["Deepfake"])
def start_deepfake_analysis(evidence_id: int, db: Session = Depends(get_db)):
    """
    Triggers deepfake detection on the face crops generated during preprocessing.
    """
    return DeepfakeDetectionService.run_deepfake_analysis(db, evidence_id)

@router.get("/evidence/{evidence_id}/deepfake", response_model=DeepfakeAnalysisResponse, tags=["Deepfake"])
def get_latest_deepfake_analysis(evidence_id: int, db: Session = Depends(get_db)):
    """
    Returns the latest deepfake analysis record for an evidence file.
    """
    evidence = db.query(Evidence).filter(Evidence.id == evidence_id).first()
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence item not found.")

    analysis = db.query(DeepfakeAnalysis).filter(
        DeepfakeAnalysis.evidence_id == evidence_id
    ).order_by(DeepfakeAnalysis.started_at.desc()).first()
    
    if not analysis:
        raise HTTPException(status_code=404, detail="No deepfake analysis logs found for this evidence item.")
        
    return analysis

@router.get("/evidence/{evidence_id}/deepfake/predictions", response_model=List[DeepfakePredictionResponse], tags=["Deepfake"])
def get_deepfake_predictions(evidence_id: int, db: Session = Depends(get_db)):
    """
    Returns individual face prediction details for the latest deepfake analysis run.
    """
    evidence = db.query(Evidence).filter(Evidence.id == evidence_id).first()
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence item not found.")

    analysis = db.query(DeepfakeAnalysis).filter(
        DeepfakeAnalysis.evidence_id == evidence_id
    ).order_by(DeepfakeAnalysis.started_at.desc()).first()
    
    if not analysis:
        raise HTTPException(status_code=404, detail="No deepfake analysis logs found for this evidence item.")

    predictions = db.query(DeepfakePrediction).filter(
        DeepfakePrediction.analysis_id == analysis.id
    ).all()
    
    return predictions

# --- ERROR LEVEL ANALYSIS (ELA) ENDPOINTS (PHASE 5) ---

@router.post("/evidence/{evidence_id}/ela/analyze", response_model=ElaAnalysisResponse, tags=["Forensic Signals"])
def start_ela_analysis(evidence_id: int, db: Session = Depends(get_db)):
    """
    Triggers JPEG-based Error Level Analysis (ELA) on registered image evidence.
    """
    return ElaService.run_ela(db, evidence_id)

@router.get("/evidence/{evidence_id}/ela", response_model=ElaAnalysisResponse, tags=["Forensic Signals"])
def get_latest_ela_analysis(evidence_id: int, db: Session = Depends(get_db)):
    """
    Returns the latest ELA analysis metadata and metrics.
    """
    evidence = db.query(Evidence).filter(Evidence.id == evidence_id).first()
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence item not found.")

    analysis = db.query(ElaAnalysis).filter(
        ElaAnalysis.evidence_id == evidence_id
    ).order_by(ElaAnalysis.started_at.desc()).first()

    if not analysis:
        raise HTTPException(status_code=404, detail="No ELA analysis logs found for this evidence item.")

    return analysis

@router.get("/analysis/ela/{filename}", tags=["Forensic Signals"])
def get_ela_visualization(filename: str):
    """
    Controlled secure route to stream ELA visualization files.
    """
    if not re.match(r"^ela_[0-9]+_[a-f0-9]+\.jpg$", filename):
        raise HTTPException(status_code=400, detail="Invalid ELA filename format.")

    file_path = settings.ELA_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="ELA visualization file not found.")

    return FileResponse(str(file_path), media_type="image/jpeg")

# --- FAST FOURIER TRANSFORM (FFT) ENDPOINTS (PHASE 5) ---

@router.post("/evidence/{evidence_id}/fft/analyze", response_model=FftAnalysisResponse, tags=["Forensic Signals"])
def start_fft_analysis(evidence_id: int, db: Session = Depends(get_db)):
    """
    Triggers 2D Fast Fourier Transform (FFT) analysis on registered image evidence.
    """
    return FftService.run_fft(db, evidence_id)

@router.get("/evidence/{evidence_id}/fft", response_model=FftAnalysisResponse, tags=["Forensic Signals"])
def get_latest_fft_analysis(evidence_id: int, db: Session = Depends(get_db)):
    """
    Returns the latest FFT analysis metadata and metrics.
    """
    evidence = db.query(Evidence).filter(Evidence.id == evidence_id).first()
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence item not found.")

    analysis = db.query(FftAnalysis).filter(
        FftAnalysis.evidence_id == evidence_id
    ).order_by(FftAnalysis.started_at.desc()).first()

    if not analysis:
        raise HTTPException(status_code=404, detail="No FFT analysis logs found for this evidence item.")

    return analysis

@router.get("/analysis/fft/{filename}", tags=["Forensic Signals"])
def get_fft_visualization(filename: str):
    """
    Controlled secure route to stream FFT visualization files.
    """
    if not re.match(r"^fft_[0-9]+_[a-f0-9]+\.png$", filename):
        raise HTTPException(status_code=400, detail="Invalid FFT filename format.")

    file_path = settings.FFT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="FFT visualization file not found.")

    return FileResponse(str(file_path), media_type="image/png")

# --- SECURE RESOURCE SERVING ENDPOINTS (PHASE 3) ---

@router.get("/evidence/{evidence_id}/preview", tags=["Evidence"])
def get_evidence_preview(evidence_id: int, db: Session = Depends(get_db)):
    """
    Controlled endpoint to serve the registered evidence file stream.
    Protects original files by preventing directory indexing or raw static mounts.
    """
    evidence = db.query(Evidence).filter(Evidence.id == evidence_id).first()
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence item not found.")

    file_path = Path(evidence.storage_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Registered file missing on storage vault.")

    return FileResponse(
        str(file_path),
        media_type=evidence.mime_type,
        filename=evidence.original_filename
    )

@router.get("/analysis/crop/{filename}", tags=["Analysis"])
def get_analysis_crop(filename: str):
    """
    Controlled endpoint to serve cropped face regions for UI bounding-box details.
    """
    if not re.match(r"^crop_[0-9]+_(frame_[0-9]+_)?face_[0-9]+_[a-f0-9]+\.(jpg|jpeg|png)$", filename):
        raise HTTPException(status_code=400, detail="Invalid crop filename query.")

    file_path = settings.FACE_CROPS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Request crop slice not found.")

    return FileResponse(str(file_path), media_type="image/jpeg")

# --- GRAD-CAM EXPLAINABILITY ENDPOINTS (PHASE 6) ---

@router.post("/evidence/{evidence_id}/face/{face_id}/gradcam", tags=["Explainability"])
def generate_face_gradcam(evidence_id: int, face_id: int, db: Session = Depends(get_db)):
    """
    Generate a Grad-CAM activation heatmap for a specific detected face crop.
    Shows facial regions influencing classification decisions.
    """
    evidence = db.query(Evidence).filter(Evidence.id == evidence_id).first()
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence item not found.")

    face = db.query(FaceDetection).filter(FaceDetection.id == face_id).first()
    if not face:
        raise HTTPException(status_code=404, detail="Face detection record not found.")

    crop_path = settings.FACE_CROPS_DIR / face.crop_path
    if not crop_path.exists():
        raise HTTPException(status_code=404, detail="Face crop file not found on disk.")

    result = GradCAMService.generate_heatmap(str(crop_path), target_class_is_real=False)
    if result["status"] == "MODEL_NOT_CONFIGURED":
        raise HTTPException(status_code=400, detail=result["error"])
    if result["status"] == "FAILED":
        raise HTTPException(status_code=500, detail=result["error"])

    return {
        "status": "COMPLETED",
        "face_id": face.id,
        "gradcam_filename": result["gradcam_filename"],
        "gradcam_url": f"/api/analysis/gradcam/{result['gradcam_filename']}",
        "target_class": result["target_class"],
        "layer_inspected": result["layer_inspected"]
    }

@router.get("/analysis/gradcam/{filename}", tags=["Explainability"])
def get_gradcam_visualization(filename: str):
    """
    Controlled secure route to stream Grad-CAM heatmap overlay images.
    """
    if not re.match(r"^gradcam_[a-f0-9]+\.jpg$", filename):
        raise HTTPException(status_code=400, detail="Invalid Grad-CAM filename query.")

    file_path = settings.GRADCAM_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Grad-CAM visualization file not found.")

    return FileResponse(str(file_path), media_type="image/jpeg")
