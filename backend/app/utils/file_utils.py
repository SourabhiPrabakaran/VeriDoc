import uuid
from pathlib import Path
from fastapi import HTTPException

# Explicit allowed extensions
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}
ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS.union(ALLOWED_VIDEO_EXTENSIONS)

# Internal safe MIME-type mapping to normalize browser values
MIME_TYPE_MAPPING = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo"
}

def validate_and_classify_file(filename: str, content_type: str = None) -> tuple[str, str, str]:
    """
    Validates the filename extension and classifies the media category (IMAGE or VIDEO).
    Returns (normalized_extension, media_category, mime_type).
    Raises HTTP 400 Bad Request on failure.
    """
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is empty or missing.")

    # Standardize suffix extraction
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported extension '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # Classify category based on file suffix
    if ext in ALLOWED_IMAGE_EXTENSIONS:
        category = "IMAGE"
    else:
        category = "VIDEO"

    # Map file suffix to standard MIME type
    mime = MIME_TYPE_MAPPING.get(ext, content_type or "application/octet-stream")

    return ext, category, mime

def generate_safe_stored_filename(original_filename: str) -> str:
    """
    Generates a unique and safe filename using a UUID.
    This prevents path traversal and shell injection vulnerabilities.
    """
    ext = Path(original_filename).suffix.lower()
    unique_id = uuid.uuid4().hex
    return f"{unique_id}{ext}"
