from fastapi import APIRouter

router = APIRouter()

@router.get("/health", tags=["System"])
def health_check():
    """
    Query the active state of the forensic backend application.
    Returns: {"status": "ok", "service": "veridoc-backend"}
    """
    return {
        "status": "ok",
        "service": "veridoc-backend"
    }
