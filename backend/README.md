# Veridoc Forensics - Backend Service

This is the Python + FastAPI backend service for **Veridoc Forensics**, a secure digital evidence authentication and deepfake analysis workstation.

---

## 1. Installation & Environment Setup

The backend runs in a project-local virtual environment.

### Virtual Environment Initialization
From the `backend/` directory, initialize a Python virtual environment:
```bash
python -m venv venv
```

### Activating the Virtual Environment
- **Windows (PowerShell)**:
  ```powershell
  venv\Scripts\Activate.ps1
  ```
- **macOS / Linux**:
  ```bash
  source venv/bin/activate
  ```

### Dependency Installation
Install the required packages in the active virtual environment using `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 2. Dependencies Configuration (`requirements.txt`)
The `requirements.txt` file specifies the exact libraries used:
- `fastapi`: Core ASGI web framework.
- `uvicorn[standard]`: Production-grade ASGI server.
- `python-multipart`: Stream parser for multipart file uploads.
- `sqlalchemy`: Object Relational Mapper (ORM) for database interactions.
- `pydantic`: Data validation schemas.
- `pydantic-settings`: Configurations manager using environment variables.

---

## 3. Database Design
Veridoc uses a local **SQLite** database managed via SQLAlchemy. 

On startup, database tables are automatically initialized at:
`backend/data/veridoc.db`

### Table: `evidence`
Keeps record of all registered media assets.
- `id` (Integer, Primary Key, Auto-increment)
- `original_filename` (String): Raw client name (e.g. `cctv_capture.png`).
- `stored_filename` (String): Safe UUID-generated name on the filesystem.
- `media_category` (String): Classified as `IMAGE` or `VIDEO`.
- `mime_type` (String): Standardized MIME type (e.g., `image/png`, `video/mp4`).
- `file_size` (Integer): Ingest size in bytes.
- `file_extension` (String): Standard suffix (e.g., `.png`).
- `sha256_hash` (String, length 64, UNIQUE index): Cryptographic content signature.
- `storage_path` (String): Absolute file system location.
- `status` (String): Initial state is `"REGISTERED"`.
- `created_at` (DateTime): UTC timestamp of ingestion.

### Table: `audit_log`
Provides an immutable chain of custody audit log.
- `id` (Integer, Primary Key, Auto-increment)
- `evidence_id` (Integer, Foreign Key to `evidence.id` with cascade deletion)
- `action` (String): Type of custody event (e.g., `EVIDENCE_REGISTERED`).
- `timestamp` (DateTime): UTC timestamp of the log event.
- `details` (Text, Nullable): Human-readable event description.

---

## 4. File Storage Strategy
- Raw files are stored under: `backend/storage/evidence/`
- DB stores metadata records referencing these files.
- Safe generated filenames (using UUIDs) are used to prevent path traversal and shell injection vulnerabilities.

---

## 5. Ingestion Pipeline & Safe Duplicate Handling

Whenever a file is uploaded, the server processes it through a strict secure sequence:
```text
UPLOAD FILE
    ↓
VALIDATION (extension check & category classification)
    ↓
WRITE TO TEMPORARY STORAGE (backend/storage/temp/)
    ↓
CALCULATE SHA-256 (from actual file bytes)
    ↓
CHECK DUPLICATE HASH (look up sha256_hash in database)
    ├─► [FOUND] ──► DELETE TEMP FILE & Raise HTTP 409 Conflict
    └─► [NEW]   ──► MOVE/RENAME into permanent storage (backend/storage/evidence/)
                         ↓
                    PERSIST IN SQLITE & WRITE AUDIT ENTRY
                         ↓
                    COMMIT TRANSACTIONS & RETURN RESPONSES
```
*If database transactions fail during registration, transaction rollbacks automatically trigger filesystem cleanups to delete the permanent file, preventing orphaned files.*

---

## 6. SHA-256 Cryptographic Ingest Hashing
Implemented in `app/services/hashing_service.py`, the SHA-256 is computed by reading the file stream in 64KB blocks. This guarantees:
- The hash represents the actual binary bytes of the file.
- The system never hashes filenames or timestamps, nor does it generate random hashes.

---

## 7. API Routing & Endpoints

- **Health Status Check**: `GET /api/health`
  - Returns `{"status": "ok", "service": "veridoc-backend"}`.
- **Upload Evidence**: `POST /api/evidence/upload`
  - Consumes multipart data, calculates hash, writes metadata, and returns the response.
- **List Evidence**: `GET /api/evidence`
  - Returns list: `{"items": [...]}`.
- **Evidence Detail**: `GET /api/evidence/{evidence_id}`
  - Returns metadata fields of the specified ID.
- **Audit Logs**: `GET /api/evidence/{evidence_id}/audit`
  - Returns the list of historical audit trail events for a file.

---

## 8. Startup Instruction
Run the server locally with Uvicorn:
```bash
uvicorn app.main:app --reload --port 8000
```
- Interactive Swagger documentation is available at: `http://127.0.0.1:8000/docs`
- Redoc documentation is available at: `http://127.0.0.1:8000/redoc`

---

## 9. Frontend Connection & CORS Settings
CORS is explicitly configured in `main.py` to allow the following development origins:
- `http://localhost:5173`
- `http://127.0.0.1:5173`
All network fetch methods are isolated in the frontend service `src/services/api.js`.
