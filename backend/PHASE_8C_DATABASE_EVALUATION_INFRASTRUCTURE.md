# Phase 8C — Evaluation Database Schema & Dataset Loading Infrastructure

This document details the SQLite schema configuration, relational integrity definitions, migration protocols, and boundary isolation checks designed to support scientific validations in VeriDoc.

---

## 1. Database Schema Specifications

Two new SQLAlchemy model structures have been defined inside [`app/models/evaluation.py`](file:///c:/Users/soura/OneDrive/Pictures/VeriDoc/backend/app/models/evaluation.py):

### A. Table: `evaluation_run`
Tracks individual benchmark execution runs and holds aggregated validation metrics:
*   `id` (Integer, Primary Key, Auto-increment)
*   `dataset_name` (String): Audited benchmark directory name.
*   `dataset_split` (String): Evaluation partition (default `test`).
*   `started_at` (DateTime): UTC timestamp when evaluation started.
*   `completed_at` (DateTime): UTC timestamp when evaluation completed.
*   `model_name` (String): MesoInception-4.
*   `model_version` (String): 1.0.
*   `checkpoint_sha256` (String): SHA-256 hash signature of the checkpoint file.
*   `total_samples` (Integer): Total audited test count.
*   `real_samples` (Integer): Ground Truth REAL samples count.
*   `fake_samples` (Integer): Ground Truth FAKE samples count.
*   `accuracy` / `precision` / `recall` / `f1_score` / `specificity` (Float): Summary statistics.
*   `roc_auc` (Float, Nullable): ROC-AUC curve area.
*   `status` (String): `PROCESSING` | `COMPLETED` | `FAILED`.

### B. Table: `evaluation_prediction`
Stores prediction outcomes for every individual crop evaluated:
*   `id` (Integer, Primary Key, Auto-increment)
*   `evaluation_run_id` (Integer, Foreign Key to `evaluation_run.id` with cascade deletion)
*   `dataset_name` (String)
*   `source_video_id` (String, Nullable): Extracted parent video ID.
*   `frame_number` (Integer, Nullable): Extracted frame index offset.
*   `ground_truth` (String): REAL | FAKE.
*   `predicted_label` (String): REAL | FAKE | MODEL_ERROR.
*   `probability` (Float): Raw sigmoid prediction.
*   `confidence` (Float): Normalized confidence.
*   `inference_time_ms` (Float): Latency.
*   `crop_path` (String): File system path.

---

## 2. Table Relationships & Cascades
*   **One-to-Many:** An `EvaluationRun` has many `EvaluationPrediction` children.
*   **Cascade Deletion:** The relationship is defined with `cascade="all, delete-orphan"`. Deleting an `EvaluationRun` row automatically deletes all associated `EvaluationPrediction` rows in SQLite, preventing database clutter on test deletions.

---

## 3. Database Migration Approach
*   **Initialization Convention:** The tables are declared under `app/models/` and registered by importing them in `app/main.py`.
*   **Schema Update:** On startup, FastAPI calls `Base.metadata.create_all(bind=engine)`. In SQLite, this automatically creates any missing tables (i.e. `evaluation_run` and `evaluation_prediction`) without altering or destroying the schema, rows, or indexes of any existing production tables (e.g. `evidence`, `audit_log`, `media_analysis`).

---

## 4. Provenance & Metadata Handling
*   The scanning pipeline parses filename structures to identify original source video IDs and frame indices automatically (e.g. `123_456.jpg` maps to `video_id="123"` and `frame=456`).
*   If filenames do not match standard formats, the fields are written as `None` (represented as `NULL` in the database) to prevent fabricating source indices.

---

## 5. Production / Evaluation Isolation
*   **Functional Boundaries:** The benchmark suite is completely separated from the forensic endpoints. 
*   **Audit Log Protection:** The evaluation service runs directly on database queries and does **not** write to the `AuditLog` table. This preserves the absolute integrity of the chain-of-custody logs.

---

## 6. Test Verification Results
All automated test suites passed successfully:
*   **`test_phase7_model.py`**: **PASSED**
*   **`test_phase4.py`**: **PASSED**
*   **`test_phase6_reports.py`**: **PASSED**
*   **`test_phase8_infrastructure.py`** (covering table creation, FK cascade deletes, duplicate runs, failures, rollbacks, and folder audits): **PASSED**
    ```text
    Ran 8 tests in 0.280s
    OK
    ```

---

## 7. Remaining Blocker
*   **Benchmark Data Placement:** The actual benchmark evaluation is stopped pending the manual placement of the test dataset under `backend/data/evaluation_dataset/`.
