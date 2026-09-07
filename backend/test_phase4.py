import urllib.request
import urllib.error
import json
import uuid
import sys
import sqlite3
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000/api"

def run_phase4_tests():
    print("=== STARTING VERIDOC FORENSICS PHASE 4 TEST PIPELINE ===")

    # Setup directories
    backend_dir = Path(__file__).resolve().parent
    db_path = backend_dir / "data" / "veridoc.db"
    
    # 1. Test missing evidence (expect 404)
    print("Test 1: Querying missing evidence for deepfake analysis...")
    try:
        req = urllib.request.Request(
            f"{BASE_URL}/evidence/99999/deepfake/analyze",
            method="POST"
        )
        urllib.request.urlopen(req)
        print("[FAIL] Missing evidence did not trigger 404!")
        sys.exit(1)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print("[SUCCESS] Missing evidence correctly returned 404.")
        else:
            print(f"[FAIL] Missing evidence returned unexpected status {e.code}")
            sys.exit(1)

    # 2. Upload a new evidence image
    print("Preparing test image...")
    from PIL import Image, ImageDraw
    img_path = backend_dir / "test_image_p4.jpg"
    img = Image.new('RGB', (320, 240), color = (80, 110, 140))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), str(uuid.uuid4()), fill=(255, 255, 255))
    img.save(img_path)

    # Upload helper
    def upload_file(file_path, filename, mime_type):
        boundary = "----WebKitFormBoundaryPhase4Test"
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: {mime_type}\r\n\r\n"
        ).encode('utf-8') + file_bytes + f"\r\n--{boundary}--\r\n".encode('utf-8')

        req = urllib.request.Request(
            f"{BASE_URL}/evidence/upload",
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}"
            }
        )
        return req

    evidence_id = None
    try:
        req = upload_file(img_path, "test_image_p4.jpg", "image/jpeg")
        with urllib.request.urlopen(req) as res:
            data = json.loads(res.read().decode('utf-8'))
            evidence_id = data["id"]
            print(f"[SUCCESS] Test image registered with ID {evidence_id}")
    except Exception as e:
        print(f"[FAIL] Upload test image failed: {e}")
        if img_path.exists():
            img_path.unlink()
        sys.exit(1)

    # 3. Test evidence with incomplete preprocessing (expect 400 or 409)
    print("Test 2: Triggering deepfake analysis before preprocessing...")
    try:
        req = urllib.request.Request(
            f"{BASE_URL}/evidence/{evidence_id}/deepfake/analyze",
            method="POST"
        )
        urllib.request.urlopen(req)
        print("[FAIL] Analysis before preprocessing did not trigger error!")
        sys.exit(1)
    except urllib.error.HTTPError as e:
        if e.code in (400, 409):
            print(f"[SUCCESS] Incomplete preprocessing correctly returned {e.code}.")
        else:
            print(f"[FAIL] Incomplete preprocessing returned unexpected status {e.code}")
            cleanup(img_path)
            sys.exit(1)

    # 4. Trigger Preprocessing (Phase 3) so it completes with 0 faces
    print("Running preprocessing...")
    preprocess_id = None
    try:
        req = urllib.request.Request(
            f"{BASE_URL}/evidence/{evidence_id}/preprocess",
            method="POST"
        )
        with urllib.request.urlopen(req) as res:
            data = json.loads(res.read().decode('utf-8'))
            preprocess_id = data["analysis_id"]
            print(f"[SUCCESS] Preprocessing completed. Analysis ID: {preprocess_id}")
    except Exception as e:
        print(f"[FAIL] Preprocessing failed: {e}")
        cleanup(img_path)
        sys.exit(1)

    # 5. Test evidence with no detected faces (expect NO_FACE_DETECTED overall)
    print("Test 3: Deepfake analysis on 0 faces image...")
    try:
        req = urllib.request.Request(
            f"{BASE_URL}/evidence/{evidence_id}/deepfake/analyze",
            method="POST"
        )
        with urllib.request.urlopen(req) as res:
            data = json.loads(res.read().decode('utf-8'))
            print(f"[SUCCESS] Deepfake analysis response for 0 faces: {data}")
            assert data["overall_label"] == "NO_FACE_DETECTED"
            assert data["overall_confidence"] is None
            assert data["status"] == "COMPLETED"
    except Exception as e:
        print(f"[FAIL] Deepfake analysis on 0 faces failed: {e}")
        cleanup(img_path)
        sys.exit(1)

    # 6. Test deepfake analysis with detected faces but no model weights (expect MODEL_NOT_CONFIGURED)
    # We will inject a mock face detection record into the SQLite database for this preprocessing run
    print("Injecting mock face detection record in SQLite...")
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        # Insert a face crop coordinates row into face_detection referencing the preprocess_id
        cursor.execute(
            "INSERT INTO face_detection (analysis_id, face_index, x, y, width, height, crop_path) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (preprocess_id, 0, 50, 50, 100, 100, "crop_mock.jpg")
        )
        conn.commit()
        conn.close()
        print("[SUCCESS] Mock face detection injected.")
        
        # Create dummy crop image file to support real inference
        crops_dir = backend_dir / "storage" / "analysis" / "face_crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        dummy_crop_path = crops_dir / "crop_mock.jpg"
        import cv2
        import numpy as np
        dummy_crop = np.ones((100, 100, 3), dtype=np.uint8) * 255
        cv2.imwrite(str(dummy_crop_path), dummy_crop)
        print(f"[SUCCESS] Dummy crop image created at {dummy_crop_path}")
    except Exception as e:
        print(f"[FAIL] Direct SQLite insertion / crop creation failed: {e}")
        cleanup(img_path)
        sys.exit(1)

    # Now, run deepfake analysis again. It should detect the mock face, run the pipeline, 
    # and return COMPLETED with overall_label="MODEL_NOT_CONFIGURED" and create deepfake_prediction records
    print("Test 4: Deepfake analysis on image containing detected face but unconfigured model...")
    deepfake_analysis_id = None
    try:
        req = urllib.request.Request(
            f"{BASE_URL}/evidence/{evidence_id}/deepfake/analyze",
            method="POST"
        )
        with urllib.request.urlopen(req) as res:
            data = json.loads(res.read().decode('utf-8'))
            print(f"[SUCCESS] Deepfake analysis response for unconfigured model: {data}")
            deepfake_analysis_id = data["id"]
            assert data["overall_label"] in ("MODEL_NOT_CONFIGURED", "REAL", "FAKE")
            if data["overall_label"] == "MODEL_NOT_CONFIGURED":
                assert data["overall_confidence"] is None
            else:
                assert data["overall_confidence"] is not None
                assert 0.0 <= data["overall_confidence"] <= 1.0
            assert data["status"] == "COMPLETED"
    except Exception as e:
        print(f"[FAIL] Deepfake analysis with mock face failed: {e}")
        cleanup(img_path)
        sys.exit(1)

    # 7. Check database persistence: retrieve predictions list from endpoint
    print("Test 5: Retrieving individual face predictions list...")
    try:
        with urllib.request.urlopen(f"{BASE_URL}/evidence/{evidence_id}/deepfake/predictions") as res:
            predictions = json.loads(res.read().decode('utf-8'))
            print(f"[SUCCESS] Face predictions list: {predictions}")
            assert len(predictions) == 1
            assert predictions[0]["label"] in ("MODEL_NOT_CONFIGURED", "REAL", "FAKE")
            assert predictions[0]["model_status"] in ("MODEL_NOT_CONFIGURED", "CONFIGURED")
            if predictions[0]["model_status"] == "MODEL_NOT_CONFIGURED":
                assert predictions[0]["confidence"] is None
            else:
                assert predictions[0]["confidence"] is not None
                assert 0.0 <= predictions[0]["confidence"] <= 1.0
    except Exception as e:
        print(f"[FAIL] Retrieving face predictions failed: {e}")
        cleanup(img_path)
        sys.exit(1)

    # 8. Check audit log event trails
    print("Test 6: Retrieving custody audit logs to verify events...")
    try:
        with urllib.request.urlopen(f"{BASE_URL}/evidence/{evidence_id}/audit") as res:
            logs = json.loads(res.read().decode('utf-8'))
            print("[SUCCESS] Audit logs retrieved:")
            for l in logs:
                print(f"  - {l['action']}: {l['details']}")
            
            # Check for DEEPFAKE_ANALYSIS_STARTED and DEEPFAKE_ANALYSIS_COMPLETED
            actions = [l["action"] for l in logs]
            assert "DEEPFAKE_ANALYSIS_STARTED" in actions
            assert "DEEPFAKE_ANALYSIS_COMPLETED" in actions
            if "MODEL_NOT_CONFIGURED" in actions:
                print("[INFO] Fallback deepfake audit trail verified.")
            else:
                print("[INFO] Production deepfake audit trail verified.")
            print("[SUCCESS] All expected deepfake audit events found.")
    except Exception as e:
        print(f"[FAIL] Auditing checks failed: {e}")
        cleanup(img_path)
        sys.exit(1)

    # 9. Test SHA-256 Tampering check mismatch before deepfake run
    print("Test 7: Tampering file and triggering deepfake analysis...")
    try:
        # Resolve physical storage path from DB
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT storage_path FROM evidence WHERE id = ?", (evidence_id,))
        stored_path_str = cursor.fetchone()[0]
        conn.close()
        
        stored_path = Path(stored_path_str)
        with open(stored_path, "rb") as f:
            orig_bytes = f.read()

        # Write different bytes
        with open(stored_path, "wb") as f:
            f.write(b"tampered deepfake analysis bytes")

        # Trigger analysis
        req = urllib.request.Request(
            f"{BASE_URL}/evidence/{evidence_id}/deepfake/analyze",
            method="POST"
        )
        try:
            urllib.request.urlopen(req)
            print("[FAIL] Tampered file did not trigger HTTP 409!")
            # Restore file
            with open(stored_path, "wb") as f:
                f.write(orig_bytes)
            cleanup(img_path)
            sys.exit(1)
        except urllib.error.HTTPError as e:
            if e.code == 409:
                print("[SUCCESS] Tampered deepfake trigger correctly returned 409 Conflict.")
            else:
                print(f"[FAIL] Tampered deepfake trigger returned unexpected status {e.code}")
                # Restore file
                with open(stored_path, "wb") as f:
                    f.write(orig_bytes)
                cleanup(img_path)
                sys.exit(1)

        # Restore file
        with open(stored_path, "wb") as f:
            f.write(orig_bytes)

        # Confirm audit log recorded integrity failure
        with urllib.request.urlopen(f"{BASE_URL}/evidence/{evidence_id}/audit") as res:
            logs = json.loads(res.read().decode('utf-8'))
            actions = [l["action"] for l in logs]
            assert "INTEGRITY_VERIFICATION_FAILED" in actions
            print("[SUCCESS] INTEGRITY_VERIFICATION_FAILED audit event recorded.")

    except Exception as e:
        print(f"[FAIL] Tampering test failed: {e}")
        cleanup(img_path)
        sys.exit(1)

    # Cleanup
    cleanup(img_path)
    print("=== ALL PHASE 4 TEST PIPELINE CHECKS PASSED SUCCESSFULLY ===")

def cleanup(img_path):
    if img_path.exists():
        img_path.unlink()
    dummy_crop_path = Path(__file__).resolve().parent / "storage" / "analysis" / "face_crops" / "crop_mock.jpg"
    if dummy_crop_path.exists():
        dummy_crop_path.unlink()

if __name__ == "__main__":
    run_phase4_tests()
