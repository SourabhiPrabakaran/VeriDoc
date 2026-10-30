import urllib.request
import urllib.error
import json
import uuid
import sys
import sqlite3
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000/api"

def run_phase6_tests():
    print("=== STARTING VERIDOC FORENSICS PHASE 6 REPORT TEST PIPELINE ===")

    backend_dir = Path(__file__).resolve().parent
    db_path = backend_dir / "data" / "veridoc.db"

    # Test 1: Generate report for non-existent evidence ID -> Expect 404
    print("Test 1: Requesting report generation for non-existent evidence (ID: 99999)...")
    try:
        req = urllib.request.Request(
            f"{BASE_URL}/evidence/99999/report",
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

    # Test 2: Upload a new evidence image and trigger report generation
    print("\nTest 2: Preparing and uploading test image for report pipeline...")
    from PIL import Image, ImageDraw
    img_path = backend_dir / "test_image_p6.jpg"
    img = Image.new('RGB', (200, 200), color = (90, 120, 150))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), str(uuid.uuid4()), fill=(255, 255, 255))
    img.save(img_path)

    # Upload helper
    boundary = "----WebKitFormBoundaryPhase6Test"
    with open(img_path, "rb") as f:
        file_bytes = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="test_image_p6.jpg"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode('utf-8') + file_bytes + f"\r\n--{boundary}--\r\n".encode('utf-8')

    req = urllib.request.Request(
        f"{BASE_URL}/evidence/upload",
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}"
        }
    )

    evidence_id = None
    try:
        with urllib.request.urlopen(req) as res:
            data = json.loads(res.read().decode('utf-8'))
            evidence_id = data["id"]
            print(f"[SUCCESS] Image registered with ID {evidence_id}")
    except Exception as e:
        print(f"[FAIL] Upload test image failed: {e}")
        if img_path.exists():
            img_path.unlink()
        sys.exit(1)

    # Trigger Preprocessing (Phase 3)
    print("Running preprocessing on test image...")
    try:
        req = urllib.request.Request(
            f"{BASE_URL}/evidence/{evidence_id}/preprocess",
            method="POST"
        )
        with urllib.request.urlopen(req) as res:
            data = json.loads(res.read().decode('utf-8'))
            print(f"[SUCCESS] Preprocessing completed. Status: {data['status']}")
    except Exception as e:
        print(f"[FAIL] Preprocessing failed: {e}")
        cleanup(img_path)
        sys.exit(1)

    # Run ELA Analysis (Phase 5)
    print("Running ELA analysis on test image...")
    try:
        req = urllib.request.Request(
            f"{BASE_URL}/evidence/{evidence_id}/ela/analyze",
            method="POST"
        )
        with urllib.request.urlopen(req) as res:
            data = json.loads(res.read().decode('utf-8'))
            print(f"[SUCCESS] ELA completed. Mean error: {data['mean_error']}")
    except Exception as e:
        print(f"[FAIL] ELA analysis failed: {e}")
        cleanup(img_path)
        sys.exit(1)

    # Run FFT Analysis (Phase 5)
    print("Running FFT analysis on test image...")
    try:
        req = urllib.request.Request(
            f"{BASE_URL}/evidence/{evidence_id}/fft/analyze",
            method="POST"
        )
        with urllib.request.urlopen(req) as res:
            data = json.loads(res.read().decode('utf-8'))
            print(f"[SUCCESS] FFT completed. Avg magnitude: {data['average_magnitude']}")
    except Exception as e:
        print(f"[FAIL] FFT analysis failed: {e}")
        cleanup(img_path)
        sys.exit(1)

    # Run Deepfake Analysis (Phase 4)
    print("Running Deepfake analysis on test image...")
    try:
        req = urllib.request.Request(
            f"{BASE_URL}/evidence/{evidence_id}/deepfake/analyze",
            method="POST"
        )
        with urllib.request.urlopen(req) as res:
            data = json.loads(res.read().decode('utf-8'))
            print(f"[SUCCESS] Deepfake completed. Overall label: {data['overall_label']}")
    except Exception as e:
        print(f"[FAIL] Deepfake analysis failed: {e}")
        cleanup(img_path)
        sys.exit(1)

    # Test 3: Trigger Forensic Report generation
    print("\nTest 3: Generating Forensic PDF report...")
    report_uuid = None
    report_id = None
    try:
        req = urllib.request.Request(
            f"{BASE_URL}/evidence/{evidence_id}/report",
            method="POST"
        )
        with urllib.request.urlopen(req) as res:
            data = json.loads(res.read().decode('utf-8'))
            print(f"[SUCCESS] Report Response: {data}")
            report_uuid = data["report_uuid"]
            report_id = data["id"]
            assert data["evidence_id"] == evidence_id
            assert data["status"] == "COMPLETED"
            assert data["file_hash"] is not None
            assert "report_uuid" in data
    except Exception as e:
        print(f"[FAIL] Report generation failed: {e}")
        cleanup(img_path)
        sys.exit(1)

    # Test 4: Verify ForensicReport record exists in SQLite
    print("\nTest 4: Verifying report persistence in SQLite...")
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT status, file_path, file_hash FROM forensic_report WHERE id = ?", (report_id,))
        row = cursor.fetchone()
        conn.close()
        assert row is not None
        assert row[0] == "COMPLETED"
        print(f"[SUCCESS] ForensicReport record exists. Path: {row[1]}, Hash: {row[2]}")
    except Exception as e:
        print(f"[FAIL] Database verification failed: {e}")
        cleanup(img_path)
        sys.exit(1)

    # Test 5: Verify report listing works
    print("\nTest 5: Retrieving reports listing...")
    try:
        with urllib.request.urlopen(f"{BASE_URL}/evidence/{evidence_id}/reports") as res:
            data = json.loads(res.read().decode('utf-8'))
            print(f"[SUCCESS] Listed Reports: {data}")
            assert "items" in data
            assert len(data["items"]) >= 1
            assert data["items"][0]["report_uuid"] == report_uuid
    except Exception as e:
        print(f"[FAIL] Reports listing failed: {e}")
        cleanup(img_path)
        sys.exit(1)

    # Test 6: Verify download works and returns PDF
    print("\nTest 6: Downloading generated report PDF...")
    try:
        with urllib.request.urlopen(f"{BASE_URL}/reports/{report_uuid}/download") as res:
            pdf_bytes = res.read()
            # Assert file begins with %PDF-
            pdf_header = pdf_bytes[:5]
            print(f"[SUCCESS] Download completed. PDF header: {pdf_header}")
            assert pdf_header == b"%PDF-"
    except Exception as e:
        print(f"[FAIL] Downloading report failed: {e}")
        cleanup(img_path)
        sys.exit(1)

    # Test 7: Verify missing report file is handled safely
    print("\nTest 7: Testing missing report file handling (deleting file and querying)...")
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT file_path FROM forensic_report WHERE report_uuid = ?", (report_uuid,))
        phys_path_str = cursor.fetchone()[0]
        conn.close()

        phys_path = Path(phys_path_str)
        # Temporarily backup the file
        backup_path = phys_path.with_suffix(".pdf.bak")
        phys_path.rename(backup_path)

        # Trigger download endpoint -> expect 404
        req = urllib.request.Request(f"{BASE_URL}/reports/{report_uuid}/download")
        try:
            urllib.request.urlopen(req)
            print("[FAIL] Missing report file on disk did not trigger 404!")
            # Restore backup
            if backup_path.exists():
                backup_path.rename(phys_path)
            cleanup(img_path)
            sys.exit(1)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print("[SUCCESS] Missing report file on disk correctly returned 404.")
            else:
                print(f"[FAIL] Missing report file returned unexpected status {e.code}")
                if backup_path.exists():
                    backup_path.rename(phys_path)
                cleanup(img_path)
                sys.exit(1)

        # Restore file
        if backup_path.exists():
            backup_path.rename(phys_path)
    except Exception as e:
        print(f"[FAIL] Safe handling of missing report file test failed: {e}")
        cleanup(img_path)
        sys.exit(1)

    # Test 8: Verify audit logs track report actions
    print("\nTest 8: Checking custody audit logs for report generation events...")
    try:
        with urllib.request.urlopen(f"{BASE_URL}/evidence/{evidence_id}/audit") as res:
            logs = json.loads(res.read().decode('utf-8'))
            actions = [l["action"] for l in logs]
            print(f"[SUCCESS] Audit actions logged: {actions}")
            assert "REPORT_GENERATION_STARTED" in actions
            assert "REPORT_GENERATION_COMPLETED" in actions
    except Exception as e:
        print(f"[FAIL] Custody logs verification failed: {e}")
        cleanup(img_path)
        sys.exit(1)

    cleanup(img_path)
    print("\n=== ALL PHASE 6 REPORT TEST PIPELINE CHECKS PASSED SUCCESSFULLY ===")

def cleanup(path):
    if path.exists():
        path.unlink()

if __name__ == "__main__":
    run_phase6_tests()
