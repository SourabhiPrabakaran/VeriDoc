import urllib.request
import urllib.error
import json
import uuid
import sys
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000/api"

def run_phase3_tests():
    print("=== STARTING VERIDOC FORENSICS PHASE 3 TEST PIPELINE ===")

    # Import cv2, numpy, PIL locally to generate files
    import cv2
    import numpy as np
    from PIL import Image

    # 1. Prepare test assets
    temp_dir = Path(__file__).resolve().parent
    img_path = temp_dir / "test_image.jpg"
    video_path = temp_dir / "test_video.mp4"

    import random
    from PIL import ImageDraw

    # Create dummy JPG
    img = Image.new('RGB', (320, 240), color = (73, 109, 137))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), str(uuid.uuid4()), fill=(255, 255, 255))
    img.save(img_path)

    # Create dummy MP4 video (5 seconds, 30 FPS, 150 frames)
    start_y = random.randint(50, 190)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (320, 240))
    for i in range(150):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.circle(frame, (10 + i * 2, start_y), 20, (0, 240, 255), -1)
        out.write(frame)
    out.release()

    # Helper function for multipart upload
    def upload_file(file_path, filename, mime_type):
        boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
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

    # Upload Image
    img_evidence_id = None
    img_stored_path = None
    img_hash = None
    try:
        req = upload_file(img_path, "test_image.jpg", "image/jpeg")
        with urllib.request.urlopen(req) as res:
            data = json.loads(res.read().decode('utf-8'))
            img_evidence_id = data["id"]
            img_hash = data["sha256_hash"]
            print(f"[SUCCESS] Ingest Image ID: {img_evidence_id}")
    except Exception as e:
        print(f"[FAIL] Upload image failed: {e}")
        cleanup_files(img_path, video_path)
        sys.exit(1)

    # Upload Video
    video_evidence_id = None
    try:
        req = upload_file(video_path, "test_video.mp4", "video/mp4")
        with urllib.request.urlopen(req) as res:
            data = json.loads(res.read().decode('utf-8'))
            video_evidence_id = data["id"]
            print(f"[SUCCESS] Ingest Video ID: {video_evidence_id}")
    except Exception as e:
        print(f"[FAIL] Upload video failed: {e}")
        cleanup_files(img_path, video_path)
        sys.exit(1)

    # 2. Preprocess Image
    try:
        req = urllib.request.Request(
            f"{BASE_URL}/evidence/{img_evidence_id}/preprocess",
            method="POST"
        )
        with urllib.request.urlopen(req) as res:
            data = json.loads(res.read().decode('utf-8'))
            print(f"[SUCCESS] Preprocess Image trigger response: {data}")
            assert data["evidence_id"] == img_evidence_id
            assert data["status"] == "COMPLETED"
            assert data["integrity_verified"] is True
    except Exception as e:
        print(f"[FAIL] Preprocess image trigger failed: {e}")
        cleanup_files(img_path, video_path)
        sys.exit(1)

    # 3. Retrieve Image Preprocessing results
    try:
        with urllib.request.urlopen(f"{BASE_URL}/evidence/{img_evidence_id}/preprocessing") as res:
            data = json.loads(res.read().decode('utf-8'))
            print(f"[SUCCESS] Get Image preprocessing results: {data}")
            assert data["status"] == "COMPLETED"
            assert data["integrity_verified"] is True
            assert "faces" in data
            assert data["faces_detected"] == len(data["faces"])
    except Exception as e:
        print(f"[FAIL] Query image preprocessing failed: {e}")
        cleanup_files(img_path, video_path)
        sys.exit(1)

    # 4. Preprocess Video
    try:
        req = urllib.request.Request(
            f"{BASE_URL}/evidence/{video_evidence_id}/preprocess",
            method="POST"
        )
        with urllib.request.urlopen(req) as res:
            data = json.loads(res.read().decode('utf-8'))
            print(f"[SUCCESS] Preprocess Video trigger response: {data}")
            assert data["evidence_id"] == video_evidence_id
            assert data["status"] == "COMPLETED"
            assert data["integrity_verified"] is True
    except Exception as e:
        print(f"[FAIL] Preprocess video trigger failed: {e}")
        cleanup_files(img_path, video_path)
        sys.exit(1)

    # 5. Retrieve Video Preprocessing details
    try:
        with urllib.request.urlopen(f"{BASE_URL}/evidence/{video_evidence_id}/preprocessing") as res:
            data = json.loads(res.read().decode('utf-8'))
            print(f"[SUCCESS] Get Video preprocessing details: {data}")
            assert data["status"] == "COMPLETED"
            assert data["video_metadata"] is not None
            assert data["video_metadata"]["frame_count"] == 150
            assert data["video_metadata"]["duration_seconds"] == 5.0
            assert data["video_metadata"]["sampled_frames"] == 5 # 1 frame/s for 5s video
    except Exception as e:
        print(f"[FAIL] Query video preprocessing failed: {e}")
        cleanup_files(img_path, video_path)
        sys.exit(1)

    # 6. Test SHA-256 Integrity check mismatch
    # Overwrite the image on disk temporarily to cause a hash mismatch
    # First, let's find the stored path from /api/evidence/{id}
    try:
        with urllib.request.urlopen(f"{BASE_URL}/evidence/{img_evidence_id}") as res:
            detail = json.loads(res.read().decode('utf-8'))
            # We don't return storage_path directly in schema if we normalized, but wait!
            # Did we include storage_path in schemas.py?
            # Let's see: in schemas/evidence.py, we only return EvidenceBase fields (original_filename, media_category, mime_type, file_size, file_extension, sha256_hash, status).
            # We did not include storage_path in the output schema (which is good for security).
            # But wait! How can we locate the file on disk to tamper with it?
            # We know the files are stored in `backend/storage/evidence/` under their safe UUID filenames.
            # We don't have the UUID, but we can look it up in the database or just find the file in the directory that matches!
            # Let's list the directory backend/storage/evidence/ and find the file that has the exact same size or hash!
            # Or we can write a quick python snippet that queries the DB directly to get the path and tamper with it!
            # Yes! We can run a python command that modifies the file.
            pass
    except Exception as e:
        print(f"[FAIL] Failed to read detail: {e}")

    # Let's write a direct tampering & preprocess check inside python:
    # We can do this by executing a python block that runs the database lookup and modifies the file, then runs the API call.
    # To do it inside this script, we can query SQLite directly since we have the db file in the workspace!
    import sqlite3
    db_path = Path(__file__).resolve().parent / "data" / "veridoc.db"
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT storage_path FROM evidence WHERE id = ?", (img_evidence_id,))
    row = cursor.fetchone()
    stored_path_str = row[0]
    conn.close()

    stored_path = Path(stored_path_str)
    
    # Save original bytes
    with open(stored_path, "rb") as f:
        original_bytes = f.read()

    # Tamper: Write different bytes
    with open(stored_path, "wb") as f:
        f.write(b"Tampered content bytes")

    # Trigger preprocess on tampered file -> Expect 409 Conflict
    try:
        req = urllib.request.Request(
            f"{BASE_URL}/evidence/{img_evidence_id}/preprocess",
            method="POST"
        )
        with urllib.request.urlopen(req) as res:
            print("[FAIL] Processing tampered file did not trigger conflict!")
            sys.exit(1)
    except urllib.error.HTTPError as e:
        if e.code == 409:
            err_msg = json.loads(e.read().decode('utf-8'))["detail"]
            print(f"[SUCCESS] Tampered file successfully rejected with 409: {err_msg}")
        else:
            print(f"[FAIL] Tampered file returned unexpected status code {e.code}")
            # Restore file
            with open(stored_path, "wb") as f:
                f.write(original_bytes)
            cleanup_files(img_path, video_path)
            sys.exit(1)

    # Restore original bytes
    with open(stored_path, "wb") as f:
        f.write(original_bytes)

    # 7. Check Audit Logs for the integrity failure
    try:
        with urllib.request.urlopen(f"{BASE_URL}/evidence/{img_evidence_id}/audit") as res:
            logs = json.loads(res.read().decode('utf-8'))
            print(f"[SUCCESS] Audit logs retrieved after tamper test: {logs}")
            # Verify that integrity verification failed audit log exists
            failed_log = [l for l in logs if l["action"] == "INTEGRITY_VERIFICATION_FAILED"]
            assert len(failed_log) >= 1
            print(f"[SUCCESS] Found expected audit log: {failed_log[0]}")
    except Exception as e:
        print(f"[FAIL] Audit log verification failed: {e}")
        cleanup_files(img_path, video_path)
        sys.exit(1)

    cleanup_files(img_path, video_path)
    print("=== ALL PHASE 3 TEST PIPELINE CHECKS PASSED SUCCESSFULLY ===")

def cleanup_files(*files):
    for f in files:
        if f.exists():
            f.unlink()

if __name__ == "__main__":
    run_phase3_tests()
