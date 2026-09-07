import urllib.request
import urllib.error
import json
import uuid
import sys
import sqlite3
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000/api"

def run_phase5_tests():
    print("=== STARTING VERIDOC FORENSICS PHASE 5 TEST PIPELINE ===")
    
    backend_dir = Path(__file__).resolve().parent
    db_path = backend_dir / "data" / "veridoc.db"
    
    # 1. Missing evidence ELA & FFT -> 404
    print("Test 1: Querying missing evidence for ELA and FFT...")
    for endpoint in ("ela", "fft"):
        try:
            req = urllib.request.Request(f"{BASE_URL}/evidence/99999/{endpoint}/analyze", method="POST")
            urllib.request.urlopen(req)
            print(f"[FAIL] Missing evidence did not trigger 404 on {endpoint}!")
            sys.exit(1)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"[SUCCESS] Missing evidence correctly returned 404 on {endpoint}.")
            else:
                print(f"[FAIL] Missing evidence returned unexpected status {e.code} on {endpoint}")
                sys.exit(1)

    # Helper function to generate and upload image
    from PIL import Image, ImageDraw
    
    def create_and_upload_image(filename, ext, mime_type, mode='RGB'):
        img_path = backend_dir / f"test_p5_{filename}.{ext}"
        if mode == 'RGBA':
            img = Image.new('RGBA', (200, 200), color=(100, 150, 200, 255))
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), str(uuid.uuid4()), fill=(255, 255, 255, 255))
        else:
            img = Image.new('RGB', (200, 200), color=(100, 150, 200))
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), str(uuid.uuid4()), fill=(255, 255, 255))
        
        img.save(img_path)
        
        # Upload
        boundary = "----WebKitFormBoundaryPhase5Test"
        with open(img_path, "rb") as f:
            file_bytes = f.read()

        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}.{ext}"\r\n'
            f"Content-Type: {mime_type}\r\n\r\n"
        ).encode('utf-8') + file_bytes + f"\r\n--{boundary}--\r\n".encode('utf-8')

        req = urllib.request.Request(
            f"{BASE_URL}/evidence/upload",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
        )
        
        try:
            with urllib.request.urlopen(req) as res:
                data = json.loads(res.read().decode('utf-8'))
                img_id = data["id"]
                print(f"[SUCCESS] Image uploaded successfully. ID: {img_id}, Ext: {ext}")
                return img_id, img_path
        except Exception as e:
            print(f"[FAIL] Upload failed for {filename}.{ext}: {e}")
            if img_path.exists():
                img_path.unlink()
            sys.exit(1)

    # 2. Test JPEG Image (ELA & FFT)
    print("Testing JPEG image...")
    jpeg_id, jpeg_path = create_and_upload_image("img_jpeg", "jpg", "image/jpeg")
    
    # ELA
    print("Test 2a: Running ELA on JPEG image...")
    try:
        req = urllib.request.Request(f"{BASE_URL}/evidence/{jpeg_id}/ela/analyze", method="POST")
        with urllib.request.urlopen(req) as res:
            ela_res = json.loads(res.read().decode('utf-8'))
            print(f"[SUCCESS] ELA Response: {ela_res}")
            assert ela_res["status"] == "COMPLETED"
            assert ela_res["mean_error"] is not None
            assert ela_res["max_error"] is not None
            assert ela_res["error_variance"] is not None
            assert ela_res["jpeg_quality"] == 95
            assert ela_res["visualization_path"] is not None
    except Exception as e:
        print(f"[FAIL] ELA on JPEG failed: {e}")
        cleanup(jpeg_path)
        sys.exit(1)

    # FFT
    print("Test 2b: Running FFT on JPEG image...")
    try:
        req = urllib.request.Request(f"{BASE_URL}/evidence/{jpeg_id}/fft/analyze", method="POST")
        with urllib.request.urlopen(req) as res:
            fft_res = json.loads(res.read().decode('utf-8'))
            print(f"[SUCCESS] FFT Response: {fft_res}")
            assert fft_res["status"] == "COMPLETED"
            assert fft_res["average_magnitude"] is not None
            assert fft_res["high_frequency_ratio"] is not None
            assert fft_res["spectral_entropy"] is not None
            assert fft_res["high_freq_radius_percent"] == 15.0
            assert fft_res["visualization_path"] is not None
    except Exception as e:
        print(f"[FAIL] FFT on JPEG failed: {e}")
        cleanup(jpeg_path)
        sys.exit(1)

    # 3. Test PNG (RGBA) Image (ELA & FFT safe handling)
    print("Testing PNG (RGBA) image for safe color channel conversion...")
    png_id, png_path = create_and_upload_image("img_png", "png", "image/png", mode='RGBA')
    
    # ELA on PNG
    print("Test 3a: Running ELA on PNG (RGBA) image...")
    try:
        req = urllib.request.Request(f"{BASE_URL}/evidence/{png_id}/ela/analyze", method="POST")
        with urllib.request.urlopen(req) as res:
            ela_res = json.loads(res.read().decode('utf-8'))
            print(f"[SUCCESS] ELA on PNG Response: {ela_res}")
            assert ela_res["status"] == "COMPLETED"
    except Exception as e:
        print(f"[FAIL] ELA on PNG (RGBA) failed: {e}")
        cleanup(jpeg_path, png_path)
        sys.exit(1)

    # FFT on PNG
    print("Test 3b: Running FFT on PNG (RGBA) image...")
    try:
        req = urllib.request.Request(f"{BASE_URL}/evidence/{png_id}/fft/analyze", method="POST")
        with urllib.request.urlopen(req) as res:
            fft_res = json.loads(res.read().decode('utf-8'))
            print(f"[SUCCESS] FFT on PNG Response: {fft_res}")
            assert fft_res["status"] == "COMPLETED"
    except Exception as e:
        print(f"[FAIL] FFT on PNG (RGBA) failed: {e}")
        cleanup(jpeg_path, png_path)
        sys.exit(1)

    # 4. Test WebP Image
    print("Testing WebP image...")
    webp_id, webp_path = create_and_upload_image("img_webp", "webp", "image/webp")
    
    # ELA on WebP
    print("Test 4a: Running ELA on WebP image...")
    try:
        req = urllib.request.Request(f"{BASE_URL}/evidence/{webp_id}/ela/analyze", method="POST")
        with urllib.request.urlopen(req) as res:
            ela_res = json.loads(res.read().decode('utf-8'))
            print(f"[SUCCESS] ELA on WebP Response: {ela_res}")
            assert ela_res["status"] == "COMPLETED"
    except Exception as e:
        print(f"[FAIL] ELA on WebP failed: {e}")
        cleanup(jpeg_path, png_path, webp_path)
        sys.exit(1)

    # FFT on WebP
    print("Test 4b: Running FFT on WebP image...")
    try:
        req = urllib.request.Request(f"{BASE_URL}/evidence/{webp_id}/fft/analyze", method="POST")
        with urllib.request.urlopen(req) as res:
            fft_res = json.loads(res.read().decode('utf-8'))
            print(f"[SUCCESS] FFT on WebP Response: {fft_res}")
            assert fft_res["status"] == "COMPLETED"
    except Exception as e:
        print(f"[FAIL] FFT on WebP failed: {e}")
        cleanup(jpeg_path, png_path, webp_path)
        sys.exit(1)

    # 5. Test SHA-256 Tampering check before ELA and FFT runs (expect 409)
    print("Test 5: Testing SHA-256 tampering rejection for ELA & FFT...")
    try:
        # Resolve physical storage path from DB
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT storage_path FROM evidence WHERE id = ?", (jpeg_id,))
        stored_path_str = cursor.fetchone()[0]
        conn.close()
        
        stored_path = Path(stored_path_str)
        with open(stored_path, "rb") as f:
            orig_bytes = f.read()

        # Write different bytes
        with open(stored_path, "wb") as f:
            f.write(b"tampered signal analysis bytes")

        # Trigger ELA -> expect 409
        req = urllib.request.Request(f"{BASE_URL}/evidence/{jpeg_id}/ela/analyze", method="POST")
        try:
            urllib.request.urlopen(req)
            print("[FAIL] Tampered file did not trigger HTTP 409 on ELA!")
            with open(stored_path, "wb") as f: f.write(orig_bytes)
            cleanup(jpeg_path, png_path, webp_path)
            sys.exit(1)
        except urllib.error.HTTPError as e:
            if e.code == 409:
                print("[SUCCESS] Tampered ELA trigger correctly returned 409 Conflict.")
            else:
                print(f"[FAIL] Tampered ELA returned status {e.code}")
                with open(stored_path, "wb") as f: f.write(orig_bytes)
                cleanup(jpeg_path, png_path, webp_path)
                sys.exit(1)

        # Trigger FFT -> expect 409
        req = urllib.request.Request(f"{BASE_URL}/evidence/{jpeg_id}/fft/analyze", method="POST")
        try:
            urllib.request.urlopen(req)
            print("[FAIL] Tampered file did not trigger HTTP 409 on FFT!")
            with open(stored_path, "wb") as f: f.write(orig_bytes)
            cleanup(jpeg_path, png_path, webp_path)
            sys.exit(1)
        except urllib.error.HTTPError as e:
            if e.code == 409:
                print("[SUCCESS] Tampered FFT trigger correctly returned 409 Conflict.")
            else:
                print(f"[FAIL] Tampered FFT returned status {e.code}")
                with open(stored_path, "wb") as f: f.write(orig_bytes)
                cleanup(jpeg_path, png_path, webp_path)
                sys.exit(1)

        # Restore file
        with open(stored_path, "wb") as f:
            f.write(orig_bytes)

    except Exception as e:
        print(f"[FAIL] Tampering tests failed: {e}")
        cleanup(jpeg_path, png_path, webp_path)
        sys.exit(1)

    # 6. Retrieve custody audit logs to verify events
    print("Test 6: Retrieving audit logs to verify forensic signal actions...")
    try:
        with urllib.request.urlopen(f"{BASE_URL}/evidence/{jpeg_id}/audit") as res:
            logs = json.loads(res.read().decode('utf-8'))
            actions = [l["action"] for l in logs]
            print(f"Audit log actions recorded: {actions}")
            
            assert "ELA_ANALYSIS_STARTED" in actions
            assert "ELA_ANALYSIS_COMPLETED" in actions
            assert "FFT_ANALYSIS_STARTED" in actions
            assert "FFT_ANALYSIS_COMPLETED" in actions
            assert "INTEGRITY_VERIFICATION_FAILED" in actions
            
            print("[SUCCESS] All expected audit log signals found.")
    except Exception as e:
        print(f"[FAIL] Audit verification failed: {e}")
        cleanup(jpeg_path, png_path, webp_path)
        sys.exit(1)

    cleanup(jpeg_path, png_path, webp_path)
    print("=== ALL PHASE 5 TEST PIPELINE CHECKS PASSED SUCCESSFULLY ===")

def cleanup(*files):
    for f in files:
        if f.exists():
            f.unlink()

if __name__ == "__main__":
    run_phase5_tests()
