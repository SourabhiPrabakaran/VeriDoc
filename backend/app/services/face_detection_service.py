import cv2
import os
from pathlib import Path

class FaceDetectionService:
    def __init__(self):
        # Resolve the Haar cascade path safely from the installed cv2 package location
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Haar cascade file could not be located at: {cascade_path}")
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect_faces(self, image_gray) -> list[dict]:
        """
        Run Haar Cascade face detection on a grayscale frame.
        Returns: A list of clamped coordinate dictionaries [{"x": x, "y": y, "width": w, "height": h}]
        """
        if image_gray is None:
            return []

        # Enforce grayscale representation
        if len(image_gray.shape) != 2:
            image_gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2GRAY)

        height, width = image_gray.shape[:2]

        # scaleFactor and minNeighbors are balanced for local dev verification
        faces = self.face_cascade.detectMultiScale(
            image_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        results = []
        for (x, y, w, h) in faces:
            # Enforce strict coordinate boundaries (clamping)
            clamped_x = max(0, min(int(x), width - 1))
            clamped_y = max(0, min(int(y), height - 1))
            clamped_w = max(1, min(int(w), width - clamped_x))
            clamped_h = max(1, min(int(h), height - clamped_y))

            results.append({
                "x": clamped_x,
                "y": clamped_y,
                "width": clamped_w,
                "height": clamped_h
            })

        return results
