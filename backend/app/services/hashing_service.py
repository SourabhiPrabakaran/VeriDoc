import hashlib
from pathlib import Path

class HashingService:
    @staticmethod
    def calculate_sha256(file_path: Path) -> str:
        """
        Calculate SHA-256 cryptographic hash of a file by reading it in 64KB blocks.
        Guarantees that the hash represents the actual binary contents of the file.
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
