"""Verify shipped source/reference hashes without requiring feature data."""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "MANIFEST.json").read_text())
    errors = []
    for relative, expected in manifest["files"].items():
        path = ROOT / relative
        if not path.is_file():
            errors.append(relative + ": missing")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(relative + ": checksum mismatch")
    if errors:
        print("\n".join(errors))
        sys.exit(1)
    print(f"Verified {len(manifest['files'])} source/reference files")
    print("Local feature files are separately checked by scripts/verify_example.py")


if __name__ == "__main__":
    main()
