#!/usr/bin/env python3
"""Check the observed example's manifest, input shapes and feature hashes."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = ROOT / "data/example"
    manifest = json.loads((base / "source_manifest.json").read_text())
    for filename, key in [
        ("tcga_coad_msi_pool.npz", "exported_pool_sha256"),
        ("tcga_coad_msi_pilot40.npz", "exported_pilot_sha256"),
    ]:
        path = base / filename
        if not path.exists():
            raise SystemExit("Prepare the public example first; see docs/data.md")
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[key]:
            raise SystemExit(f"Example file differs from recorded export: {filename}")
    with (
        np.load(base / "tcga_coad_msi_pool.npz", allow_pickle=False) as pool,
        np.load(base / "tcga_coad_msi_pilot40.npz", allow_pickle=False) as pilot,
    ):
        rows = manifest["pilot_row_indices"]
        np.testing.assert_array_equal(pool["X"][rows], pilot["X"])
        np.testing.assert_array_equal(pool["y"][rows], pilot["y"])
        assert pool["X"].shape == (112, 1536)
        assert pilot["X"].shape == (40, 1536)
        assert all(str(unit).startswith("TCGA-") for unit in pool["ids"])
    print("PASS: documented public TCGA pool and observed 40-record pilot match")


if __name__ == "__main__":
    main()
