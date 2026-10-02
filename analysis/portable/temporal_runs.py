"""Run the three retained temporal configurations; GPU results may vary."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIGURATIONS = {
    "imagenet_full": [],
    "simclr_full": ["--ssl", "e2_train/runs/simclr_backbone.pt"],
    "imagenet_frozen": ["--freeze"],
}

if __name__ == "__main__":
    for name, extra in CONFIGURATIONS.items():
        subprocess.run(
            [
                sys.executable,
                "e2_train/train_baseline.py",
                "--backbone",
                "resnet50",
                "--epochs",
                "15",
                *extra,
            ],
            cwd=ROOT,
            check=True,
        )
        shutil.copy2(
            ROOT / "e2_train/runs/baseline_result.json", ROOT / f"e9_revision/finetune_{name}.json"
        )
    print(
        json.dumps(
            {
                "configurations": list(CONFIGURATIONS),
                "exact_bitwise_training_reproduction_guaranteed": False,
                "preexisting_run": "The earlier unlogged ImageNet run is retained only as an aggregate reference.",
            }
        )
    )
