"""Scientific invariants and reproducibility-runner integration checks."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduce import stage  # noqa: E402


class ReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.out = Path(cls.temporary.name) / "run"
        cls.work = stage(cls.out)
        cls.env = os.environ.copy()
        cls.env.update(
            {k: "1" for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]}
        )
        cls.env["PYTHONDONTWRITEBYTECODE"] = "1"

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def command(self, *args, cwd=None):
        result = subprocess.run(
            [sys.executable, *map(str, args)],
            cwd=cwd or self.work,
            env=self.env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        return result

    def test_original_eight_scientific_invariants(self):
        result = self.command("-m", "unittest", "discover", "-s", "e11_method_audit", "-v")
        self.assertIn("Ran 8 tests", result.stdout)

    def test_original_four_grouping_and_refitting_checks(self):
        self.command("e13_review/reanalyse.py", "summary")
        result = self.command("-m", "unittest", "discover", "-s", "e13_review", "-v")
        self.assertIn("Ran 4 tests", result.stdout)

    def test_every_table_has_an_export_and_key_numbers_match(self):
        out = self.out / "tables"
        self.command(ROOT / "scripts/export_tables.py", "--root", self.work, "--out", out)
        mapping = json.loads((ROOT / "configs/experiment_map.json").read_text())
        self.assertEqual(len(mapping["tables"]), 16)
        self.assertEqual(len(mapping["figures"]), 13)
        for table in mapping["tables"]:
            self.assertTrue((out / (table["id"] + ".csv")).exists())
        with (out / "Table_3.csv").open() as f:
            rows = {r["policy"]: r for r in csv.DictReader(f)}
        self.assertAlmostEqual(
            float(rows["Historical 0.57"]["false_stop_percent"]), 47.78342250615149
        )
        self.assertAlmostEqual(float(rows["Always stop"]["accuracy"]), 0.5411991513023103)
        with (out / "Table_S8.csv").open() as f:
            rows = {r["task"]: r for r in csv.DictReader(f)}
        self.assertEqual(rows["serous_subtype"]["positive"], "17")
        self.assertEqual(float(rows["grade3"]["phikon"]), 0.809)
        self.assertEqual(float(rows["dMMR"]["phikon"]), 0.503)
        with (out / "Table_S1.csv").open() as f:
            reader = csv.DictReader(f)
            self.assertNotIn("ci95", reader.fieldnames)
            additional = list(reader)
            controls = [r for r in additional if r["task"] == "grade3_control"]
        self.assertEqual(len(controls), 3)
        lvsi = next(
            r for r in additional if r["task"] == "lvsi" and r["representation"] == "phikon"
        )
        self.assertEqual(float(lvsi["fold_sd"]), 0.086)
        with (out / "Table_S3.csv").open() as f:
            reader = csv.DictReader(f)
            self.assertNotIn("ci_low", reader.fieldnames)
            temporal = list(reader)
        self.assertEqual(len(temporal), 8)
        for row in temporal:
            images, records, positives = (
                (173, 163, 32) if row["test"] == "test_2023" else (121, 114, 26)
            )
            self.assertEqual(int(row["n_image_rows"]), images)
            self.assertEqual(int(row["records_with_usable_images"]), records)
            self.assertEqual(int(row["positive_records_with_usable_images"]), positives)
        with (out / "Table_S12.csv").open() as f:
            intervals = {r["interval"]: r for r in csv.DictReader(f)}
        self.assertAlmostEqual(float(intervals["refit"]["stop_percent"]), 100 * 3 / 402)

    def test_output_is_never_overwritten(self):
        with self.assertRaises(FileExistsError):
            stage(self.out)

    def test_public_manifest_matches_task_units(self):
        d = json.loads((ROOT / "data/public_tasks.json").read_text())
        self.assertEqual(d["n_tasks"], 85)
        self.assertEqual(len({t["cohort"] for t in d["tasks"]}), 15)
        for task in d["tasks"]:
            ids = [p["participant_id"] for p in task["participants"]]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(len(ids), task["n"])
            self.assertTrue(all(x.startswith("TCGA-") and len(x) == 12 for x in ids))
            self.assertEqual(sum(p["label"] for p in task["participants"]), task["positives"])

    def test_missing_real_data_are_not_replaced_with_generated_data(self):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/run_example.py"),
                "--root",
                str(self.work),
                "--data",
                str(self.out / "missing.npz"),
                "--out",
                str(self.out / "missing_example"),
            ],
            env=self.env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Real example data are missing", result.stdout)
        self.assertFalse((self.out / "missing_example").exists())

    @unittest.skipUnless(
        (ROOT / "data/example/tcga_coad_msi_pool.npz").exists(),
        "Local public features are intentionally Git-ignored",
    )
    def test_real_example_hashes(self):
        self.command(ROOT / "scripts/verify_example.py", cwd=ROOT)


if __name__ == "__main__":
    unittest.main()
