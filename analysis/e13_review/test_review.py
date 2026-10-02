"""Scientific regression checks for grouped evaluation and resampling."""

import unittest
from pathlib import Path
import json
import numpy as np
import pandas as pd
from reanalyse import grouped_meta, paired_gain_interval, resampled_oof


class ReviewTests(unittest.TestCase):
    def test_target_outcomes_do_not_enter_target_fit(self):
        a = np.linspace(0.1, 0.9, 24)
        y = np.tile([0, 1, 1, 0], 6)
        g = np.repeat(["A", "B", "C"], 8)
        p, folds = grouped_meta(a, y, g)
        changed = y.copy()
        changed[g == "A"] = 1 - changed[g == "A"]
        q, _ = grouped_meta(a, changed, g)
        np.testing.assert_allclose(p[g == "A"], q[g == "A"], rtol=0, atol=0)
        self.assertTrue(all(f["training_groups"] == 2 for f in folds))

    def test_identical_model_predictions_have_zero_gain_interval(self):
        y = np.array([0, 0, 1, 1, 1])
        p = np.array([0.1, 0.4, 0.3, 0.7, 0.9])
        self.assertEqual(paired_gain_interval(y, p, p, 13), [0.0, 0.0])

    def test_duplicate_record_grouping_in_refits(self):
        # Internal assertions require disjoint original row indices in every fold.
        rng = np.random.RandomState(51)
        X = rng.normal(size=(40, 6))
        y = np.arange(40) % 2
        vals = [resampled_oof(X, y, rng) for _ in range(8)]
        self.assertTrue(all(v is not None and 0 <= v <= 1 for v in vals))

    def test_all_repetitions_survive_summary_join(self):
        root = Path(__file__).resolve().parent
        original = pd.read_csv(root.parent / "e11_method_audit/outputs/single_budget_draws.csv")
        actual = pd.read_csv(root / "outputs/test_sensitivities.csv")
        expected = original[original.sampling == "natural"]
        self.assertEqual(len(actual), len(expected))
        keys = ["dataset", "cohort", "seed"]
        self.assertEqual(
            set(map(tuple, actual[keys].values)), set(map(tuple, expected[keys].values))
        )
        s = json.loads((root / "outputs/sensitivity.json").read_text())
        self.assertEqual(s["refit_diagnostic"]["n_repetitions"], 402)


if __name__ == "__main__":
    unittest.main()
