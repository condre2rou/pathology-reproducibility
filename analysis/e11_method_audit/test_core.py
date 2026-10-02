"""Regression tests for leakage, grouping, uncertainty and decision accounting."""

import unittest
from unittest.mock import patch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from core import cv_predictions, metrics, decide, balanced_sample, oof_interval
from data import cohort_of, endpoint_of, Task
from run import test_membership


class ScientificInvariants(unittest.TestCase):
    def test_preprocessing_only_sees_training_folds(self):
        rng = np.random.RandomState(3)
        X = rng.normal(size=(40, 5))
        y = np.arange(40) % 2
        seen = []
        original = StandardScaler.fit

        def capture(obj, x, *a, **kw):
            seen.append(x.copy())
            return original(obj, x, *a, **kw)

        with patch.object(StandardScaler, "fit", capture):
            p, folds = cv_predictions(X, y, return_folds=True)
        self.assertEqual(len(seen), 5)
        for x, (tr, te) in zip(seen, folds):
            np.testing.assert_array_equal(x, X[tr])
            self.assertEqual(len(x), 32)
        self.assertTrue(np.isfinite(p).all())

    def test_shared_patients_are_grouped(self):
        self.assertEqual(cohort_of("phikon@dMMR"), cohort_of("res96@grade3"))
        self.assertEqual(cohort_of("WSI-BRCA_IDC"), cohort_of("WSI-BRCA_IDC_PR"))
        self.assertEqual(cohort_of("WSI-BRCA_IDC-TP53"), cohort_of("WSI-BRCA_IDC_PR"))
        self.assertEqual(endpoint_of("WSI-UCEC-TP53"), endpoint_of("WSI-COAD-TP53"))

    def test_split_independent_of_labels(self):
        ids = np.array(["a", "b", "c", "d", "e"])
        X = np.zeros((5, 2))
        y = np.array([0, 0, 1, 1, 0])
        t = Task("x", "cohort", "y", "cfg", X, y, ids)
        other = Task("z", "cohort", "other", "cfg2", X, y[::-1], ids)
        np.testing.assert_array_equal(test_membership(t, 0), test_membership(other, 0))

    def test_false_stop_denominator(self):
        y = np.array([1] * 72 + [0] * 62)
        stop = np.array([1] * 55 + [0] * 17 + [1] * 11 + [0] * 51)
        m = metrics(y, stop)
        self.assertAlmostEqual(m["false_stop_rate"], 11 / 66)
        self.assertAlmostEqual(m["missed_opportunity_rate"], 11 / 62)

    def test_abstention(self):
        self.assertEqual(decide(0.51, [0.40, 0.56]), "stop")
        self.assertEqual(decide(0.70, [0.59, 0.82]), "continue")
        self.assertEqual(decide(0.51, [0.30, 0.70]), "insufficient")
        self.assertEqual(decide(float("nan")), "insufficient")

    def test_interval_matches_explicit_bootstrap(self):
        y = np.arange(40) % 2
        p = np.random.RandomState(4).rand(40)
        ci = oof_interval(y, p, seed=2, n_boot=200)
        counts = np.random.RandomState(2).multinomial(40, np.full(40, 1 / 40), size=200)
        values = []
        for count in counts:
            ix = np.repeat(np.arange(40), count)
            if len(np.unique(y[ix])) == 2:
                values.append(roc_auc_score(y[ix], p[ix]))
        np.testing.assert_allclose(ci, np.quantile(values, [0.025, 0.975]))

    def test_cli_uses_one_pilot_and_matching_curve_endpoint(self):
        from cli import assess

        rng = np.random.RandomState(7)
        X = rng.normal(size=(40, 5))
        y = np.arange(40) % 2
        r = assess(X, y)
        self.assertEqual(r["auroc"], r["learning_curve"][-1]["auroc"])
        with self.assertRaises(ValueError):
            assess(X[:39], y[:39])

    def test_single_class_defers(self):
        self.assertIsNone(cv_predictions(np.ones((40, 3)), np.zeros(40, int)))
        with self.assertRaises(ValueError):
            balanced_sample(np.zeros(40, int), 40, np.random.RandomState(0))


if __name__ == "__main__":
    unittest.main()
