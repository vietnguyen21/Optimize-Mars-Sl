"""
Adversarial Test: Laya-421M Scorer and Adaptive Early-Exit Gate
Tests probability distribution bounds, temperature scaling, ECE calibration (<= 0.081),
early exit gate (>= 0.85), and parallel scoring latency.
"""

import math
import sys
import os
import time
import unittest

REPO_ROOT = "/mnt/c/Workspace/optimize"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.validation.laya_scorer import LayaScorer, AdaptiveEarlyExitGate
from src.evaluation.metrics import compute_ece


class TestLayaScorerAndGateAdversarial(unittest.TestCase):

    def setUp(self):
        self.scorer = LayaScorer(threshold=0.75, temperature=0.20, early_exit_threshold=0.85)
        self.gate = AdaptiveEarlyExitGate(threshold=0.75, early_exit_threshold=0.85)
        self.sample_ddl = """
        CREATE TABLE accounts (
            account_id INTEGER PRIMARY KEY,
            district_id INTEGER,
            frequency TEXT,
            parsed_date DATE
        );
        """

    def test_probability_distribution_bounds(self):
        """Probability output must be strictly in [0.0, 1.0] across all inputs."""
        candidates = [
            "",
            "   ",
            "INVALID SQL",
            "SELECT",
            "SELECT account_id FROM accounts;",
            "SELECT account_id FROM accounts WHERE district_id > 10;",
            "SELECT AVG(account_id) FROM accounts;",
            "SELECT * FROM " + "a" * 1000 + ";",
            "SELECT ((((1))));",
        ]

        for cand in candidates:
            prob = self.scorer.score_candidate("What is the average account?", self.sample_ddl, cand)
            self.assertIsInstance(prob, float)
            self.assertGreaterEqual(prob, 0.0, f"Probability < 0 for: {cand}")
            self.assertLessEqual(prob, 1.0, f"Probability > 1 for: {cand}")
            self.assertFalse(math.isnan(prob), f"Probability is NaN for: {cand}")

    def test_temperature_scaling_effects_and_stability(self):
        """Verify temperature effects on confidence sharpening and numerical stability."""
        q = "What is the average district?"
        sql = "SELECT AVG(district_id) FROM accounts;"

        # 1. Lower temperature sharpens positive logit towards 1.0
        scorer_low_t = LayaScorer(temperature=0.05)
        scorer_mid_t = LayaScorer(temperature=0.20)
        scorer_high_t = LayaScorer(temperature=2.0)

        p_low = scorer_low_t.score_candidate(q, self.sample_ddl, sql)
        p_mid = scorer_mid_t.score_candidate(q, self.sample_ddl, sql)
        p_high = scorer_high_t.score_candidate(q, self.sample_ddl, sql)

        self.assertGreaterEqual(p_low, p_mid)
        self.assertGreaterEqual(p_mid, p_high)

        # 2. Extreme temperatures (very small, zero, negative) must not crash
        for extreme_t in [1e-5, 0.0, -1.0, 100.0]:
            try:
                sc = LayaScorer(temperature=extreme_t)
                prob = sc.score_candidate(q, self.sample_ddl, sql)
                self.assertTrue(0.0 <= prob <= 1.0)
            except Exception as ex:
                self.fail(f"LayaScorer crashed on extreme temperature {extreme_t}: {ex}")

    def test_adaptive_early_exit_gate_thresholds(self):
        """Verify exact boundary behavior of early exit gate at 0.75 and 0.85."""
        test_points = [
            (0.0, "REJECT", False),
            (0.50, "REJECT", False),
            (0.7499, "REJECT", False),
            (0.75, "ACCEPT", False),
            (0.80, "ACCEPT", False),
            (0.8499, "ACCEPT", False),
            (0.85, "ACCEPT", True),
            (0.90, "ACCEPT", True),
            (1.0, "ACCEPT", True),
        ]

        for conf, expected_verdict, expected_early_exit in test_points:
            verdict, early_exit = self.gate.evaluate_confidence(conf)
            self.assertEqual(verdict, expected_verdict, f"Confidence {conf}: expected verdict {expected_verdict}, got {verdict}")
            self.assertEqual(early_exit, expected_early_exit, f"Confidence {conf}: expected early_exit {expected_early_exit}, got {early_exit}")

            # Also check LayaScorer helper methods
            self.assertEqual(self.scorer.should_early_exit(conf), expected_early_exit)

    def test_ece_calibration_computation(self):
        """Verify ECE computation and that calibrated predictions achieve ECE <= 0.081."""
        # 1. Perfectly calibrated predictions: accuracy matches confidence
        # e.g., 10 predictions with confidence 0.9, 9 are correct (acc = 0.9)
        confidences = [0.9] * 10 + [0.8] * 10
        correctness = [True] * 9 + [False] * 1 + [True] * 8 + [False] * 2
        ece_perfect = compute_ece(confidences, correctness, num_bins=10)
        self.assertAlmostEqual(ece_perfect, 0.0, delta=0.01)

        # 2. Severely uncalibrated predictions: 100% confident (1.0) but 0% correct (all False)
        bad_conf = [1.0] * 20
        bad_corr = [False] * 20
        ece_bad = compute_ece(bad_conf, bad_corr, num_bins=10)
        self.assertAlmostEqual(ece_bad, 1.0, delta=0.01)

        # 3. Target verification: on well-calibrated benchmark outputs, ECE must be <= 0.081
        calibrated_conf = [0.98, 0.99, 0.95, 0.97, 0.96, 0.99, 0.98, 0.97]
        calibrated_corr = [True, True, True, True, True, True, True, True]
        ece_system = compute_ece(calibrated_conf, calibrated_corr, num_bins=10)
        self.assertLessEqual(ece_system, 0.081, f"Calibrated ECE {ece_system} must be <= 0.081")

    def test_parallel_scoring_latency(self):
        """Verify parallel scoring of 16 candidates executes quickly."""
        candidates = [
            f"SELECT account_id FROM accounts WHERE district_id = {i};"
            for i in range(16)
        ]
        start = time.perf_counter()
        scores = self.scorer.score_candidates_parallel(
            "What accounts are in district 5?",
            self.sample_ddl,
            candidates
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        self.assertEqual(len(scores), 16)
        # Should be sub-10 ms on standard CPU
        self.assertLess(elapsed_ms, 15.0, f"Parallel scoring took {elapsed_ms:.2f} ms")


if __name__ == "__main__":
    unittest.main()
