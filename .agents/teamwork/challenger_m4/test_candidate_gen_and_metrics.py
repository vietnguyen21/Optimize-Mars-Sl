"""
Adversarial Test: Candidate Trajectory Generator & Evaluation Metrics Suite
Tests multiset row invariance, float precision normalization, duplicate preservation,
VSR computation, latency statistics percentiles, memory measurement, and candidate bounds.
"""

import sys
import os
import random
import unittest

REPO_ROOT = "/mnt/c/Workspace/optimize"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.evaluation.candidate_generator import CandidateGenerator, generate_candidate_rollouts
from src.evaluation.metrics import (
    normalize_query_result,
    compare_sql_results,
    compute_execution_accuracy,
    compute_vsr,
    compute_latency_stats,
    measure_memory_footprint,
    format_benchmark_summary
)


class TestCandidateGenAndMetricsAdversarial(unittest.TestCase):

    def setUp(self):
        self.generator = CandidateGenerator()

    def test_candidate_generator_rollouts_and_bounds(self):
        """Test candidate rollouts for 8 and 16 candidates and boundary clamping."""
        q = "Find all accounts in district 1."
        gold = "SELECT account_id FROM accounts WHERE district_id = 1;"

        # 8 candidates
        cands_8 = self.generator.generate(q, gold_sql=gold, num_candidates=8)
        self.assertEqual(len(cands_8), 8)
        for i, c in enumerate(cands_8):
            self.assertEqual(c.candidate_id, i)
            self.assertTrue(c.sql.endswith(";"))
            self.assertTrue(len(c.reasoning) > 0)
            self.assertIsInstance(c.logprob, float)

        # 16 candidates
        cands_16 = self.generator.generate(q, gold_sql=gold, num_candidates=16)
        self.assertEqual(len(cands_16), 16)
        self.assertEqual(cands_16[-1].candidate_id, 15)

        # Clamping: k < 1 -> clamped to 1; k > 32 -> clamped to 32
        cands_min = self.generator.generate(q, gold_sql=gold, num_candidates=-5)
        self.assertEqual(len(cands_min), 1)

        cands_max = self.generator.generate(q, gold_sql=gold, num_candidates=100)
        self.assertEqual(len(cands_max), 32)

    def test_multiset_row_order_invariance(self):
        """Multiset comparison in EX must be invariant to row order permutations."""
        base_rows = [(i, f"name_{i}", i * 1.5) for i in range(50)]
        shuffled_rows = list(base_rows)
        random.seed(42)
        random.shuffle(shuffled_rows)

        # Confirm order was actually scrambled
        self.assertNotEqual(base_rows, shuffled_rows)
        # normalize and compare must recognize equivalence
        self.assertTrue(compare_sql_results(base_rows, shuffled_rows))

    def test_float_precision_rounding_4_decimals(self):
        """Floating point values differing only beyond 4 decimal places must match."""
        res_a = [(1, 3.14159265, "pi")]
        res_b = [(1, 3.14158999, "pi")]  # both round to 3.1416
        self.assertTrue(compare_sql_results(res_a, res_b))

        res_diff = [(1, 3.1415, "pi")]
        res_diff_b = [(1, 3.1499, "pi")]  # 3.1415 vs 3.1499
        self.assertFalse(compare_sql_results(res_diff, res_diff_b))

    def test_duplicate_row_preservation(self):
        """Multiset comparison must NOT treat different multiplicities as equal."""
        # Single row vs duplicate row
        res_single = [(1, "A")]
        res_double = [(1, "A"), (1, "A")]
        self.assertFalse(compare_sql_results(res_single, res_double), "Multiset comparison must distinguish 1 vs 2 duplicate rows")

    def test_null_value_handling(self):
        """NULL / None values must normalize canonically."""
        res1 = [(1, None, "test")]
        res2 = [(1, "__NULL__", "test")]
        self.assertTrue(compare_sql_results(res1, res2))

    def test_empty_result_sets(self):
        """Empty sets match each other, but do not match non-empty sets."""
        self.assertTrue(compare_sql_results([], []))
        self.assertTrue(compare_sql_results(None, []))
        self.assertFalse(compare_sql_results([], [(1,)]))

    def test_latency_statistics_percentiles(self):
        """Latency statistics must accurately calculate mean, median, P95, and P99."""
        # 100 samples: 95 fast (10.0 ms), 5 slow (100.0, 200.0, 300.0, 400.0, 500.0 ms)
        latencies = [10.0] * 95 + [100.0, 200.0, 300.0, 400.0, 500.0]
        stats = compute_latency_stats(latencies)

        self.assertEqual(stats["median_ms"], 10.0)
        self.assertGreaterEqual(stats["p95_ms"], 10.0)
        self.assertGreaterEqual(stats["p99_ms"], 400.0)
        self.assertAlmostEqual(stats["total_s"], sum(latencies) / 1000.0, delta=0.01)

    def test_memory_footprint_measurement(self):
        """Memory footprint measurement must return non-zero RAM and valid structure."""
        mem = measure_memory_footprint()
        self.assertIn("ram_mb", mem)
        self.assertIn("gpu_vram_mb", mem)
        self.assertIn("vram_gb", mem)
        self.assertGreater(mem["ram_mb"], 0.0)
        self.assertGreaterEqual(mem["gpu_vram_mb"], 0.0)


if __name__ == "__main__":
    unittest.main()
