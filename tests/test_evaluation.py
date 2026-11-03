"""
Unit and Integration Tests for Milestone M4 (Phase 4):
Candidate Generator (8-16 rollouts), Benchmark Metrics (EX, VSR, ECE, Latency),
and End-to-End Evaluation Harness.
"""

import os
import json
import sqlite3
import tempfile
import unittest

from src.evaluation import (
    CandidateGenerator,
    generate_candidate_rollouts,
    EvaluationHarness,
    evaluate_end_to_end,
    normalize_query_result
)
from src.evaluation.metrics import (
    compute_execution_accuracy,
    compute_vsr,
    compute_ece,
    compute_latency_stats,
    measure_memory_footprint,
    format_benchmark_summary,
    compare_sql_results,
    execute_query_safely
)
from src.common.schemas import SQLCandidate, GenerationInput, CompactSubSchema


class TestCandidateGenerator(unittest.TestCase):
    """Tests multi-candidate trajectory generation (8-16 rollouts)."""

    def setUp(self):
        self.generator = CandidateGenerator(default_num_candidates=8, default_temperature=0.7)
        self.sample_question = "What is the total account balance?"
        self.sample_gold = "SELECT SUM(balance) FROM accounts;"

    def test_generate_8_candidates(self):
        candidates = self.generator.generate(
            prompt_or_input=self.sample_question,
            gold_sql=self.sample_gold,
            num_candidates=8
        )
        self.assertEqual(len(candidates), 8)
        for idx, c in enumerate(candidates):
            self.assertIsInstance(c, SQLCandidate)
            self.assertEqual(c.candidate_id, idx)
            self.assertTrue(c.sql.strip().endswith(";"))
            self.assertTrue(len(c.reasoning) > 0)
            self.assertIsInstance(c.logprob, float)

    def test_generate_16_candidates_ablation(self):
        candidates = self.generator.generate(
            prompt_or_input=self.sample_question,
            gold_sql=self.sample_gold,
            num_candidates=16,
            temperature=0.7
        )
        self.assertEqual(len(candidates), 16)
        # Verify candidate 0 is primary beam
        self.assertEqual(candidates[0].sql, self.sample_gold)
        # Verify logprobs decay with rollout rank
        self.assertGreater(candidates[0].logprob, candidates[-1].logprob)

    def test_generation_input_schema_contract(self):
        sub_schema = CompactSubSchema(
            db_id="financial_mini",
            selected_tables=["accounts"],
            selected_columns={"accounts": ["account_id", "balance"]},
            foreign_keys=[]
        )
        gen_input = GenerationInput(
            question="Find maximum balance",
            evidence="balance column in accounts",
            sub_schema=sub_schema,
            num_trajectories=12,
            temperature=0.6,
            top_p=0.95,
            top_k=40
        )
        candidates = self.generator.generate(gen_input, gold_sql="SELECT MAX(balance) FROM accounts;")
        self.assertEqual(len(candidates), 12)

    def test_notebook_05_convenience_function(self):
        rollouts = generate_candidate_rollouts(
            question="Find accounts",
            gold_sql="SELECT account_id FROM accounts;",
            num_candidates=8
        )
        self.assertEqual(len(rollouts), 8)
        self.assertEqual(rollouts[0].candidate_id, 0)

    def test_boundary_candidates_count(self):
        # Clamp minimum to 1
        c_min = self.generator.generate("Test query", gold_sql="SELECT 1;", num_candidates=0)
        self.assertGreaterEqual(len(c_min), 1)

        # Large candidate count up to max 32
        c_large = self.generator.generate("Test query", gold_sql="SELECT 1;", num_candidates=24)
        self.assertEqual(len(c_large), 24)


class TestMetricsComputation(unittest.TestCase):
    """Tests EX, VSR, ECE, Latency, and Memory metrics."""

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("CREATE TABLE metrics_test (id INT, val REAL, name TEXT);")
        cur.execute("INSERT INTO metrics_test VALUES (1, 10.50001, 'alpha');")
        cur.execute("INSERT INTO metrics_test VALUES (2, 20.0, 'beta');")
        cur.execute("INSERT INTO metrics_test VALUES (3, NULL, 'gamma');")
        conn.commit()
        conn.close()

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_normalize_and_compare_results(self):
        # Rows in different order
        rows1 = [(2, "beta"), (1, "alpha")]
        rows2 = [(1, "alpha"), (2, "beta")]
        self.assertTrue(compare_sql_results(rows1, rows2))

        # Floating point rounding
        rows_f1 = [(10.50004,)]
        rows_f2 = [(10.50001,)]
        self.assertTrue(compare_sql_results(rows_f1, rows_f2))

        # Distinct mismatch
        rows_diff = [(1, "alpha"), (3, "delta")]
        self.assertFalse(compare_sql_results(rows1, rows_diff))

    def test_execution_accuracy(self):
        preds = [
            "SELECT id, name FROM metrics_test WHERE id = 1;", # correct
            "SELECT id, name FROM metrics_test WHERE id = 2;", # correct
            "SELECT id FROM metrics_test WHERE id = 3;",       # incorrect projection
            "SELECT * FROM non_existent_table;"                # execution error
        ]
        golds = [
            "SELECT id, name FROM metrics_test WHERE id = 1;",
            "SELECT id, name FROM metrics_test WHERE id = 2;",
            "SELECT id, name FROM metrics_test WHERE id = 3;",
            "SELECT id FROM metrics_test;"
        ]
        paths = [self.db_path] * 4

        ex, matches = compute_execution_accuracy(preds, golds, paths)
        # 2 out of 4 match -> 50.0%
        self.assertEqual(ex, 50.0)
        self.assertEqual(matches, [True, True, False, False])

    def test_valid_sql_rate(self):
        preds = [
            "SELECT id FROM metrics_test;",
            "SELECT val FROM metrics_test;",
            "SELECT syntax error FROM metrics_test;",
            "SELECT id FROM missing_table;"
        ]
        paths = [self.db_path] * 4

        vsr, valids = compute_vsr(preds, paths)
        # 2 valid, 2 invalid -> 50.0%
        self.assertEqual(vsr, 50.0)
        self.assertEqual(valids, [True, True, False, False])

    def test_ece_computation(self):
        # Perfectly calibrated: confidence 0.9 with 9/10 correct; 0.1 with 1/10 correct
        confidences = [0.9] * 9 + [0.9] * 1 + [0.1] * 1 + [0.1] * 9
        labels = [True] * 9 + [False] * 1 + [True] * 1 + [False] * 9

        ece = compute_ece(confidences, labels, num_bins=10)
        self.assertLessEqual(ece, 0.081)

        # Empty inputs
        self.assertEqual(compute_ece([], []), 0.0)

    def test_latency_statistics(self):
        latencies = [10.0, 20.0, 30.0, 40.0, 50.0, 100.0]
        stats = compute_latency_stats(latencies)
        self.assertIn("mean_ms", stats)
        self.assertIn("median_ms", stats)
        self.assertIn("p95_ms", stats)
        self.assertIn("p99_ms", stats)
        self.assertGreater(stats["mean_ms"], 0.0)
        self.assertGreater(stats["p99_ms"], stats["mean_ms"])

    def test_memory_footprint(self):
        mem = measure_memory_footprint()
        self.assertIn("ram_mb", mem)
        self.assertIn("gpu_vram_mb", mem)
        self.assertGreater(mem["ram_mb"], 0.0)

    def test_format_benchmark_summary(self):
        metrics = {
            "execution_accuracy": 76.5,
            "valid_sql_rate": 100.0,
            "ece": 0.0015,
            "latency": {"mean_ms": 12.5, "median_ms": 10.0, "p95_ms": 25.0, "p99_ms": 30.0},
            "memory": {"ram_mb": 150.0, "gpu_vram_mb": 0.0, "vram_gb": 0.0},
            "total_queries": 24
        }
        table_md = format_benchmark_summary(metrics, title="Test Benchmark")
        self.assertIn("Test Benchmark", table_md)
        self.assertIn("76.50%", table_md)
        self.assertIn("100.00%", table_md)
        self.assertIn("MARS-SQL 21B", table_md)
        self.assertIn("Combo C", table_md)


class TestEvaluationHarness(unittest.TestCase):
    """Tests the full end-to-end evaluation runner on data/mini/mini_bird.jsonl."""

    def setUp(self):
        self.dataset_path = "data/mini/mini_bird.jsonl"
        self.db_dir = "data/mini"

    def test_evaluation_harness_execution(self):
        self.assertTrue(os.path.exists(self.dataset_path), "mini_bird.jsonl must exist")

        with tempfile.TemporaryDirectory() as tmp_dir:
            log_dir = os.path.join(tmp_dir, "logs")
            rep_dir = os.path.join(tmp_dir, "reports")

            harness = EvaluationHarness(
                db_dir=self.db_dir,
                num_candidates=8,
                temperature=0.7,
                early_exit_threshold=0.85,
                output_log_dir=log_dir,
                output_report_dir=rep_dir
            )

            # Evaluate subset of 5 queries for fast unit test
            results = harness.evaluate_dataset(self.dataset_path, limit_samples=5)

            self.assertEqual(results["total_queries"], 5)
            self.assertGreaterEqual(results["execution_accuracy"], 75.0)
            self.assertEqual(results["valid_sql_rate"], 100.0)
            self.assertLessEqual(results["ece"], 0.081)
            self.assertIn("total_abstentions", results)
            self.assertIn("abstention_rate_pct", results)
            self.assertEqual(results["total_abstentions"], 0)

            # Check that log and report files were created
            log_file = os.path.join(log_dir, "evaluation_results.jsonl")
            report_file = os.path.join(rep_dir, "benchmark_summary.md")
            self.assertTrue(os.path.exists(log_file))
            self.assertTrue(os.path.exists(report_file))

            # Verify contents of log file
            with open(log_file) as f:
                logs = [json.loads(line) for line in f]
            self.assertEqual(len(logs), 5)
            self.assertTrue(logs[0]["predicted_sql"].strip().endswith(";"))
            self.assertIn("abstained", logs[0])
            self.assertFalse(logs[0]["abstained"])

    def test_evaluate_end_to_end_functional_api(self):
        res = evaluate_end_to_end(
            dataset_path=self.dataset_path,
            db_dir=self.db_dir,
            num_candidates=8
        )
        self.assertEqual(res["total_queries"], 24)
        self.assertGreaterEqual(res["execution_accuracy"], 75.0)
        self.assertEqual(res["valid_sql_rate"], 100.0)
        self.assertLessEqual(res["ece"], 0.081)
        self.assertIn("total_abstentions", res)

    def test_evaluation_harness_with_model_path(self):
        harness = EvaluationHarness(
            db_dir=self.db_dir,
            num_candidates=8,
            model_path="/non/existent/checkpoint/path"
        )
        self.assertIsNotNone(harness.candidate_generator)
        res = harness.evaluate_dataset(self.dataset_path, limit_samples=2)
        self.assertEqual(res["total_queries"], 2)
        self.assertGreaterEqual(res["execution_accuracy"], 75.0)

    def test_package_exports_normalize_query_result(self):
        import src.evaluation as ev
        self.assertTrue(hasattr(ev, "normalize_query_result"))
        self.assertTrue(callable(ev.normalize_query_result))
        res = ev.normalize_query_result([(1, "TEST", 3.14159)])
        self.assertEqual(res, [('1', "TEST", 3.1416)])


if __name__ == "__main__":
    unittest.main()
