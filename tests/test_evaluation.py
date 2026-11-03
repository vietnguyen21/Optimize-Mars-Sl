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
    build_prompt,
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
from tests.stub_lm import StubModel, StubTokenizer


class TestCandidateGenerator(unittest.TestCase):
    """Candidate generation from a loaded model; no gold SQL, no fabricated candidates."""

    def setUp(self):
        self.tokenizer = StubTokenizer()
        self.generator = CandidateGenerator(model=StubModel(), tokenizer=self.tokenizer, default_num_candidates=8)

    def test_no_model_raises(self):
        with self.assertRaises(RuntimeError):
            CandidateGenerator().generate("What is the total account balance?")

    def test_gold_sql_is_not_an_input(self):
        with self.assertRaises(TypeError):
            self.generator.generate("q", gold_sql="SELECT 1;")

    def test_generate_8_candidates_parses_answer(self):
        candidates = self.generator.generate("What is the total account balance?", num_candidates=8)
        self.assertEqual(len(candidates), 8)
        for idx, c in enumerate(candidates):
            self.assertIsInstance(c, SQLCandidate)
            self.assertEqual(c.candidate_id, idx)
            self.assertEqual(c.sql, "SELECT 1;")
            self.assertEqual(c.reasoning, "reasoning")

    def test_prompt_uses_schema_and_evidence(self):
        sub_schema = CompactSubSchema(
            db_id="financial_mini",
            selected_tables=["accounts"],
            selected_columns={"accounts": ["account_id", "balance"]},
            foreign_keys=[],
            ddl_statements="CREATE TABLE accounts (account_id INT, balance REAL);"
        )
        gen_input = GenerationInput(
            question="Find maximum balance",
            evidence="balance column in accounts",
            sub_schema=sub_schema,
            num_trajectories=12
        )
        candidates = self.generator.generate(gen_input)
        self.assertEqual(len(candidates), 12)
        prompt = self.tokenizer.prompts[-1]
        self.assertIn("CREATE TABLE accounts", prompt)
        self.assertIn("balance column in accounts", prompt)
        self.assertIn("Find maximum balance", prompt)
        self.assertEqual(prompt, build_prompt("Find maximum balance", "balance column in accounts",
                                              "CREATE TABLE accounts (account_id INT, balance REAL);"))

    def test_boundary_candidates_count(self):
        self.assertEqual(len(self.generator.generate("q", num_candidates=0)), 8)  # 0 -> default
        self.assertEqual(len(self.generator.generate("q", num_candidates=24)), 24)
        self.assertEqual(len(self.generator.generate("q", num_candidates=99)), 32)


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

        # BIRD official rule: raw set equality, no float rounding; duplicates ignored
        self.assertFalse(compare_sql_results([(10.50004,)], [(10.50001,)]))
        self.assertTrue(compare_sql_results([(1,), (1,)], [(1,)]))
        self.assertTrue(compare_sql_results([(1,)], [(1.0,)]))

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

    @unittest.skipIf(os.name == "nt", "RAM is measured with the Unix-only resource module")
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
        # Only measured values: no hard-coded baselines or targets
        self.assertNotIn("MARS-SQL", table_md)
        self.assertNotIn("Combo", table_md)

    def test_query_timeout(self):
        slow = ("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) "
                "SELECT COUNT(*) FROM c;")
        ok, err = execute_query_safely(slow, self.db_path, timeout=0.5)
        self.assertFalse(ok)
        self.assertIn("interrupt", err.lower())


class TestEvaluationHarness(unittest.TestCase):
    """End-to-end harness wiring on data/mini with a stub model that cannot see gold SQL."""

    def setUp(self):
        self.dataset_path = "data/mini/mini_bird.jsonl"
        self.db_dir = "data/mini"

    def _harness(self, tmp_dir, tokenizer):
        return EvaluationHarness(
            db_dir=self.db_dir,
            num_candidates=4,
            output_log_dir=os.path.join(tmp_dir, "logs"),
            output_report_dir=os.path.join(tmp_dir, "reports"),
            model=StubModel(),
            tokenizer=tokenizer
        )

    def test_harness_runs_without_gold_leak(self):
        with open(self.dataset_path, encoding="utf-8") as f:
            golds = [json.loads(line)["SQL"] for line in f if line.strip()][:5]
        tokenizer = StubTokenizer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            results = self._harness(tmp_dir, tokenizer).evaluate_dataset(self.dataset_path, limit_samples=5)

            self.assertEqual(results["total_queries"], 5)
            self.assertEqual(results["scorer_type"], "heuristic")
            # The stub always answers "SELECT 1;", which matches no gold query.
            self.assertEqual(results["execution_accuracy"], 0.0)
            self.assertEqual(len(tokenizer.prompts), 5)
            for prompt, gold in zip(tokenizer.prompts, golds):
                self.assertNotIn(gold.strip().rstrip(";"), prompt)
                self.assertIn("CREATE TABLE", prompt)

            with open(os.path.join(tmp_dir, "logs", "evaluation_results.jsonl")) as f:
                logs = [json.loads(line) for line in f]
            self.assertEqual(len(logs), 5)
            self.assertEqual(logs[0]["predicted_sql"], "SELECT 1;")
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "reports", "benchmark_summary.md")))

    def test_harness_without_model_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            harness = EvaluationHarness(db_dir=self.db_dir, output_log_dir=tmp_dir, output_report_dir=tmp_dir)
            with self.assertRaises(RuntimeError):
                harness.evaluate_dataset(self.dataset_path, limit_samples=1)

    def test_bad_model_path_fails(self):
        with self.assertRaises(Exception):
            EvaluationHarness(db_dir=self.db_dir, model_path="/non/existent/checkpoint/path")

    def test_package_exports_normalize_query_result(self):
        import src.evaluation as ev
        self.assertTrue(hasattr(ev, "normalize_query_result"))
        self.assertTrue(callable(ev.normalize_query_result))
        res = ev.normalize_query_result([(1, "TEST", 3.14159)])
        self.assertEqual(res, [('1', "TEST", 3.1416)])


if __name__ == "__main__":
    unittest.main()
