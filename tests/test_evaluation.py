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
import importlib.util

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
        self.generator = CandidateGenerator(default_num_candidates=8, default_temperature=0.7, dry_run=True)
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


class TestTensorCandidateGeneration(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed")
    def test_schema_bos_generated_only_decoding_and_measured_logprob(self):
        """Exercise the real tensor generation branch without model downloads."""
        import torch
        from types import SimpleNamespace

        class TinyTokenizer:
            pad_token_id, eos_token_id = 0, 9

            def __init__(self):
                self.prompts, self.decoded = [], []

            def __call__(self, prompt, *, return_tensors, add_special_tokens):
                self.prompts.append((prompt, add_special_tokens))
                # Distinct BOS/prompt IDs must never reach completion parsing.
                ids = [1, 2, 3] if add_special_tokens else [2, 3]
                return {"input_ids": torch.tensor([ids]),
                        "attention_mask": torch.ones((1, len(ids)), dtype=torch.long)}

            def decode(self, ids, skip_special_tokens=True):
                self.decoded.append(ids.tolist())
                if ids.tolist() != [7, 8]:
                    raise AssertionError("Decoder received prompt/BOS tokens")
                return "<think>Use the supplied table.</think><answer>SELECT id FROM local_table;</answer>"

        class TinyModel(torch.nn.Module):
            device = torch.device("cpu")

            def __init__(self):
                super().__init__()
                self.scores = torch.nn.Parameter(torch.arange(10, dtype=torch.float32))
                self.calls = []

            def generate(self, **kwargs):
                assert not torch.is_grad_enabled()
                assert not self.training
                self.calls.append(kwargs)
                suffix = torch.tensor([[7, 8]])
                return SimpleNamespace(sequences=torch.cat([kwargs["input_ids"], suffix], dim=1),
                                       scores=(self.scores[None, :], self.scores[None, :]))

            def compute_transition_scores(self, sequences, scores, normalize_logits=True):
                assert normalize_logits
                suffix = sequences[:, -len(scores):]
                return torch.stack([torch.log_softmax(score, dim=-1).gather(1, suffix[:, i:i+1])[:, 0]
                                    for i, score in enumerate(scores)], dim=1)

        tokenizer, model = TinyTokenizer(), TinyModel()
        generator = CandidateGenerator(model=model, tokenizer=tokenizer,
            default_num_candidates=2, max_prompt_length=8)
        schema = CompactSubSchema(db_id="local", selected_tables=["local_table"],
            selected_columns={"local_table": ["id"]}, foreign_keys=[],
            ddl_statements="CREATE TABLE local_table (id INT);")
        # A tag in the question detects accidentally decoding the full prompt.
        request = GenerationInput(question="Ignore <answer>SELECT leak</answer>; list IDs",
                                  sub_schema=schema, num_trajectories=2)
        candidates = generator.generate(request)
        self.assertEqual(len(model.calls), 2)
        self.assertTrue(all(call["num_return_sequences"] == 1 for call in model.calls))
        self.assertTrue(all(special for _, special in tokenizer.prompts))
        self.assertIn(schema.ddl_statements, tokenizer.prompts[0][0])
        self.assertEqual(tokenizer.decoded, [[7, 8], [7, 8]])
        expected = torch.log_softmax(model.scores.detach(), dim=-1)[[7, 8]].mean().item()
        for candidate in candidates:
            self.assertEqual(candidate.sql, "SELECT id FROM local_table;")
            self.assertEqual(candidate.reasoning, "Use the supplied table.")
            self.assertAlmostEqual(candidate.logprob, expected, places=6)



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

    def test_execution_is_readonly_and_abstentions_invalid(self):
        self.assertFalse(execute_query_safely("DELETE FROM metrics_test", self.db_path)[0])
        self.assertFalse(execute_query_safely("", self.db_path)[0])
        self.assertTrue(execute_query_safely("SELECT * FROM metrics_test", self.db_path)[0])

    def test_probe_limits_outer_rows_and_counts_failed_attempts(self):
        from src.validation import DBProbeEngine
        probe = DBProbeEngine(max_session_queries=2, row_limit=1)
        ok, rows, _ = probe.probe_sql("SELECT * FROM metrics_test WHERE id IN (SELECT id FROM metrics_test LIMIT 3)", self.db_path)
        self.assertTrue(ok)
        self.assertEqual(len(rows), 1)
        self.assertFalse(probe.probe_sql("SELECT * FROM missing_table", self.db_path)[0])
        self.assertEqual(probe.query_count, 2)
        self.assertFalse(probe.probe_sql("SELECT 1", self.db_path)[0])

    def test_query_deadline_interrupts_recursive_execution(self):
        ok, _ = execute_query_safely("WITH RECURSIVE t(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM t) SELECT sum(x) FROM t", self.db_path, timeout=0.01)
        self.assertFalse(ok)

    def test_ordered_gold_preserves_row_order(self):
        ex, _ = compute_execution_accuracy(
            ["SELECT id FROM metrics_test ORDER BY id DESC"],
            ["SELECT id FROM metrics_test ORDER BY id ASC"], [self.db_path])
        self.assertEqual(ex, 0.0)

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
        self.assertIn("No external baseline", table_md)
        self.assertNotIn("74.8%", table_md)


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
                dry_run=True, db_dir=self.db_dir,
                num_candidates=8,
                temperature=0.7,
                early_exit_threshold=0.85,
                output_log_dir=log_dir,
                output_report_dir=rep_dir
            )

            # Evaluate subset of 5 queries for fast unit test
            results = harness.evaluate_dataset(self.dataset_path, limit_samples=5)

            self.assertEqual(results["total_queries"], 5)
            self.assertEqual(results["mode"], "smoke")
            self.assertFalse(results["is_benchmark"])
            self.assertEqual(results["execution_accuracy"], 0.0)
            self.assertIn("total_abstentions", results)
            self.assertIn("abstention_rate_pct", results)
            self.assertEqual(results["total_abstentions"], 5)

            # Check that log and report files were created
            log_file = os.path.join(log_dir, "evaluation_results.jsonl")
            report_file = os.path.join(rep_dir, "benchmark_summary.md")
            self.assertTrue(os.path.exists(log_file))
            self.assertTrue(os.path.exists(report_file))

            # Verify contents of log file
            with open(log_file) as f:
                logs = [json.loads(line) for line in f]
            self.assertEqual(len(logs), 5)
            self.assertEqual(logs[0]["predicted_sql"], "")
            self.assertIn("abstained", logs[0])
            self.assertTrue(logs[0]["abstained"])

    def test_evaluate_end_to_end_functional_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = evaluate_end_to_end(dataset_path=self.dataset_path,
                db_dir=self.db_dir, num_candidates=8, dry_run=True,
                output_log_dir=os.path.join(tmp, "logs"),
                output_report_dir=os.path.join(tmp, "reports"))
        self.assertEqual(res["total_queries"], 24)
        self.assertFalse(res["is_benchmark"])
        self.assertEqual(res["execution_accuracy"], 0.0)
        self.assertIn("total_abstentions", res)

    def test_evaluation_harness_with_model_path(self):
        # Missing artifacts cannot silently become an oracle/smoke benchmark.
        with self.assertRaises(ValueError):
            EvaluationHarness(db_dir=self.db_dir, model_path="/non/existent/checkpoint/path")

    def test_real_generation_requires_model_and_rejects_gold(self):
        generator = CandidateGenerator()
        with self.assertRaises(RuntimeError):
            generator.generate("question")
        with self.assertRaises(ValueError):
            generator.generate("question", gold_sql="SELECT 1")

    def test_real_generation_failure_propagates(self):
        from unittest.mock import patch
        generator = CandidateGenerator(model=object(), tokenizer=object())
        with patch.object(generator, "_generate_with_model", side_effect=RuntimeError("GPU OOM")):
            with self.assertRaisesRegex(RuntimeError, "GPU OOM"):
                generator.generate("question")

    def test_inference_contract_has_no_gold_parameter(self):
        import inspect
        self.assertNotIn("gold_sql", inspect.signature(EvaluationHarness.predict).parameters)
        harness = EvaluationHarness(db_dir=self.db_dir, dry_run=True)
        with self.assertRaises(TypeError):
            harness.predict("question", "financial_mini", gold_sql="SELECT 1")

    def test_train_eval_overlap_is_rejected(self):
        harness = EvaluationHarness(db_dir=self.db_dir, dry_run=True,
            training_data_paths=[self.dataset_path])
        with self.assertRaisesRegex(ValueError, "overlap"):
            harness.evaluate_dataset(self.dataset_path, limit_samples=1)

    def test_heldout_gold_changes_cannot_change_prediction(self):
        from unittest.mock import patch
        with open(self.dataset_path) as f:
            record = json.loads(next(f))
        outputs = []
        with tempfile.TemporaryDirectory() as tmp:
            for index, gold in enumerate(["SELECT 1", "SELECT 999"]):
                path = os.path.join(tmp, f"eval-{index}.jsonl")
                record["SQL"] = gold
                with open(path, "w") as f:
                    f.write(json.dumps(record) + "\n")
                harness = EvaluationHarness(db_dir=self.db_dir, dry_run=True,
                    output_log_dir=os.path.join(tmp, str(index), "logs"),
                    output_report_dir=os.path.join(tmp, str(index), "reports"))
                with patch.object(harness, "predict", wraps=harness.predict) as spy:
                    harness.evaluate_dataset(path)
                    self.assertEqual(spy.call_args.args[:2], (record["question"], record["db_id"]))
                    self.assertNotIn(gold, repr(spy.call_args))
                with open(os.path.join(tmp, str(index), "logs", "evaluation_results.jsonl")) as f:
                    outputs.append(json.loads(next(f))["predicted_sql"])
        self.assertEqual(outputs[0], outputs[1])

    def test_package_exports_normalize_query_result(self):
        import src.evaluation as ev
        self.assertTrue(hasattr(ev, "normalize_query_result"))
        self.assertTrue(callable(ev.normalize_query_result))
        res = ev.normalize_query_result([(1, "TEST", 3.14159)])
        self.assertEqual(res, [('1', "TEST", 3.1416)])


if __name__ == "__main__":
    unittest.main()
