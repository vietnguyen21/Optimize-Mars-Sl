#!/usr/bin/env python3
"""
Adversarial Test Suite for evaluate_end_to_end.py CLI and model_path Handling
Milestone M5 - Empirical Challenger

Verifies:
1. CLI argument parsing: --model_path is exposed in --help and accepts custom strings.
2. Graceful fallback when model_path does not exist on disk (non_existent_model).
3. Graceful fallback when model_path points to an existing directory that lacks HF model weights.
4. Python API EvaluationHarness and evaluate_end_to_end handling with various model_path values.
"""

import os
import sys
import tempfile
import subprocess
import unittest

# Add repository root to sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.evaluation.evaluate_end_to_end import EvaluationHarness, evaluate_end_to_end


class TestModelPathCLIAdversarial(unittest.TestCase):

    def setUp(self):
        self.script_path = os.path.join(REPO_ROOT, "src", "evaluation", "evaluate_end_to_end.py")
        self.dataset_path = os.path.join(REPO_ROOT, "data", "mini", "mini_bird.jsonl")
        self.db_dir = os.path.join(REPO_ROOT, "data", "mini")

    def test_cli_help_displays_model_path(self):
        """CLI --help must document the --model_path argument."""
        cmd = [sys.executable, self.script_path, "--help"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("--model_path", res.stdout)
        self.assertIn("Path to fine-tuned model checkpoint", res.stdout)

    def test_cli_execution_with_non_existent_model_path(self):
        """CLI with --model_path pointing to non-existent path runs safely and exits with 0."""
        cmd = [
            sys.executable,
            self.script_path,
            "--dataset", self.dataset_path,
            "--db_dir", self.db_dir,
            "--limit", "2",
            "--model_path", "non_existent_model_path_adversarial"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"CLI crashed: {res.stderr}")
        self.assertIn("Benchmark Evaluation Summary", res.stdout)
        self.assertIn("Execution Accuracy (EX)", res.stdout)
        self.assertIn("Valid SQL Rate (VSR)", res.stdout)

    def test_cli_execution_with_corrupt_empty_model_dir(self):
        """CLI with --model_path pointing to an empty existing directory triggers safe fallback."""
        with tempfile.TemporaryDirectory() as empty_dir:
            cmd = [
                sys.executable,
                self.script_path,
                "--dataset", self.dataset_path,
                "--db_dir", self.db_dir,
                "--limit", "2",
                "--model_path", empty_dir
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"CLI crashed on empty dir: {res.stderr}")
            self.assertIn("Benchmark Evaluation Summary", res.stdout)

    def test_python_api_harness_with_non_existent_model(self):
        """Python EvaluationHarness handles non-existent model path without throwing."""
        harness = EvaluationHarness(
            db_dir=self.db_dir,
            num_candidates=4,
            model_path="/path/that/does/not/exist"
        )
        self.assertEqual(harness.model_path, "/path/that/does/not/exist")
        self.assertIsNone(harness.candidate_generator.model)

        # Execute evaluation on 2 samples
        results = harness.evaluate_dataset(self.dataset_path, limit_samples=2)
        self.assertEqual(results["total_queries"], 2)
        self.assertGreaterEqual(results["execution_accuracy"], 75.0)

    def test_python_api_harness_with_empty_directory_fallback(self):
        """Python EvaluationHarness handles existing directory without HF checkpoint gracefully."""
        with tempfile.TemporaryDirectory() as empty_dir:
            harness = EvaluationHarness(
                db_dir=self.db_dir,
                num_candidates=4,
                model_path=empty_dir
            )
            # Model should have failed to load and fallen back to None
            self.assertIsNone(harness.candidate_generator.model)
            results = harness.evaluate_dataset(self.dataset_path, limit_samples=2)
            self.assertEqual(results["total_queries"], 2)

    def test_convenience_function_with_model_path(self):
        """evaluate_end_to_end() convenience function passes model_path through."""
        results = evaluate_end_to_end(
            dataset_path=self.dataset_path,
            db_dir=self.db_dir,
            num_candidates=4,
            model_path="non_existent_mock_model"
        )
        self.assertEqual(results["total_queries"], 24)
        self.assertIn("execution_accuracy", results)


if __name__ == "__main__":
    unittest.main()
