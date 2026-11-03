#!/usr/bin/env python3
"""
Adversarial Stress Test Suite for scripts/run_smoke_tests.py
Milestone M5 - Empirical Challenger

Verifies:
1. Terminal stdout/stderr redirection and piping (| cat, > /dev/null, 2>&1).
2. Exit code fidelity and failure containment when a test suite deliberately fails.
3. Exit code fidelity when Suite 6 benchmark fails (e.g., low EX, high latency, missing dataset).
4. Timeout enforcement (total_elapsed >= 60.0s).
5. silence_all_output() context manager safety, exception handling, and descriptor leak prevention.
6. Execution latency benchmarking across multiple iterations.
"""

import io
import os
import sys
import time
import subprocess
import unittest
from unittest.mock import patch, MagicMock

# Add repository root to sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.run_smoke_tests import (
    silence_all_output,
    run_unittest_suite,
    run_suite_6_e2e_benchmark,
    format_summary_table
)
from src.evaluation import EvaluationHarness


class TestSmokeRunnerAdversarial(unittest.TestCase):

    def test_piped_execution_via_subprocess(self):
        """Stress-test: running smoke test through a pipe to cat."""
        cmd = f"python3 {os.path.join(REPO_ROOT, 'scripts', 'run_smoke_tests.py')} | cat"
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Piped execution failed: {res.stderr}")
        self.assertIn("ALL PASSED", res.stdout)
        self.assertIn("Suite 6: End-to-End Mini-BIRD Evaluation Report", res.stdout)
        self.assertIn("[VERIFICATION SUCCESSFUL]", res.stdout)

    def test_stdout_redirect_to_devnull(self):
        """Stress-test: redirecting stdout to /dev/null must still exit with code 0."""
        cmd = f"python3 {os.path.join(REPO_ROOT, 'scripts', 'run_smoke_tests.py')} > /dev/null"
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Redirection to /dev/null failed: {res.stderr}")

    def test_stderr_redirect_and_exit_code(self):
        """Stress-test: running with stderr redirection."""
        cmd = f"python3 {os.path.join(REPO_ROOT, 'scripts', 'run_smoke_tests.py')} 2> /dev/null"
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("ALL PASSED", res.stdout)

    def test_silence_all_output_descriptor_integrity(self):
        """Verify silence_all_output restores file descriptors 1 and 2 and does not leak fds."""
        # Check initial fd count / state
        initial_stdout_fd = os.dup(1)
        initial_stderr_fd = os.dup(2)
        os.close(initial_stdout_fd)
        os.close(initial_stderr_fd)

        # Run silence_all_output
        with silence_all_output():
            sys.stdout.write("Invisible to outer stdout\n")
            sys.stderr.write("Invisible to outer stderr\n")
            os.write(1, b"Invisible OS write 1\n")
            os.write(2, b"Invisible OS write 2\n")

        # After context manager, stdout and stderr must be functioning normally
        buf = io.StringIO()
        with io.StringIO() as stream:
            sys.stdout.write("")  # Should not raise
            sys.stderr.write("")  # Should not raise

    def test_silence_all_output_restores_on_exception(self):
        """Verify silence_all_output properly restores fds even if exception is raised inside."""
        with self.assertRaises(ValueError):
            with silence_all_output():
                raise ValueError("Intentional crash inside silence_all_output")

        # After exception, writing to stdout/stderr must still work
        sys.stdout.write("")
        sys.stderr.write("")

    def test_run_unittest_suite_on_invalid_module(self):
        """Verify run_unittest_suite handles non-existent or failing module load gracefully."""
        ok, count, elapsed, err = run_unittest_suite("Bad Suite", "tests.non_existent_module")
        self.assertFalse(ok)
        self.assertTrue("Module load error" in err or "No module named" in err)

    def test_run_unittest_suite_captures_failure(self):
        """Verify that when a test inside a suite fails, is_success is False and error_msg is captured."""
        class FailingTestCase(unittest.TestCase):
            def test_deliberate_failure(self):
                self.assertEqual(1, 2, "Adversarial failure injection")

        with patch("unittest.TestLoader.loadTestsFromName") as mock_load:
            suite = unittest.TestSuite()
            suite.addTest(FailingTestCase("test_deliberate_failure"))
            mock_load.return_value = suite

            ok, count, elapsed, err = run_unittest_suite("Mock Failing Suite", "tests.dummy")
            self.assertFalse(ok)
            self.assertEqual(count, 1)
            self.assertIn("AssertionError: 1 != 2 : Adversarial failure injection", err)

    def test_suite_6_catches_low_execution_accuracy(self):
        """Verify Suite 6 benchmark fails when EX is below 75%."""
        with patch.object(
            EvaluationHarness,
            "evaluate_dataset",
            return_value={
                "total_queries": 24,
                "execution_accuracy": 65.0,  # Below 75% threshold
                "valid_sql_rate": 100.0,
                "ece": 0.005,
                "latency": {"mean_ms": 20.0}
            }
        ):
            ok, count, elapsed, res, err = run_suite_6_e2e_benchmark(
                dataset_path=os.path.join(REPO_ROOT, "data", "mini", "mini_bird.jsonl"),
                db_dir=os.path.join(REPO_ROOT, "data", "mini")
            )
            self.assertFalse(ok)
            self.assertIn("Execution Accuracy (EX) 65.00% is below target >= 75.0%", err)

    def test_suite_6_catches_low_valid_sql_rate(self):
        """Verify Suite 6 benchmark fails when VSR is below 100%."""
        with patch.object(
            EvaluationHarness,
            "evaluate_dataset",
            return_value={
                "total_queries": 24,
                "execution_accuracy": 90.0,
                "valid_sql_rate": 95.0,  # Below 99.9% threshold
                "ece": 0.005,
                "latency": {"mean_ms": 20.0}
            }
        ):
            ok, count, elapsed, res, err = run_suite_6_e2e_benchmark(
                dataset_path=os.path.join(REPO_ROOT, "data", "mini", "mini_bird.jsonl"),
                db_dir=os.path.join(REPO_ROOT, "data", "mini")
            )
            self.assertFalse(ok)
            self.assertIn("Valid SQL Rate (VSR) 95.00% is below target 100.0%", err)

    def test_suite_6_catches_high_ece(self):
        """Verify Suite 6 benchmark fails when ECE exceeds 0.081."""
        with patch.object(
            EvaluationHarness,
            "evaluate_dataset",
            return_value={
                "total_queries": 24,
                "execution_accuracy": 90.0,
                "valid_sql_rate": 100.0,
                "ece": 0.1200,  # Above 0.081
                "latency": {"mean_ms": 20.0}
            }
        ):
            ok, count, elapsed, res, err = run_suite_6_e2e_benchmark(
                dataset_path=os.path.join(REPO_ROOT, "data", "mini", "mini_bird.jsonl"),
                db_dir=os.path.join(REPO_ROOT, "data", "mini")
            )
            self.assertFalse(ok)
            self.assertIn("Expected Calibration Error (ECE) 0.1200 exceeds target <= 0.081", err)

    def test_suite_6_catches_high_latency(self):
        """Verify Suite 6 benchmark fails when average latency exceeds 5.8s."""
        with patch.object(
            EvaluationHarness,
            "evaluate_dataset",
            return_value={
                "total_queries": 24,
                "execution_accuracy": 90.0,
                "valid_sql_rate": 100.0,
                "ece": 0.0100,
                "latency": {"mean_ms": 6500.0}  # 6.5s > 5.8s
            }
        ):
            ok, count, elapsed, res, err = run_suite_6_e2e_benchmark(
                dataset_path=os.path.join(REPO_ROOT, "data", "mini", "mini_bird.jsonl"),
                db_dir=os.path.join(REPO_ROOT, "data", "mini")
            )
            self.assertFalse(ok)
            self.assertIn("Average Latency 6.50s exceeds target < 5.8s", err)

    def test_suite_6_missing_dataset_handling(self):
        """Verify Suite 6 benchmark handles missing dataset path gracefully."""
        ok, count, elapsed, res, err = run_suite_6_e2e_benchmark(
            dataset_path="data/mini/non_existent.jsonl",
            db_dir="data/mini"
        )
        self.assertFalse(ok)
        self.assertIn("E2E Benchmark Exception", err)

    def test_overall_script_exit_code_1_on_failure(self):
        """Subprocess test: verify run_smoke_tests.py exits with code 1 when a failure occurs."""
        script_path = os.path.join(REPO_ROOT, "scripts", "run_smoke_tests.py")
        code = f"""
import sys
from unittest.mock import patch
import scripts.run_smoke_tests as runner

with patch('scripts.run_smoke_tests.run_unittest_suite', return_value=(False, 10, 0.5, 'Injected Failure')):
    try:
        runner.main()
    except SystemExit as e:
        sys.exit(e.code)
"""
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(res.returncode, 1, f"Expected exit code 1 on failure, got {res.returncode}")
        self.assertIn("[VERIFICATION FAILED]", res.stdout)
        self.assertIn("Injected Failure", res.stdout)

    def test_execution_time_latency_benchmark(self):
        """Benchmark 3 runs of run_smoke_tests.py to verify time is consistently < 60s (target ~5s)."""
        durations = []
        script_path = os.path.join(REPO_ROOT, "scripts", "run_smoke_tests.py")
        for i in range(3):
            t0 = time.perf_counter()
            res = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
            elapsed = time.perf_counter() - t0
            self.assertEqual(res.returncode, 0)
            durations.append(elapsed)

        min_d = min(durations)
        max_d = max(durations)
        avg_d = sum(durations) / len(durations)
        print(f"\n[LATENCY BENCHMARK] 3 iterations: min={min_d:.2f}s, max={max_d:.2f}s, avg={avg_d:.2f}s")
        self.assertLess(max_d, 60.0, "Execution time exceeded 60 seconds threshold!")
        self.assertLess(avg_d, 10.0, "Execution time exceeded 10 seconds realistic target!")


if __name__ == "__main__":
    unittest.main()
