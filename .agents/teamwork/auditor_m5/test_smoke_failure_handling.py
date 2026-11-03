import unittest
import sys
import os

REPO_ROOT = "/mnt/c/Workspace/optimize"
sys.path.insert(0, REPO_ROOT)

from scripts.run_smoke_tests import run_unittest_suite, run_suite_6_e2e_benchmark

def test_runner_catches_failure():
    ok, count, elapsed, err = run_unittest_suite("Mock Suite", "non_existent_module_for_testing")
    assert not ok, "Expected non-existent module to fail"
    assert "FAILED (errors=1)" in err or "ImportError" in err, f"Expected error in msg, got {err}"
    print("[TEST 1 PASSED] Non-existent module correctly caught, ok=False, error captured.")

def test_e2e_catches_failure():
    ok, count, elapsed, results, err = run_suite_6_e2e_benchmark(dataset_path="/non/existent/path.jsonl")
    assert not ok, "Expected non-existent dataset to fail"
    assert "E2E Benchmark Exception" in err, f"Expected exception in err, got {err}"
    print("[TEST 2 PASSED] E2E benchmark failure correctly caught and reported, ok=False.")

if __name__ == "__main__":
    test_runner_catches_failure()
    test_e2e_catches_failure()
    print("All adversarial failure handling tests passed!")
