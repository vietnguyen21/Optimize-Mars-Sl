#!/usr/bin/env python3
"""
Master Adversarial Verification & Stress Test Suite Runner for Milestone M5.
Consolidates all empirical challenger tests:
1. Smoke Test Runner Stress Suite (14 tests)
2. PV-SQL Refinements & Boundary Suite (21 tests)
3. Model Path CLI & Fallback Suite (6 tests)
Total: 41 adversarial test cases.
"""

import os
import sys
import time
import unittest

CHALLENGER_DIR = os.path.dirname(os.path.abspath(__file__))
if CHALLENGER_DIR not in sys.path:
    sys.path.insert(0, CHALLENGER_DIR)

from test_adversarial_smoke_runner import TestSmokeRunnerAdversarial
from test_adversarial_pv_sql_refinements import TestPVSQLAdversarialRefinements
from test_adversarial_model_path_cli import TestModelPathCLIAdversarial


def main():
    print("=" * 80)
    print("  EMPIRICAL CHALLENGER M5 — ADVERSARIAL VERIFICATION & STRESS TEST SUITE")
    print("=" * 80)
    print("Target: Milestone M5 (Automated Smoke Testing & Comprehensive Verification)")
    print("-" * 80)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestSmokeRunnerAdversarial))
    suite.addTests(loader.loadTestsFromTestCase(TestPVSQLAdversarialRefinements))
    suite.addTests(loader.loadTestsFromTestCase(TestModelPathCLIAdversarial))

    total_cases = suite.countTestCases()
    print(f"Loaded {total_cases} adversarial test cases across 3 specialized suites.\n")

    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.perf_counter()
    result = runner.run(suite)
    elapsed = time.perf_counter() - start_time

    print("\n" + "=" * 80)
    print(f"Adversarial Verification Complete in {elapsed:.2f} seconds.")
    print(f"Total Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Status: {'PASSED' if result.wasSuccessful() else 'FAILED'}")
    print("=" * 80)

    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
