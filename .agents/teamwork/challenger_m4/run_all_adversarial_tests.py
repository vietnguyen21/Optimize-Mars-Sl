"""
Master Adversarial Verification Runner for Milestone M4
Discovers and executes all challenger test suites:
- test_stress_grammar_pda_fallback.py
- test_challenge_pv_sql_rules.py
- test_security_db_probe.py
- test_laya_scorer_and_gate.py
- test_candidate_gen_and_metrics.py
"""

import sys
import os
import time
import unittest

CHALLENGER_DIR = os.path.dirname(os.path.abspath(__file__))
if CHALLENGER_DIR not in sys.path:
    sys.path.insert(0, CHALLENGER_DIR)

from test_stress_grammar_pda_fallback import TestStressGrammarPDAAndFallback
from test_challenge_pv_sql_rules import TestChallengePVSQLRules
from test_security_db_probe import TestSecurityDBProbeAdversarial
from test_laya_scorer_and_gate import TestLayaScorerAndGateAdversarial
from test_candidate_gen_and_metrics import TestCandidateGenAndMetricsAdversarial


def run_master_adversarial_suite():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suites = [
        ("XGrammar PDA & Fallback Stress", loader.loadTestsFromTestCase(TestStressGrammarPDAAndFallback)),
        ("PV-SQL 10 AST Rules Challenge", loader.loadTestsFromTestCase(TestChallengePVSQLRules)),
        ("DB Probe Security & Injection", loader.loadTestsFromTestCase(TestSecurityDBProbeAdversarial)),
        ("Laya-421M Scorer & Early-Exit Gate", loader.loadTestsFromTestCase(TestLayaScorerAndGateAdversarial)),
        ("Candidate Generator & Metrics Suite", loader.loadTestsFromTestCase(TestCandidateGenAndMetricsAdversarial)),
    ]

    for name, s in suites:
        suite.addTests(s)

    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.perf_counter()
    result = runner.run(suite)
    elapsed = time.perf_counter() - start_time

    print("\n" + "=" * 70)
    print(f"ADVERSARIAL STRESS TEST SUITE EXECUTION SUMMARY")
    print(f"Total Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Elapsed Time: {elapsed:.4f} seconds")
    print("=" * 70)

    if result.wasSuccessful():
        print("RESULT: ALL ADVERSARIAL STRESS TESTS PASSED! (100% SUCCESS)")
        return 0
    else:
        print("RESULT: ADVERSARIAL TEST FAILURES DETECTED!")
        return 1


if __name__ == "__main__":
    sys.exit(run_master_adversarial_suite())
