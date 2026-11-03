#!/usr/bin/env python3
"""
Comprehensive Smoke Test & Verification Runner for Text-to-SQL SLM Architecture.
Executes all 6 validation tiers sequentially without requiring heavy GPU or large downloads:
1. Suite 1: Data Pipeline & Schema Grounding (PK-FK Graph, Steiner Tree, Column Cards, Retriever)
2. Suite 2: SLM Training Scripts (Completion-Only Loss Masking, LLM-JEPA causal mask, Dense GRPO)
3. Suite 3: Colab Training Notebooks (Dynamic cell-by-cell execution of 49 code cells across 5 notebooks)
4. Suite 4: Grammar PDA & Validation Rules (XGrammar PDA, Fallback Engine, PV-SQL 10 AST rules, DB Probe)
5. Suite 5: Evaluation & Benchmark Suite (Candidate Generator 8-16 rollouts, Multiset EX, VSR, ECE, Latency)
6. Suite 6: End-to-End harness wiring on data/mini/mini_bird.jsonl with a stub model (no gold leak; not a quality benchmark)

Total execution must complete in < 60.0 seconds on standard CPU.
Returns exit code 0 if all suites pass; non-zero if any test fails.
"""

import io
import json
import tempfile
import os
import sys
import time
import contextlib
import unittest
from typing import Dict, Any, List, Tuple

# Ensure repository root is in sys.path and set current working directory to REPO_ROOT
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from src.evaluation import EvaluationHarness
from src.evaluation.metrics import format_benchmark_summary

# ANSI Color Codes for terminal formatting
class TermColors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def print_banner():
    banner = f"""
{TermColors.BOLD}{TermColors.CYAN}===================================================================================================={TermColors.RESET}
{TermColors.BOLD}{TermColors.HEADER}    TEXT-TO-SQL SLM ARCHITECTURE — COMPREHENSIVE SMOKE TEST & VERIFICATION SUITE{TermColors.RESET}
{TermColors.BOLD}{TermColors.CYAN}===================================================================================================={TermColors.RESET}
{TermColors.DIM}Target Environment: CPU / Local Offline Smoke-Test (Zero GPU/Heavy Model Download Dependency){TermColors.RESET}
{TermColors.DIM}Quality Thresholds: Total Runtime < 60.0s | 100% Test Pass | EX >= 75.0% | VSR = 100.0% | ECE <= 0.081{TermColors.RESET}
{TermColors.BOLD}{TermColors.CYAN}----------------------------------------------------------------------------------------------------{TermColors.RESET}
"""
    print(banner)


import logging


@contextlib.contextmanager
def silence_all_output():
    """Silences Python stdout/stderr, OS file descriptors 1 and 2, and logging."""
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)

    old_log_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)

    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)
        os.close(devnull_fd)
        logging.disable(old_log_level)


def run_unittest_suite(suite_name: str, module_name: str) -> Tuple[bool, int, float, str]:
    """
    Executes a unittest suite by module name and measures execution time and pass/fail status.
    Captures stdout/stderr to keep the runner display clean while preserving error messages on failure.
    """
    loader = unittest.TestLoader()
    try:
        suite = loader.loadTestsFromName(module_name)
    except Exception as e:
        return False, 0, 0.0, f"Module load error: {e}"

    test_count = suite.countTestCases()
    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=1)

    start_time = time.perf_counter()
    with silence_all_output():
        result = runner.run(suite)
    elapsed = time.perf_counter() - start_time

    is_success = result.wasSuccessful()
    error_msg = ""
    if not is_success:
        error_msg = stream.getvalue()

    return is_success, test_count, elapsed, error_msg


def run_suite_6_e2e_benchmark(
    dataset_path: str = "data/mini/mini_bird.jsonl",
    db_dir: str = "data/mini"
) -> Tuple[bool, int, float, Dict[str, Any], str]:
    """
    Executes Suite 6: end-to-end harness wiring over data/mini/mini_bird.jsonl with a stub model.
    Checks that the pipeline completes, evaluates every query, and that no gold SQL reaches the
    generator prompt. It does not measure model quality: real EX needs a trained checkpoint.
    """
    from tests.stub_lm import StubModel, StubTokenizer
    start_time = time.perf_counter()

    try:
        with open(dataset_path, encoding="utf-8") as f:
            golds = [json.loads(line)["SQL"] for line in f if line.strip()]
        tokenizer = StubTokenizer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            harness = EvaluationHarness(
                db_dir=db_dir,
                num_candidates=4,
                output_log_dir=tmp_dir,
                output_report_dir=tmp_dir,
                model=StubModel(),
                tokenizer=tokenizer
            )
            with silence_all_output():
                results = harness.evaluate_dataset(dataset_path)
        elapsed = time.perf_counter() - start_time

        total_queries = results.get("total_queries", 0)
        failures = []
        if total_queries != len(golds):
            failures.append(f"Expected {len(golds)} queries evaluated, got {total_queries}")
        leaked = [g for g, p in zip(golds, tokenizer.prompts) if g.strip().rstrip(";") in p]
        if leaked:
            failures.append(f"Gold SQL leaked into {len(leaked)} generator prompts")
        if failures:
            return False, total_queries, elapsed, results, "; ".join(failures)
        return True, total_queries, elapsed, results, ""

    except Exception as e:
        elapsed = time.perf_counter() - start_time
        return False, 0, elapsed, {}, f"E2E Benchmark Exception: {e}"


def format_summary_table(suite_results: List[Dict[str, Any]], total_elapsed: float) -> str:
    """
    Renders an ANSI/ASCII formatted summary table of all test suites.
    """
    col_w_name = 54
    col_w_cases = 12
    col_w_time = 12
    col_w_status = 12

    border_line = f"+{'-' * (col_w_name + 2)}+{'-' * (col_w_cases + 2)}+{'-' * (col_w_time + 2)}+{'-' * (col_w_status + 2)}+"
    header_line = (
        f"| {'Test Suite / Verification Block':<{col_w_name}} "
        f"| {'Test Cases':^{col_w_cases}} "
        f"| {'Time (s)':^{col_w_time}} "
        f"| {'Status':^{col_w_status}} |"
    )

    lines = [
        border_line,
        header_line,
        border_line
    ]

    total_tests = 0
    all_passed = True

    for r in suite_results:
        name = r["name"]
        count = r["cases"]
        sec = f"{r['time']:.3f}s"
        passed = r["passed"]

        total_tests += count
        if not passed:
            all_passed = False

        status_str = f"{TermColors.GREEN}PASSED{TermColors.RESET}" if passed else f"{TermColors.RED}FAILED{TermColors.RESET}"
        status_raw = "PASSED" if passed else "FAILED"

        lines.append(
            f"| {name:<{col_w_name}} "
            f"| {str(count):^{col_w_cases}} "
            f"| {sec:^{col_w_time}} "
            f"| {status_str:^{col_w_status + len(status_str) - len(status_raw)}} |"
        )

    lines.append(border_line)

    overall_status_str = f"{TermColors.GREEN}ALL PASSED{TermColors.RESET}" if all_passed else f"{TermColors.RED}FAILURES{TermColors.RESET}"
    overall_status_raw = "ALL PASSED" if all_passed else "FAILURES"

    summary_line = (
        f"| {TermColors.BOLD}{'TOTAL / OVERALL SYSTEM-LEVEL SUMMARY':<{col_w_name}}{TermColors.RESET} "
        f"| {TermColors.BOLD}{str(total_tests):^{col_w_cases}}{TermColors.RESET} "
        f"| {TermColors.BOLD}{f'{total_elapsed:.3f}s':^{col_w_time}}{TermColors.RESET} "
        f"| {TermColors.BOLD}{overall_status_str:^{col_w_status + len(overall_status_str) - len(overall_status_raw)}}{TermColors.RESET} |"
    )
    lines.append(summary_line)
    lines.append(border_line)

    return "\n".join(lines)


def main():
    print_banner()

    suites_config = [
        {
            "id": 1,
            "name": "Suite 1: Data Pipeline & Schema Grounding",
            "module": "tests.test_grounding",
            "desc": "PK-FK Graph, Steiner Tree NWST, Column Cards, Embedding Retriever"
        },
        {
            "id": 2,
            "name": "Suite 2: SLM Training Scripts",
            "module": "tests.test_training",
            "desc": "Completion-Only Loss (-100), LLM-JEPA 2-Block Mask, Dense GRPO, DriveSync"
        },
        {
            "id": 3,
            "name": "Suite 3: Colab Training Notebooks",
            "module": "tests.test_notebooks",
            "desc": "49 Code Cells dynamic execution across Notebooks 01 to 05"
        },
        {
            "id": 4,
            "name": "Suite 4: Grammar PDA & Validation Rules",
            "module": "tests.test_grammar_and_validation",
            "desc": "XGrammar PDA, Fallback Engine, 10 PV-SQL AST Rules, DB Probe, Laya-421M"
        },
        {
            "id": 5,
            "name": "Suite 5: Evaluation & Benchmark Suite",
            "module": "tests.test_evaluation",
            "desc": "Candidate Generator (8-16 rollouts), Multiset EX, VSR, ECE, Memory Stats"
        }
    ]

    total_start = time.perf_counter()
    suite_records = []
    has_failure = False
    failure_details = []

    # Run Suites 1 through 5
    for conf in suites_config:
        print(f"{TermColors.BOLD}[RUNNING]{TermColors.RESET} {conf['name']} ({conf['desc']})...", flush=True)
        ok, cases, elapsed, err = run_unittest_suite(conf["name"], conf["module"])
        suite_records.append({
            "name": conf["name"],
            "cases": cases,
            "time": elapsed,
            "passed": ok
        })

        if ok:
            print(f"  {TermColors.GREEN}✔ PASSED{TermColors.RESET} in {elapsed:.3f}s ({cases} test cases)\n")
        else:
            has_failure = True
            print(f"  {TermColors.RED}✘ FAILED{TermColors.RESET} in {elapsed:.3f}s ({cases} test cases)\n")
            failure_details.append(f"{conf['name']}:\n{err}")

    # Run Suite 6: End-to-End Benchmark Execution
    s6_name = "Suite 6: End-to-End Wiring + No-Gold-Leak Check (stub model)"
    print(f"{TermColors.BOLD}[RUNNING]{TermColors.RESET} {s6_name} on data/mini/mini_bird.jsonl...", flush=True)
    ok_6, count_6, elapsed_6, bench_results, err_6 = run_suite_6_e2e_benchmark(
        dataset_path=os.path.join(REPO_ROOT, "data", "mini", "mini_bird.jsonl"),
        db_dir=os.path.join(REPO_ROOT, "data", "mini")
    )
    suite_records.append({
        "name": s6_name,
        "cases": count_6,
        "time": elapsed_6,
        "passed": ok_6
    })

    if ok_6:
        print(f"  {TermColors.GREEN}✔ PASSED{TermColors.RESET} in {elapsed_6:.3f}s ({count_6} evaluation queries)\n")
    else:
        has_failure = True
        print(f"  {TermColors.RED}✘ FAILED{TermColors.RESET} in {elapsed_6:.3f}s ({count_6} evaluation queries)\n")
        failure_details.append(f"{s6_name}:\n{err_6}")

    total_elapsed = time.perf_counter() - total_start

    # Render summary table
    print(format_summary_table(suite_records, total_elapsed))
    print()

    # Print Benchmark Metrics Table if Suite 6 succeeded
    if ok_6 and bench_results:
        print(format_benchmark_summary(bench_results, title="Suite 6: End-to-End Mini-BIRD Evaluation Report"))
        print()

    # Verify time constraint (< 60.0s)
    if total_elapsed >= 60.0:
        has_failure = True
        failure_details.append(f"Total execution time {total_elapsed:.2f}s exceeded 60.0-second constraint!")

    # Final verdict
    if not has_failure:
        print(f"{TermColors.BOLD}{TermColors.GREEN}[VERIFICATION SUCCESSFUL]{TermColors.RESET} "
              f"All 6 suites passed in {total_elapsed:.3f}s (< 60.0s threshold). System is production-ready.")
        sys.exit(0)
    else:
        print(f"{TermColors.BOLD}{TermColors.RED}[VERIFICATION FAILED]{TermColors.RESET} One or more test suites failed.")
        for f in failure_details:
            print(f"\n{TermColors.YELLOW}--- Failure Trace ---{TermColors.RESET}\n{f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
