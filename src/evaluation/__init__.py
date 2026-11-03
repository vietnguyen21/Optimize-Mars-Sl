"""
Evaluation and Benchmarking Package for Text-to-SQL SLM Architecture.
Exports CandidateGenerator, EvaluationHarness, evaluate_end_to_end,
and benchmark metrics computation functions (EX, VSR, ECE, latency).
"""

from src.evaluation.candidate_generator import CandidateGenerator, generate_candidate_rollouts
from src.evaluation.evaluate_end_to_end import EvaluationHarness, evaluate_end_to_end
from src.evaluation.metrics import (
    compute_execution_accuracy,
    compute_vsr,
    compute_ece,
    compute_latency_stats,
    measure_memory_footprint,
    format_benchmark_summary,
    compare_sql_results,
    normalize_query_result,
    execute_query_safely
)

__all__ = [
    "CandidateGenerator",
    "generate_candidate_rollouts",
    "EvaluationHarness",
    "evaluate_end_to_end",
    "compute_execution_accuracy",
    "compute_vsr",
    "compute_ece",
    "compute_latency_stats",
    "measure_memory_footprint",
    "format_benchmark_summary",
    "compare_sql_results",
    "normalize_query_result",
    "execute_query_safely"
]
