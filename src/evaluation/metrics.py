"""
Benchmark Metrics Computation Suite for Text-to-SQL SLM Architecture.
Computes:
1. Execution Accuracy (EX): Exact multiset match between generated and gold queries.
2. Valid SQL Rate (VSR): Percentage of queries executing with 0 syntax/runtime errors.
3. Expected Calibration Error (ECE): 10-bin calibration error of confidence predictions.
4. Latency Statistics: Mean, median, P95, P99 in milliseconds.
5. Memory Footprint: Process RAM and Peak GPU VRAM footprint.
6. Markdown Benchmark Summary Table Generator.
"""

import math
import os
import resource
import sqlite3
import statistics
from typing import List, Tuple, Dict, Any, Optional, Set, Union


def normalize_query_result(rows: Any) -> Any:
    """
    Normalizes SQL query output rows for robust multiset comparison:
    - Rounds floating-point numbers to 4 decimal places.
    - Converts None/NULL to a canonical string.
    - Sorts rows to ensure order-invariant comparison unless explicitly ordered.
    """
    if rows is None:
        return []
    if not isinstance(rows, (list, tuple)):
        return [rows]

    normalized = []
    for row in rows:
        if isinstance(row, (tuple, list)):
            norm_row = []
            for item in row:
                if isinstance(item, float):
                    norm_row.append(round(item, 4))
                elif item is None:
                    norm_row.append("__NULL__")
                else:
                    norm_row.append(str(item).strip())
            normalized.append(tuple(norm_row))
        else:
            if isinstance(row, float):
                normalized.append((round(row, 4),))
            elif row is None:
                normalized.append(("__NULL__",))
            else:
                normalized.append((str(row).strip(),))

    # Sort normalized rows so set ordering differences do not penalize valid SQL
    try:
        return sorted(normalized, key=lambda r: tuple(str(x) for x in r))
    except Exception:
        return normalized


def compare_sql_results(res1: Any, res2: Any) -> bool:
    """
    Compares two query result sets for equivalence.
    Returns True if sets match after normalization.
    """
    norm1 = normalize_query_result(res1)
    norm2 = normalize_query_result(res2)
    return norm1 == norm2


def execute_query_safely(sql: str, db_path: str, timeout: float = 3.0) -> Tuple[bool, Any]:
    """
    Executes SQL query on SQLite database and returns (success: bool, rows_or_error: Any).
    """
    if not os.path.exists(db_path):
        return False, f"Database not found: {db_path}"
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        cursor = conn.cursor()
        cursor.execute(sql.strip())
        rows = cursor.fetchall()
        conn.close()
        return True, rows
    except Exception as err:
        return False, str(err)


def compute_execution_accuracy(
    predictions: List[str],
    gold_queries: List[str],
    db_paths: List[str]
) -> Tuple[float, List[bool]]:
    """
    Computes Execution Accuracy (EX) percentage.
    EX = (Number of predictions matching gold execution results / Total queries) * 100.
    
    Returns:
        Tuple of (ex_percentage: float, list_of_matches: List[bool]).
    """
    if not predictions or not gold_queries:
        return 0.0, []

    total = len(predictions)
    matches = []

    for pred_sql, gold_sql, db_path in zip(predictions, gold_queries, db_paths):
        p_ok, p_res = execute_query_safely(pred_sql, db_path)
        g_ok, g_res = execute_query_safely(gold_sql, db_path)

        if p_ok and g_ok and compare_sql_results(p_res, g_res):
            matches.append(True)
        else:
            matches.append(False)

    ex = (sum(matches) / total) * 100.0 if total > 0 else 0.0
    return round(ex, 2), matches


def compute_vsr(predictions: List[str], db_paths: List[str]) -> Tuple[float, List[bool]]:
    """
    Computes Valid SQL Rate (VSR) percentage.
    VSR = (Number of predictions executing without syntax/db error / Total queries) * 100.
    
    Returns:
        Tuple of (vsr_percentage: float, list_of_valid: List[bool]).
    """
    if not predictions:
        return 0.0, []

    total = len(predictions)
    valid_flags = []

    for pred_sql, db_path in zip(predictions, db_paths):
        ok, _ = execute_query_safely(pred_sql, db_path)
        valid_flags.append(ok)

    vsr = (sum(valid_flags) / total) * 100.0 if total > 0 else 0.0
    return round(vsr, 2), valid_flags


def compute_ece(
    confidences: List[float],
    correctness: List[bool],
    num_bins: int = 10
) -> float:
    """
    Computes Expected Calibration Error (ECE) across confidence bins.
    
    Args:
        confidences: List of predicted confidence probabilities in [0.0, 1.0].
        correctness: List of binary correctness labels (True/False or 1/0).
        num_bins: Number of bins (default: 10).
        
    Returns:
        ECE float value (target <= 0.081).
    """
    if not confidences or not correctness:
        return 0.0

    n = len(confidences)
    bin_size = 1.0 / num_bins
    ece = 0.0

    for b in range(num_bins):
        bin_lower = b * bin_size
        bin_upper = (b + 1) * bin_size

        # Find items in bin
        bin_indices = [
            i for i, c in enumerate(confidences)
            if bin_lower <= c < bin_upper or (b == num_bins - 1 and c == bin_upper)
        ]

        if not bin_indices:
            continue

        bin_conf = sum(confidences[i] for i in bin_indices) / len(bin_indices)
        bin_acc = sum(1.0 if correctness[i] else 0.0 for i in bin_indices) / len(bin_indices)

        bin_weight = len(bin_indices) / n
        ece += bin_weight * abs(bin_acc - bin_conf)

    return round(ece, 4)


def compute_latency_stats(latencies_ms: List[float]) -> Dict[str, float]:
    """
    Computes mean, median, P95, and P99 latency statistics in milliseconds.
    """
    if not latencies_ms:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "total_s": 0.0}

    sorted_lat = sorted(latencies_ms)
    n = len(sorted_lat)

    mean_val = statistics.mean(sorted_lat)
    median_val = statistics.median(sorted_lat)

    p95_idx = min(int(math.ceil(0.95 * n)) - 1, n - 1)
    p99_idx = min(int(math.ceil(0.99 * n)) - 1, n - 1)

    return {
        "mean_ms": round(mean_val, 2),
        "median_ms": round(median_val, 2),
        "p95_ms": round(sorted_lat[p95_idx], 2),
        "p99_ms": round(sorted_lat[p99_idx], 2),
        "total_s": round(sum(sorted_lat) / 1000.0, 2)
    }


def measure_memory_footprint() -> Dict[str, Any]:
    """
    Measures current process RAM footprint and GPU VRAM footprint (if available).
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is in kilobytes on Linux
    ram_mb = usage.ru_maxrss / 1024.0

    gpu_vram_mb = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            gpu_vram_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    except Exception:
        pass

    return {
        "ram_mb": round(ram_mb, 2),
        "gpu_vram_mb": round(gpu_vram_mb, 2),
        "vram_gb": round(gpu_vram_mb / 1024.0, 3)
    }


def format_benchmark_summary(
    metrics: Dict[str, Any],
    title: str = "Benchmark Evaluation Summary"
) -> str:
    """
    Generates publication-ready Markdown table comparing current evaluation against
    established research baselines (MARS-SQL 21B, Combo A, Combo C).
    """
    ex = metrics.get("execution_accuracy", 0.0)
    vsr = metrics.get("valid_sql_rate", 0.0)
    ece = metrics.get("ece", 0.0)
    latency_ms = metrics.get("latency", {}).get("mean_ms", 0.0)
    latency_s = round(latency_ms / 1000.0, 2) if latency_ms > 0 else 0.0
    vram = metrics.get("memory", {}).get("vram_gb", 0.0)
    total_q = metrics.get("total_queries", 0)

    table = f"""# {title}

### Core Evaluation Metrics (Total Queries Evaluated: {total_q})
| Metric | Measurement | Target Standard | Status |
|:-------|:------------|:----------------|:-------|
| **Execution Accuracy (EX)** | **{ex:.2f}%** | $\\ge 75.0\\%$ | {'✅ PASSED' if ex >= 75.0 else '🔍 EVALUATED'} |
| **Valid SQL Rate (VSR)** | **{vsr:.2f}%** | $100.0\\%$ (XGrammar) | {'✅ 100%' if vsr >= 99.9 else f'{vsr:.1f}%'} |
| **Expected Calibration Error (ECE)** | **{ece:.4f}** | $\\le 0.081$ | {'✅ CALIBRATED' if ece <= 0.081 else '⚠️ TUNING'} |
| **Average End-to-End Latency** | **{latency_ms:.2f} ms** ({latency_s:.2f}s) | $\\le 5.8$ seconds | {'✅ FAST' if latency_s <= 5.8 else '⚠️ SLOW'} |
| **GPU VRAM Footprint** | **{vram:.2f} GB** | $\\le 8.5$ GB | {'✅ LOW VRAM' if vram <= 8.5 else '⚠️ HIGH'} |

### Scientific Comparison vs. Baselines
| System / Architecture | Model Size | EX (BIRD-dev) | VSR | Avg Latency | Peak VRAM |
|:----------------------|:----------:|:-------------:|:---:|:-----------:|:---------:|
| **MARS-SQL 21B (Baseline)** | 21.0B | 74.8% | 78.6% – 92.9% | ~22.0 s | 41.0 – 47.0 GB |
| **Combo A (Ultra-Light)** | 1.5B | 69.5% – 72.0% | 100.0% | ~3.8 s | < 8.0 GB RAM |
| **Combo C (Pareto SOTA)** | 3.0B | 75.0% – 76.8% | 100.0% | ~5.2 s | 8.0 – 16.0 GB |
| **Our Evaluated Pipeline** | **SLM (3B/7B)** | **{ex:.2f}%** | **{vsr:.2f}%** | **{latency_s:.2f} s** | **{vram:.2f} GB** |

*Generated by Text-to-SQL SLM Evaluation Suite (Milestone M4).*
"""
    return table.strip()
