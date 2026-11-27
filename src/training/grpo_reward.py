"""Arctic execution rewards and Dr. GRPO objectives, W1/U17/U18.

Gold SQL is evaluator-only input. Responses must contain a nonempty answer tag.
Formatting and AST scores never contribute to the optimized scalar reward.
"""
from functools import lru_cache
import math
from pathlib import Path
import re


class GoldExecutionError(RuntimeError):
    """Invalid gold execution: exclude and count the question, never score it."""


class GoldErrorRateExceeded(RuntimeError):
    """More than 1% gold failures after at least 100 distinct attempted questions."""


def _run_sql(sql, db_path, timeout):
    # Local import avoids a circular import with the trainer's public API.
    from src.training.train_grpo_generator import execute_sqlite_query
    return execute_sqlite_query(sql, str(db_path), timeout=timeout)


@lru_cache(maxsize=None)
def _cached_gold(gold_sql, db_path, file_signature, timeout):
    ok, rows, error = _run_sql(gold_sql, db_path, timeout)
    if not ok:
        # Cache the diagnostic too, so a failed question is not executed per candidate.
        return False, str(error)
    return True, tuple(tuple(row) for row in rows)


def clear_gold_cache():
    _cached_gold.cache_clear()


def load_gold_rows(gold_sql, db_path, timeout=30.0):
    """Execute a gold query once per cache key; errors raise GoldExecutionError.

    Cache identity includes DB path, size, mtime_ns, SQL and deadline. Cached rows
    are immutable. The cache lasts for this process/dataset until explicitly
    cleared. A new runtime executes gold again; no stale rows cross sessions.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Gold timeout must be positive and finite")
    path = Path(db_path).resolve()
    try:
        stat = path.stat()
    except OSError as exc:
        raise GoldExecutionError(f"Gold database unavailable: {path}") from exc
    ok, result = _cached_gold(gold_sql, str(path), (stat.st_size, stat.st_mtime_ns), float(timeout))
    if not ok:
        raise GoldExecutionError(f"Gold execution failed: {result}")
    return result


def compute_arctic_reward(response_text, gold_rows, db_path, timeout=5.0) -> float:
    """Return 1.0 / 0.1 / 0.0, comparing raw row sets through the shared metric."""
    from src.evaluation.metrics import compare_sql_results
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Candidate timeout must be positive and finite")
    if gold_rows is None:
        raise GoldExecutionError("Gold rows are unavailable")
    match = re.search(r"<answer>(.*?)</answer>", response_text or "", re.DOTALL)
    if match is None or not match.group(1).strip():
        return 0.0
    ok, rows, _ = _run_sql(match.group(1).strip(), db_path, timeout)
    if not ok:
        return 0.0
    return 1.0 if compare_sql_results(rows, gold_rows) else 0.1


def dr_grpo_advantages(rewards) -> list:
    values = list(rewards)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Rewards must be finite")
    if not values:
        return []
    if all(value == values[0] for value in values):
        return [0.0] * len(values)
    mean = math.fsum(values) / len(values)
    return [value - mean for value in values]


def dr_grpo_token_objective(new_logprobs, old_logprobs, reference_logprobs, advantage,
                            norm_constant, clip_eps=0.2, beta_kl=0.001):
    """Clipped token sum / fixed Lmax, including EOS, for policy and reference KL.

    Inputs are one unpadded completion. The caller averages over sampled responses
    and accumulation windows. No division by actual completion length or reward std.
    """
    import torch
    if not math.isfinite(norm_constant) or norm_constant <= 0:
        raise ValueError("norm_constant must be positive and finite")
    if (new_logprobs.ndim != 1 or new_logprobs.numel() == 0
            or new_logprobs.shape != old_logprobs.shape
            or new_logprobs.shape != reference_logprobs.shape):
        raise ValueError("Pass matching nonempty 1D completion-token vectors")
    if not 0 < clip_eps < 1 or not math.isfinite(beta_kl) or beta_kl < 0:
        raise ValueError("Invalid clipping or KL coefficient")
    ratio = (new_logprobs - old_logprobs.detach()).exp()
    surrogate = torch.minimum(ratio * advantage,
                              ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantage)
    delta = reference_logprobs.detach() - new_logprobs
    kl = delta.exp() - delta - 1
    return (-surrogate + beta_kl * kl).sum() / norm_constant


class GoldExecutionMonitor:
    """Track distinct question identities and enforce the U17 error threshold."""

    def __init__(self):
        self._attempted = set()
        self._failed = set()

    @property
    def attempted(self):
        return len(self._attempted)

    @property
    def failed(self):
        return len(self._failed)

    @property
    def fraction(self):
        return self.failed / self.attempted if self.attempted else 0.0

    def observe(self, question_id, *, failed):
        key = str(question_id)
        self._attempted.add(key)
        if failed:
            self._failed.add(key)

    def check(self):
        if self.attempted >= 100 and self.fraction > 0.01:
            raise GoldErrorRateExceeded(
                f"Gold failures {self.failed}/{self.attempted} = {self.fraction:.2%} exceed 1%"
            )

    def state(self):
        return {"attempted": sorted(self._attempted), "failed": sorted(self._failed)}

    @classmethod
    def from_state(cls, state):
        monitor = cls()
        if state:
            monitor._attempted = set(state["attempted"])
            monitor._failed = set(state["failed"])
            if not monitor._failed <= monitor._attempted:
                raise ValueError("Invalid restored gold-error counts")
        return monitor
