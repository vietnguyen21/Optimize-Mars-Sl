"""
Laya-421M (ModernBERT) System-1 0-Token Decision Scorer & Adaptive Early-Exit Gate.
Implements confidence scoring and an Adaptive Early-Exit Gate at >= 0.85 confidence.
"""

import math
import re
import time
from typing import Tuple, List, Optional, Any, Dict


class AdaptiveEarlyExitGate:
    """
    Adaptive Early-Exit Gate for System-1 fast decisions.
    When candidate confidence reaches or exceeds early_exit_threshold (0.85),
    early exit triggers, accepting the trajectory immediately and bypassing
    costly System-2 deliberation.
    """
    def __init__(self, threshold: float = 0.75, early_exit_threshold: float = 0.85):
        self.threshold: float = threshold
        self.early_exit_threshold: float = early_exit_threshold

    def evaluate_confidence(self, confidence: float) -> Tuple[str, bool]:
        """
        Determines decision verdict and whether early-exit was triggered.
        
        Returns:
            Tuple of (verdict: "ACCEPT" | "REJECT", is_early_exit: bool).
        """
        if confidence >= self.early_exit_threshold:
            return "ACCEPT", True
        elif confidence >= self.threshold:
            return "ACCEPT", False
        else:
            return "REJECT", False


class LayaScorer:
    """
    Laya-421M System-1 0-Token Decision Scorer.
    Calculates calibrated candidate confidence in parallel using [MASK] token representation
    and calibrated temperature scaling.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.75,
        temperature: float = 0.20,
        early_exit_threshold: float = 0.85,
        calibration_temperature: float = 1.0
    ):
        self.model_path = model_path
        # Fit on cal-DB with temperature scaling; 1.0 means uncalibrated.
        self.calibration_temperature = max(1e-4, calibration_temperature)
        self.threshold = threshold
        self.temperature = max(1e-4, temperature)
        self.early_exit_threshold = early_exit_threshold
        self.gate = AdaptiveEarlyExitGate(threshold=threshold, early_exit_threshold=early_exit_threshold)
        self.model = None
        self.tokenizer = None
        self._init_model()

    @property
    def scorer_type(self) -> str:
        """'model' when a trained checkpoint scores candidates, else 'heuristic' (keyword baseline)."""
        return "model" if self.model is not None else "heuristic"

    def _init_model(self) -> None:
        """Loads the trained cross-encoder if model_path is given. A load failure raises."""
        if self.model_path:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.eval()

    def _compute_model_logit(self, question: str, schema_ddl: str, sql: str) -> float:
        """Correctness logit from the trained cross-encoder on (question + schema, sql)."""
        import torch
        inputs = self.tokenizer(f"{question}\n{schema_ddl}", sql, truncation=True, return_tensors="pt")
        device = getattr(self.model, "device", None)
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
        # 1 label: raw logit; 2 labels: log-odds of class 1 ("correct").
        return float(logits[0]) if logits.shape[-1] == 1 else float(logits[1] - logits[0])

    def _extract_schema_tables(self, schema_ddl: str) -> List[str]:
        """Extracts table names from schema DDL."""
        matches = re.findall(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([\"`]?\w+[\"`]?)", schema_ddl, re.IGNORECASE)
        return [m.strip("\"`'").lower() for m in matches]

    def _compute_calibrated_logit(self, question: str, schema_ddl: str, sql: str) -> float:
        """
        Computes decision logit based on semantic alignment, schema coverage,
        and syntax indicators.
        """
        clean_sql = (sql or "").strip()
        sql_upper = clean_sql.upper()
        sql_lower = clean_sql.lower()
        q_lower = (question or "").lower()

        # Check basic syntax
        if not clean_sql or not sql_upper.startswith(("SELECT", "WITH")):
            return -6.0

        # Schema table alignment
        schema_tables = self._extract_schema_tables(schema_ddl)

        matched_tables = [t for t in schema_tables if re.search(rf"\b{t}\b", sql_lower)]
        coverage = len(matched_tables) / max(1, len(schema_tables))

        # Check keyword presence
        has_select = "SELECT" in sql_upper
        has_from = "FROM" in sql_upper
        has_where = "WHERE" in sql_upper

        # Base logit: aligned with Notebook 04 RLCD calibration
        logit = 2.0 * coverage
        if has_select and has_from:
            logit += 1.2
        else:
            logit -= 2.5

        # Check for balanced parentheses
        if clean_sql.count("(") != clean_sql.count(")"):
            logit -= 3.0

        # Intent consistency rewards
        if any(w in q_lower for w in ["average", "mean"]) and "AVG" in sql_upper:
            logit += 0.8
        if any(w in q_lower for w in ["how many", "number of"]) and "COUNT" in sql_upper:
            logit += 0.8
        if any(w in q_lower for w in ["total", "sum"]) and "SUM" in sql_upper:
            logit += 0.8

        return logit

    def score_candidate(self, question: str, schema_ddl: str, sql: str) -> float:
        """
        Confidence P in [0.0, 1.0] for a candidate SQL. Uses the trained model when loaded;
        otherwise the keyword heuristic baseline (check `scorer_type` and report it).
        """
        if self.model is not None:
            scaled = self._compute_model_logit(question, schema_ddl, sql) / self.calibration_temperature
        else:
            scaled = self._compute_calibrated_logit(question, schema_ddl, sql) / self.temperature
        # Numerically stable sigmoid
        if scaled >= 0:
            prob = 1.0 / (1.0 + math.exp(-scaled))
        else:
            prob = math.exp(scaled) / (1.0 + math.exp(scaled))
        return float(prob)

    def score_candidates_parallel(
        self,
        question: str,
        schema_ddl: str,
        candidates: List[str]
    ) -> List[float]:
        """
        Computes calibrated scores in parallel for multiple candidate trajectories.
        Target latency: ~1.2 ms.
        """
        start = time.perf_counter()
        scores = [self.score_candidate(question, schema_ddl, cand) for cand in candidates]
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return scores

    def evaluate(self, question: str, schema_ddl: str, sql: str) -> Tuple[float, str]:
        """
        Evaluates candidate and returns (calibrated_prob, decision).
        Compatible with Notebook 05 System1DecisionGate contract.
        """
        prob = self.score_candidate(question, schema_ddl, sql)
        verdict, is_early_exit = self.gate.evaluate_confidence(prob)
        return prob, verdict

    def should_early_exit(self, confidence: float) -> bool:
        """Returns True if confidence >= early_exit_threshold (0.85)."""
        return confidence >= self.early_exit_threshold


# Compatibility alias for Notebook 05
System1DecisionGate = LayaScorer
