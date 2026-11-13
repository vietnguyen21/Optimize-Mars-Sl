"""
Laya-421M (ModernBERT) System-1 0-Token Decision Scorer & Adaptive Early-Exit Gate.
Loads trained sequence-classification weights and held-out temperature scaling.
Latency and ECE must be measured on the target hardware and held-out data.
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
    Calculates candidate confidence with the trained ModernBERT classification
    head and exported temperature. Heuristics are reserved for smoke mode.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.75,
        temperature: float = 0.20,
        early_exit_threshold: float = 0.85,
        device: str = "cpu",
        dry_run: bool = True
    ):
        self.device = device
        self.dry_run = dry_run
        self.max_length = 512
        if not dry_run and not model_path:
            raise ValueError("Real System-1 scoring requires a trained model_path")
        self.model_path = model_path
        self.threshold = threshold
        self.temperature = max(1e-4, temperature)
        self.early_exit_threshold = early_exit_threshold
        self.gate = AdaptiveEarlyExitGate(threshold=threshold, early_exit_threshold=early_exit_threshold)
        self.model = None
        self.tokenizer = None
        self._init_model()

    def _init_model(self) -> None:
        """Loads ModernBERT model if model_path is provided and environment supports torch."""
        if not self.model_path:
            return
        import json
        from pathlib import Path
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        meta_path = Path(self.model_path) / 'calibration_meta.json'
        if not meta_path.exists():
            raise FileNotFoundError(f'Trained System-1 calibration metadata missing: {meta_path}')
        meta = json.loads(meta_path.read_text())
        if not meta.get('trained') or meta.get('mode') != 'train':
            raise ValueError('System-1 artifact is a smoke artifact, not a trained model')
        self.temperature = float(meta['temperature'])
        self.max_length = int(meta.get('max_length', 512))
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError('Invalid trained temperature')
        from src.training.hf_auth import model_hf_options
        hf_options = model_hf_options(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, **hf_options)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path, attn_implementation='eager', reference_compile=False, **hf_options).to(self.device)
        self.model.eval()


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
        if not schema_tables:
            # Fallback table extraction from common names
            schema_tables = ["accounts", "clients", "cards", "districts", "schools", "frpm", "satscores"]

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
        Calculates calibrated confidence probability P in [0.0, 1.0] for a candidate SQL.
        """
        if self.model is not None:
            return self.score_candidates_parallel(question, schema_ddl, [sql])[0]
        if not self.dry_run:
            raise RuntimeError("Real scoring cannot use heuristic fallback")
        logit = self._compute_calibrated_logit(question, schema_ddl, sql)
        # Calibrated temperature scaling
        scaled = logit / self.temperature
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
        Peak inference batch size is bounded; latency is hardware dependent.
        """
        if self.model is None:
            return [self.score_candidate(question, schema_ddl, cand) for cand in candidates]
        import torch
        from src.training.train_decision_model import decision_text, validate_decision_budget
        scores = []
        # Bound peak memory; classifier never autoregressively generates tokens.
        with torch.inference_mode():
            for offset in range(0, len(candidates), 4):
                batch=candidates[offset:offset+4]
                for sql in batch:
                    validate_decision_budget(self.tokenizer,question,sql,self.max_length)
                texts = [decision_text(question, schema_ddl, sql) for sql in batch]
                inputs = self.tokenizer(texts, padding=True, truncation=True,
                                        max_length=self.max_length, return_tensors='pt')
                inputs = {k:v.to(self.device) for k,v in inputs.items()}
                logits = self.model(**inputs).logits.float()
                if logits.shape[-1] == 1:
                    margins = logits[:,0]
                elif logits.shape[-1] == 2:
                    margins = logits[:,1] - logits[:,0]
                else:
                    raise ValueError('System-1 classifier must have one or two logits')
                scores.extend((margins / self.temperature).sigmoid().cpu().tolist())
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
