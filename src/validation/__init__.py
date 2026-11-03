"""
Validation and System-1 Scoring Package for Text-to-SQL SLM Architecture.
Exports PV-SQL 10 AST rules validator, lightweight DB probing engine,
Laya-421M System-1 parallel scorer, security filter, and candidate validation pipeline.
"""

from typing import Optional, List, Dict, Any
from src.common.schemas import SQLCandidate, ValidationResult, CompactSubSchema
from src.validation.pv_sql_rules import PVSQLValidator
from src.validation.db_probe import DBProbeEngine, probe_sql_execution
from src.validation.laya_scorer import LayaScorer, AdaptiveEarlyExitGate, System1DecisionGate
from src.validation.security_filter import SecurityFilter, AbstentionManager


def validate_candidate(
    candidate: SQLCandidate,
    question: str,
    db_path: str,
    schema_ddl: str = "",
    known_tables: Optional[List[str]] = None,
    known_columns: Optional[Dict[str, List[str]]] = None,
    validator: Optional[PVSQLValidator] = None,
    probe_engine: Optional[DBProbeEngine] = None,
    scorer: Optional[LayaScorer] = None,
    security_filter: Optional[SecurityFilter] = None
) -> ValidationResult:
    """
    Validates a single candidate SQL trajectory through the complete validation pipeline:
    1. Security Filter: checks prompt injection and schema hallucinations.
    2. PV-SQL 10 AST Rules: verifies question-AST alignment.
    3. Safe DB Probe: executes read-only probe on target SQLite with LIMIT 5.
    4. Laya-421M System-1 Scoring: computes calibrated confidence score.
    5. Final Verdict Decision: produces ACCEPT, REJECT, or ABSTAIN.
    
    Returns:
        ValidationResult adhering to canonical interface contract.
    """
    sec = security_filter or SecurityFilter()
    pv = validator or PVSQLValidator()
    probe = probe_engine or DBProbeEngine()
    sys1 = scorer or LayaScorer()

    sql = candidate.sql

    # 1. Security Check
    is_safe, sec_reason = sec.is_injection_safe(sql, question=question)
    if not is_safe:
        return ValidationResult(
            candidate_id=candidate.candidate_id,
            is_valid_syntax=False,
            pv_sql_passed=False,
            probe_passed=False,
            system1_score=0.0,
            security_safe=False,
            final_verdict="ABSTAIN",
            rejection_reason=sec_reason
        )

    # 2. Schema Hallucination Check
    if known_tables:
        no_hallucination, h_reasons = sec.check_schema_hallucinations(sql, known_tables, known_columns)
        if not no_hallucination:
            return ValidationResult(
                candidate_id=candidate.candidate_id,
                is_valid_syntax=False,
                pv_sql_passed=False,
                probe_passed=False,
                system1_score=0.05,
                security_safe=True,
                final_verdict="REJECT",
                rejection_reason="; ".join(h_reasons)
            )

    # 3. PV-SQL 10 AST Rules Check
    pv_passed, violations = pv.validate(question, sql)

    # 4. DB Probe Execution
    probe_passed, rows, probe_err = probe.probe_sql(sql, db_path)

    # 5. System-1 Confidence Score
    system1_score = sys1.score_candidate(question, schema_ddl, sql)

    # 6. Final Verdict
    rejection_reasons = []
    if not pv_passed:
        rejection_reasons.append(f"PV-SQL violations: {', '.join(violations)}")
    if not probe_passed:
        rejection_reasons.append(f"DB probe error: {probe_err}")

    if probe_passed and pv_passed and system1_score >= sys1.threshold:
        final_verdict = "ACCEPT"
        rejection_reason = None
    else:
        final_verdict = "REJECT"
        rejection_reason = "; ".join(rejection_reasons) if rejection_reasons else "Score below threshold"

    return ValidationResult(
        candidate_id=candidate.candidate_id,
        is_valid_syntax=probe_passed,
        pv_sql_passed=pv_passed,
        probe_passed=probe_passed,
        system1_score=system1_score,
        security_safe=True,
        final_verdict=final_verdict,
        rejection_reason=rejection_reason
    )


__all__ = [
    "PVSQLValidator",
    "DBProbeEngine",
    "probe_sql_execution",
    "LayaScorer",
    "AdaptiveEarlyExitGate",
    "System1DecisionGate",
    "SecurityFilter",
    "AbstentionManager",
    "validate_candidate"
]
