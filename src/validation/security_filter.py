"""
Security Filter and Anti-Hallucination Abstention Engine.
Defends against SQL prompt injection, dangerous primitives, stacked queries,
and schema hallucinations, triggering ABSTAIN when confidence or safety fails.
"""

import re
from typing import Tuple, List, Optional, Dict, Set, Any
from src.common.schemas import CompactSubSchema


class SecurityFilter:
    """
    Security filter protecting the execution environment from:
    1. SQL Prompt Injection (stacked queries, payload execution, comment evasion)
    2. Hostile SQLite primitives (load_extension, readfile, writefile)
    3. Schema Hallucinations (referencing non-existent tables or columns)
    """

    HOSTILE_PATTERNS = [
        re.compile(r";\s*(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|CREATE|REPLACE)\b", re.IGNORECASE),
        re.compile(r"\b(EXEC|EXECUTE|XP_CMDSHELL|SP_EXECUTESQL)\b", re.IGNORECASE),
        re.compile(r"\b(LOAD_EXTENSION|READFILE|WRITEFILE)\s*\(", re.IGNORECASE),
        re.compile(r"\b(ATTACH\s+DATABASE|DETACH\s+DATABASE)\b", re.IGNORECASE),
        re.compile(r"\bPRAGMA\s+WRITABLE_SCHEMA\b", re.IGNORECASE),
        re.compile(r"/\*.*?\*/", re.DOTALL), # C-style comments used for obfuscation
    ]

    PROMPT_INJECTION_KEYWORDS = [
        "ignore previous instructions",
        "ignore all previous",
        "system prompt",
        "you are now in dan mode",
        "you are now a helpful hacker",
        "bypass guardrails",
        "drop all tables"
    ]

    def __init__(self):
        pass

    def is_injection_safe(self, sql: str, question: str = "") -> Tuple[bool, Optional[str]]:
        """
        Validates that SQL and question do not contain injection attacks or hostile primitives.
        
        Returns:
            Tuple of (is_safe: bool, reason: Optional[str]).
        """
        if not sql or not isinstance(sql, str):
            return False, "SQL statement is empty"

        # Check question for prompt injection keywords
        q_lower = (question or "").lower()
        for phrase in self.PROMPT_INJECTION_KEYWORDS:
            if phrase in q_lower:
                return False, f"Prompt injection detected in input question: '{phrase}'"

        # Check SQL for hostile patterns
        for pattern in self.HOSTILE_PATTERNS:
            match = pattern.search(sql)
            if match:
                return False, f"Hostile SQL pattern blocked: '{match.group(0)}'"

        # Check for multiple executable statements separated by semicolon
        statements = [s.strip() for s in sql.split(";") if s.strip()]
        if len(statements) > 1:
            return False, f"Multiple stacked SQL statements detected ({len(statements)} statements)"

        return True, None

    def check_schema_hallucinations(
        self,
        sql: str,
        known_tables: List[str],
        known_columns: Optional[Dict[str, List[str]]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Verifies that all tables and columns referenced in SQL exist in the schema.
        
        Returns:
            Tuple of (no_hallucinations: bool, list_of_hallucinated_identifiers: List[str]).
        """
        if not known_tables:
            return True, []

        clean_sql = sql.strip()
        lower_tables = {t.lower() for t in known_tables}
        hallucinations = []

        # Extract table names from FROM and JOIN clauses
        table_matches = re.findall(
            r"\b(?:FROM|JOIN)\s+([\"`]?\w+[\"`]?)",
            clean_sql,
            re.IGNORECASE
        )

        for tbl in table_matches:
            clean_tbl = tbl.strip("\"`'").lower()
            # Ignore subqueries or common aliases
            if clean_tbl not in lower_tables and not clean_tbl.startswith(("subq", "t", "temp")):
                hallucinations.append(f"Table '{tbl}' does not exist in schema")

        # Extract column names if known_columns is provided
        if known_columns:
            all_known_cols = set()
            for cols in known_columns.values():
                for c in cols:
                    all_known_cols.add(c.lower())

            # Look for dotted column expressions: tbl.col
            dotted_cols = re.findall(r"\b[\w]+\.([\"`]?\w+[\"`]?)", clean_sql)
            for c in dotted_cols:
                clean_col = c.strip("\"`'").lower()
                if clean_col not in all_known_cols and clean_col not in {"*"}:
                    hallucinations.append(f"Column '{c}' does not exist in schema")

        return len(hallucinations) == 0, hallucinations


class AbstentionManager:
    """
    Manages Anti-Hallucination Abstention.
    When candidate generation yields invalid, hallucinated, or uncalibrated results,
    triggers an explicit ABSTAIN verdict with diagnostic explanation.
    """

    def __init__(self, min_confidence: float = 0.50):
        self.min_confidence = min_confidence

    def should_abstain(
        self,
        valid_candidates_count: int,
        best_confidence: float,
        has_security_violation: bool,
        has_hallucinations: bool
    ) -> Tuple[bool, Optional[str]]:
        """
        Determines whether the system must abstain from answering.
        
        Returns:
            Tuple of (should_abstain: bool, reason: Optional[str]).
        """
        if has_security_violation:
            return True, "Security violation: prompt injection or forbidden SQL primitives detected"

        if has_hallucinations:
            return True, "Schema hallucination: candidate references non-existent tables/columns"

        if valid_candidates_count == 0:
            return True, "All generated candidate trajectories failed syntax, AST rules, or DB probing"

        if best_confidence < self.min_confidence:
            return True, f"Confidence {best_confidence:.4f} below abstention threshold {self.min_confidence:.2f}"

        return False, None
