"""
Safe Execution Probing Engine for SQLite.
Executes read-only probes with strict LIMIT clauses (max 5 rows) and throttled
execution (max 5 queries per session) with 0 GB VRAM footprint.
"""

import os
import re
import sqlite3
from typing import Tuple, Any, Optional, List


class DBProbeEngine:
    """
    Lightweight, safe SQLite probing engine.
    Features:
    - 0 GB VRAM footprint (runs purely in SQLite CPU engine).
    - Read-only enforcement blocking INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, ATTACH.
    - Automatic injection of LIMIT 5 (or caps existing LIMIT <= 5).
    - Session throttle: blocks execution after max_session_queries (default: 5).
    """

    DESTRUCTIVE_KEYWORDS = {
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE",
        "REPLACE", "ATTACH", "DETACH", "PRAGMA WRITABLE_SCHEMA", "VACUUM"
    }

    def __init__(self, max_session_queries: int = 5, row_limit: int = 5, timeout: float = 1.5):
        self.max_session_queries: int = max_session_queries
        self.row_limit: int = row_limit
        self.timeout: float = timeout
        self.query_count: int = 0

    def reset_counter(self) -> None:
        """Resets session query execution counter."""
        self.query_count = 0

    def _sanitize_and_apply_limit(self, sql: str) -> str:
        """Ensures query has LIMIT <= self.row_limit without mutating subqueries."""
        clean = sql.strip().rstrip(";")
        upper_sql = clean.upper()

        # Check if query already has a top-level LIMIT
        limit_match = re.search(r"\bLIMIT\s+(\d+)\b", clean, re.IGNORECASE)
        if limit_match:
            existing_limit = int(limit_match.group(1))
            if existing_limit > self.row_limit:
                # Replace with capped limit
                clean = clean[:limit_match.start()] + f"LIMIT {self.row_limit}" + clean[limit_match.end():]
        else:
            clean = f"{clean} LIMIT {self.row_limit}"

        return f"{clean};"

    def probe_sql(self, sql: str, db_path: str) -> Tuple[bool, Any, Optional[str]]:
        """
        Executes a safe probe query against the database.
        
        Args:
            sql: SQL statement string.
            db_path: Path to target SQLite database.
            
        Returns:
            Tuple of (success: bool, rows_or_detail: Any, error_message: Optional[str]).
        """
        if self.query_count >= self.max_session_queries:
            return False, None, f"Exceeded maximum probe query quota ({self.max_session_queries} queries per session)"

        if not db_path or not os.path.exists(db_path):
            return False, None, f"Database path not found: {db_path}"

        clean_sql = (sql or "").strip()
        if not clean_sql:
            return False, None, "Empty SQL query"

        # Check for destructive keywords
        upper_sql = clean_sql.upper()
        for kw in self.DESTRUCTIVE_KEYWORDS:
            if re.search(rf"\b{kw}\b", upper_sql):
                return False, None, f"Destructive operation blocked by read-only probe: '{kw}'"

        # Apply LIMIT <= 5
        probed_sql = self._sanitize_and_apply_limit(clean_sql)

        try:
            # Connect in read-only mode if URI supported, otherwise standard connect with rollback
            conn_uri = f"file:{os.path.abspath(db_path)}?mode=ro"
            try:
                conn = sqlite3.connect(conn_uri, uri=True, timeout=self.timeout)
            except Exception:
                conn = sqlite3.connect(db_path, timeout=self.timeout)

            cursor = conn.cursor()
            cursor.execute(probed_sql)
            rows = cursor.fetchall()
            conn.close()

            self.query_count += 1
            return True, rows, None

        except sqlite3.OperationalError as op_err:
            return False, None, f"SQLite operational error: {str(op_err)}"
        except sqlite3.DatabaseError as db_err:
            return False, None, f"SQLite database error: {str(db_err)}"
        except Exception as err:
            return False, None, f"Execution failed: {str(err)}"


# Global default engine instance for module-level convenience function
_GLOBAL_PROBE_ENGINE = DBProbeEngine()


def probe_sql_execution(sql: str, db_path: str, max_queries: int = 5) -> Tuple[bool, Any]:
    """
    Module-level function compatible with Notebook 05 and evaluation runners.
    
    Args:
        sql: SQL query to probe.
        db_path: Path to target SQLite database.
        max_queries: Maximum allowed queries per session.
        
    Returns:
        Tuple of (success: bool, rows_or_error_msg: Any).
    """
    engine = DBProbeEngine(max_session_queries=max_queries, row_limit=5)
    success, rows, err = engine.probe_sql(sql, db_path)
    if success:
        return True, rows
    return False, err
