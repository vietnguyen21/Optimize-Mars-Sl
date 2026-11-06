"""
Safe Execution Probing Engine for SQLite.
Executes read-only probes with strict LIMIT clauses (max 5 rows) and throttled
execution (max 5 queries per session) with 0 GB VRAM footprint.
"""

import os
import re
import sqlite3
import time
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
        """Cap returned rows without rewriting nested LIMIT or ORDER BY clauses."""
        clean = sql.strip().rstrip(";")
        # Newlines prevent a trailing -- comment from consuming our wrapper.
        return f"SELECT * FROM (\n{clean}\n) AS _probe_result LIMIT {self.row_limit};"

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

        conn = None
        self.query_count += 1  # Failed execution attempts also consume probe budget.
        try:
            conn_uri = f"file:{os.path.abspath(db_path)}?mode=ro"
            conn = sqlite3.connect(conn_uri, uri=True, timeout=self.timeout)
            conn.execute("PRAGMA query_only = ON")
            deadline = time.monotonic() + self.timeout
            conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
            cursor = conn.execute(probed_sql)
            rows = cursor.fetchmany(self.row_limit)
            return True, rows, None
        except sqlite3.DatabaseError as db_err:
            return False, None, f"SQLite database error: {str(db_err)}"
        except Exception as err:
            return False, None, f"Execution failed: {str(err)}"
        finally:
            if conn is not None:
                conn.close()


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
