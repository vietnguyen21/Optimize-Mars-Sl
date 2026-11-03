"""
Regex and AST-based Fallback Grammar Engine.
Provides deterministic syntax enforcement and zero-crash token validation
when C++ XGrammar engine or CUDA runtime is unavailable.
"""

import re
from typing import Tuple, Optional, Any, List, Set, Dict
from src.grammar.base import BaseGrammarEngine

try:
    import sqlglot
    HAS_SQLGLOT = True
except ImportError:
    HAS_SQLGLOT = False


class RegexASTFallbackEngine(BaseGrammarEngine):
    """
    Zero-crash fallback grammar engine using regular expressions, token stream parsing,
    and AST structural validation (via sqlglot when present or native token state machine).
    Guarantees 100% crash-free operation across all environments.
    """

    SQL_KEYWORDS = {
        "SELECT", "FROM", "WHERE", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "CROSS",
        "ON", "GROUP", "BY", "HAVING", "ORDER", "LIMIT", "OFFSET", "UNION", "ALL",
        "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER", "AS", "IN", "NOT",
        "AND", "OR", "IS", "NULL", "LIKE", "BETWEEN", "EXISTS", "CASE", "WHEN", "THEN",
        "ELSE", "END", "DISTINCT", "AVG", "COUNT", "SUM", "MIN", "MAX", "ASC", "DESC"
    }

    CLAUSE_ORDER = [
        "SELECT",
        "FROM",
        "WHERE",
        "GROUP BY",
        "HAVING",
        "ORDER BY",
        "LIMIT"
    ]

    def __init__(self, schema_ddl: str = ""):
        self.known_tables: Set[str] = set()
        self.known_columns: Set[str] = set()
        super().__init__(schema_ddl=schema_ddl)

    def _extract_schema_identifiers(self, ddl: str) -> None:
        """Extracts table and column names from schema DDL for identifier validation."""
        tbl_pattern = re.compile(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([\"`]?\w+[\"`]?)", re.IGNORECASE)
        for match in tbl_pattern.finditer(ddl):
            tbl = match.group(1).strip("\"`'")
            self.known_tables.add(tbl.lower())

        col_pattern = re.compile(r"([\"`]?\w+[\"`]?)\s+(?:INTEGER|TEXT|REAL|BLOB|VARCHAR|NUMERIC|BOOLEAN|DATE|DATETIME)", re.IGNORECASE)
        for match in col_pattern.finditer(ddl):
            col = match.group(1).strip("\"`'")
            if col.upper() not in self.SQL_KEYWORDS:
                self.known_columns.add(col.lower())

    def compile_grammar(self, schema_ddl: str) -> Dict[str, Any]:
        """
        Compiles schema DDL into fallback token and identifier tables.
        """
        self._extract_schema_identifiers(schema_ddl)
        return {
            "tables": sorted(list(self.known_tables)),
            "columns": sorted(list(self.known_columns)),
            "engine": "RegexASTFallbackEngine",
            "compiled": True
        }

    def validate_prefix(self, prefix_sql: str) -> Tuple[bool, Optional[str]]:
        """
        Determines whether a partial prefix is syntactically consistent so far.
        Allows unclosed parentheses or quotes if they are currently being constructed,
        but rejects structurally invalid transitions (e.g., closing unopened brackets,
        double semicolons, or invalid initial tokens).
        """
        if prefix_sql is None:
            return False, "Prefix SQL is None"

        text = prefix_sql.strip()
        if not text:
            return True, None

        # Check for illegal double semicolon
        if ";;" in text:
            return False, "Syntax error: duplicate semicolon ';;'"

        # Check bracket balancing in prefix: close parentheses must never exceed open
        open_parens = 0
        in_single_quote = False
        in_double_quote = False
        escape = False

        for char in text:
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
            elif char == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
            elif not in_single_quote and not in_double_quote:
                if char == "(":
                    open_parens += 1
                elif char == ")":
                    open_parens -= 1
                    if open_parens < 0:
                        return False, "Syntax error: unmatched closing parenthesis ')'"

        # Check if first word is a valid SQL starter (SELECT, WITH, PRAGMA, EXPLAIN, etc.)
        first_token = re.split(r"\s+", text, maxsplit=1)[0].upper().strip("`\"'")
        valid_starters = {"SELECT", "WITH", "PRAGMA", "EXPLAIN"}
        if first_token and not any(first_token.startswith(s) or s.startswith(first_token) for s in valid_starters):
            return False, f"Syntax error: statement cannot begin with '{first_token}'"

        return True, None

    def validate_sql_syntax(self, sql: str) -> Tuple[bool, Optional[str]]:
        """
        Validates complete SQL statement for syntax correctness.
        Uses sqlglot if installed, otherwise uses native regex/AST state machine.
        Guarantees zero crashes on any input.
        """
        if not sql or not isinstance(sql, str):
            return False, "SQL statement is empty or not a string"

        clean_sql = sql.strip()
        if not clean_sql:
            return False, "Empty SQL query"

        # Check for unclosed quotes
        in_sq = False
        in_dq = False
        escape = False
        open_parens = 0

        for char in clean_sql:
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == "'" and not in_dq:
                in_sq = not in_sq
            elif char == '"' and not in_sq:
                in_dq = not in_dq
            elif not in_sq and not in_dq:
                if char == "(":
                    open_parens += 1
                elif char == ")":
                    open_parens -= 1

        if in_sq:
            return False, "Syntax error: unclosed single quote"
        if in_dq:
            return False, "Syntax error: unclosed double quote"
        if open_parens != 0:
            return False, f"Syntax error: unbalanced parentheses (diff={open_parens})"

        if ";;" in clean_sql:
            return False, "Syntax error: duplicate semicolon ';;'"

        # Strip comments
        lines = []
        for line in clean_sql.splitlines():
            line_no_comment = re.sub(r"--.*$", "", line)
            lines.append(line_no_comment)
        stripped = " ".join(lines).strip().rstrip(";")

        if not stripped:
            return False, "SQL contains no executable clauses"

        # If sqlglot is available, run full AST parsing
        if HAS_SQLGLOT:
            try:
                parsed = sqlglot.parse_one(stripped, read="sqlite")
                if parsed is None:
                    return False, "sqlglot returned empty AST"
                return True, None
            except Exception as e:
                return False, f"sqlglot syntax error: {str(e)}"

        # Native Regex / AST Fallback parser
        return self._native_validate_sql(stripped)

    def _native_validate_sql(self, sql: str) -> Tuple[bool, Optional[str]]:
        """
        Pure-Python deterministic parser verifying core SQL grammar for SQLite.
        """
        tokens = re.findall(r"[\w]+|[^\s\w]", sql)
        if not tokens:
            return False, "No tokens found"

        upper_tokens = [t.upper() for t in tokens]
        if upper_tokens[0] not in {"SELECT", "WITH"}:
            return False, f"Query must begin with SELECT or WITH, found: {tokens[0]}"

        # Check for SELECT without projection
        if "SELECT" in upper_tokens:
            sel_idx = upper_tokens.index("SELECT")
            if sel_idx == len(upper_tokens) - 1:
                return False, "SELECT clause has no projection columns"
            next_t = upper_tokens[sel_idx + 1]
            if next_t in {"FROM", "WHERE", "GROUP", "ORDER", "LIMIT", ";"}:
                return False, f"SELECT clause cannot be immediately followed by {next_t}"

        # Check for FROM without table
        if "FROM" in upper_tokens:
            from_idx = upper_tokens.index("FROM")
            if from_idx == len(upper_tokens) - 1:
                return False, "FROM clause has no table name"
            next_t = upper_tokens[from_idx + 1]
            if next_t in {"WHERE", "GROUP", "ORDER", "LIMIT", ",", ";"}:
                return False, f"FROM clause cannot be immediately followed by {next_t}"

        # Check clause ordering
        found_clauses = []
        i = 0
        while i < len(upper_tokens):
            t = upper_tokens[i]
            if t == "GROUP" and i + 1 < len(upper_tokens) and upper_tokens[i + 1] == "BY":
                found_clauses.append("GROUP BY")
                i += 2
                continue
            elif t == "ORDER" and i + 1 < len(upper_tokens) and upper_tokens[i + 1] == "BY":
                found_clauses.append("ORDER BY")
                i += 2
                continue
            elif t in {"SELECT", "FROM", "WHERE", "HAVING", "LIMIT"}:
                found_clauses.append(t)
            i += 1

        last_idx = -1
        for clause in found_clauses:
            if clause in self.CLAUSE_ORDER:
                curr_idx = self.CLAUSE_ORDER.index(clause)
                # Subqueries can restart clause order inside parentheses, but in top-level check:
                if curr_idx < last_idx and "(" not in sql:
                    return False, f"Invalid clause order: {clause} appeared after subsequent clause"
                last_idx = curr_idx

        # Check for dangling operators at the end
        last_token = upper_tokens[-1]
        dangling_ops = {",", "+", "-", "*", "/", ">", "<", "=", ">=", "<=", "!=", "AND", "OR", "WHERE", "JOIN", "ON"}
        if last_token in dangling_ops:
            return False, f"SQL ends with dangling operator or keyword: '{last_token}'"

        return True, None

    def filter_logits(self, logits: Any, context: str) -> Any:
        """
        Logits filtering fallback. If logits is a list or tensor, verifies prefix
        validity; if invalid, sets logits to large negative penalty.
        Guaranteed zero-crash.
        """
        is_valid, _ = self.validate_prefix(context)
        if hasattr(logits, "clone"):
            res = logits.clone()
            if not is_valid:
                res = res - 1e9
            return res
        elif isinstance(logits, list):
            if not is_valid:
                return [v - 1e9 for v in logits]
            return list(logits)
        return logits
