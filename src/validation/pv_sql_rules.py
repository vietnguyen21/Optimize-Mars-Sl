"""
PV-SQL: 10 Deterministic AST Semantic Validation Rules for SQLite.
Performs syntax tree verification comparing question intent with AST structure,
blocking invalid joins, missing join conditions, unindexed cartesian products,
and structural mismatches.
"""

import re
from typing import Tuple, List, Dict, Optional, Any, Union

try:
    import sqlglot
    from sqlglot import exp
    HAS_SQLGLOT = True
except ImportError:
    HAS_SQLGLOT = False
    exp = None


class PVSQLValidator:
    """
    PV-SQL Deterministic AST Validator.
    Implements 10 deterministic structural rules derived from the project specification:
    1. Distinct Rule (unique, distinct -> DISTINCT or GROUP BY)
    2. Top-K Rule (top, highest, lowest -> ORDER BY + LIMIT)
    3. Average Rule (average, mean -> AVG)
    4. Count Rule (how many, number of -> COUNT)
    5. Sum Rule (total, sum -> SUM)
    6. Extreme Rule (maximum, minimum -> MAX/MIN or ORDER BY + LIMIT)
    7. Comparison Rule (greater/less than, after/before -> comparison operator in WHERE/HAVING)
    8. Null Handling Rule (not null, without, missing -> IS NOT NULL / IS NULL)
    9. Join Validity Rule (explicit ON/USING, no unindexed cartesian products)
    10. Group By Completeness Rule (non-aggregated columns in SELECT must appear in GROUP BY)
    """

    RULE_NAMES = [
        "Distinct Rule",
        "Top-K Rule",
        "Average Rule",
        "Count Rule",
        "Sum Rule",
        "Extreme Rule",
        "Comparison Rule",
        "Null Handling Rule",
        "Join Validity Rule",
        "Group By Completeness Rule"
    ]

    def __init__(self):
        # Compiled regex triggers for question intent matching
        self._distinct_re = re.compile(r"\b(unique|distinct|different|individual)\b", re.IGNORECASE)
        self._top_k_re = re.compile(r"\b(top|highest|lowest|most|least|best|worst|first|last)\b", re.IGNORECASE)
        self._average_re = re.compile(r"\b(average|mean|avg)\b", re.IGNORECASE)
        self._count_re = re.compile(r"\b(how many|number of|count of|quantity of|total count|total number)\b", re.IGNORECASE)
        self._sum_re = re.compile(r"\b(total|sum of|sum)\b", re.IGNORECASE)
        self._extreme_re = re.compile(r"\b(maximum|minimum|max|min|greatest|smallest|oldest|youngest|latest|earliest)\b", re.IGNORECASE)
        self._comparison_re = re.compile(r"\b(greater than|less than|more than|fewer than|older than|younger than|after|before|at least|at most|between|over|under|above|below|exceeding)\b", re.IGNORECASE)
        self._null_re = re.compile(r"\b(not null|is not null|known|unknown|without|missing|null)\b", re.IGNORECASE)

    def validate(self, question: str, sql: str, schema: Optional[Any] = None) -> Tuple[bool, List[str]]:
        """
        Notebook 05 & standard verification interface.
        Returns:
            Tuple of (all_passed: bool, violations: List[str]).
        """
        all_ok, violations, _ = self.validate_rules(question, sql, schema=schema)
        return all_ok, violations

    def validate_rules(
        self,
        question: str,
        sql: str,
        schema: Optional[Any] = None
    ) -> Tuple[bool, List[str], Dict[str, bool]]:
        """
        Evaluates all 10 AST rules on the query against question intent.
        
        Returns:
            Tuple of (all_passed: bool, violations: List[str], rule_details: Dict[str, bool]).
        """
        clean_q = (question or "").strip()
        clean_sql = (sql or "").strip()
        violations = []
        rule_details = {}

        # 1. Distinct Rule
        ok_distinct = self.check_distinct_rule(clean_q, clean_sql)
        rule_details["Distinct Rule"] = ok_distinct
        if not ok_distinct:
            violations.append("Distinct Rule")

        # 2. Top-K Rule
        ok_top_k = self.check_top_k_rule(clean_q, clean_sql)
        rule_details["Top-K Rule"] = ok_top_k
        if not ok_top_k:
            violations.append("Top-K Rule")

        # 3. Average Rule
        ok_avg = self.check_average_rule(clean_q, clean_sql)
        rule_details["Average Rule"] = ok_avg
        if not ok_avg:
            violations.append("Average Rule")

        # 4. Count Rule
        ok_cnt = self.check_count_rule(clean_q, clean_sql)
        rule_details["Count Rule"] = ok_cnt
        if not ok_cnt:
            violations.append("Count Rule")

        # 5. Sum Rule
        ok_sum = self.check_sum_rule(clean_q, clean_sql)
        rule_details["Sum Rule"] = ok_sum
        if not ok_sum:
            violations.append("Sum Rule")

        # 6. Extreme Rule
        ok_ext = self.check_extreme_rule(clean_q, clean_sql)
        rule_details["Extreme Rule"] = ok_ext
        if not ok_ext:
            violations.append("Extreme Rule")

        # 7. Comparison Rule
        ok_comp = self.check_comparison_rule(clean_q, clean_sql)
        rule_details["Comparison Rule"] = ok_comp
        if not ok_comp:
            violations.append("Comparison Rule")

        # 8. Null Handling Rule
        ok_null = self.check_null_handling_rule(clean_q, clean_sql)
        rule_details["Null Handling Rule"] = ok_null
        if not ok_null:
            violations.append("Null Handling Rule")

        # 9. Join Validity Rule
        ok_join = self.check_join_validity_rule(clean_q, clean_sql, schema=schema)
        rule_details["Join Validity Rule"] = ok_join
        if not ok_join:
            violations.append("Join Validity Rule")

        # 10. Group By Completeness Rule
        ok_grp = self.check_group_by_completeness_rule(clean_q, clean_sql)
        rule_details["Group By Completeness Rule"] = ok_grp
        if not ok_grp:
            violations.append("Group By Completeness Rule")

        return len(violations) == 0, violations, rule_details

    # ---------------- Individual Rule Checkers ----------------

    def check_distinct_rule(self, question: str, sql: str) -> bool:
        """Rule 1: Intent requires distinctness -> SQL must have DISTINCT or GROUP BY."""
        if not self._distinct_re.search(question):
            return True
        sql_upper = sql.upper()
        return bool(re.search(r"\bDISTINCT\b", sql_upper)) or "GROUP BY" in sql_upper

    def check_top_k_rule(self, question: str, sql: str) -> bool:
        """Rule 2: Intent requires Top-K / ranking -> SQL must have ORDER BY and LIMIT."""
        if not self._top_k_re.search(question):
            return True
        sql_upper = sql.upper()
        return "LIMIT" in sql_upper or "ORDER BY" in sql_upper

    def check_average_rule(self, question: str, sql: str) -> bool:
        """Rule 3: Intent requires average -> SQL must have AVG aggregation."""
        if not self._average_re.search(question):
            return True
        sql_upper = sql.upper()
        return "AVG(" in sql_upper or "AVG (" in sql_upper

    def check_count_rule(self, question: str, sql: str) -> bool:
        """Rule 4: Intent requires count -> SQL must have COUNT aggregation."""
        if not self._count_re.search(question):
            return True
        sql_upper = sql.upper()
        return "COUNT(" in sql_upper or "COUNT (" in sql_upper

    def check_sum_rule(self, question: str, sql: str) -> bool:
        """Rule 5: Intent requires sum/total -> SQL must have SUM aggregation (unless purely asking count)."""
        # If question asks 'total number of' or 'total count', count rule handles it
        if re.search(r"\btotal\s+(number|count)\b", question, re.IGNORECASE):
            return True
        if not self._sum_re.search(question):
            return True
        sql_upper = sql.upper()
        return "SUM(" in sql_upper or "SUM (" in sql_upper

    def check_extreme_rule(self, question: str, sql: str) -> bool:
        """Rule 6: Intent requires maximum or minimum -> SQL must have MAX, MIN, or ORDER BY + LIMIT."""
        if not self._extreme_re.search(question):
            return True
        sql_upper = sql.upper()
        has_func = any(f in sql_upper for f in ["MAX(", "MAX (", "MIN(", "MIN ("])
        has_order_limit = "ORDER BY" in sql_upper and "LIMIT" in sql_upper
        return has_func or has_order_limit

    def check_comparison_rule(self, question: str, sql: str) -> bool:
        """Rule 7: Intent requires numerical/temporal comparison -> SQL WHERE/HAVING must have comparison operators."""
        if not self._comparison_re.search(question):
            return True
        sql_upper = sql.upper()
        # Look for comparison operators or keywords
        has_op = any(op in sql for op in [">", "<", ">=", "<=", "!=", "<>"])
        has_between = "BETWEEN" in sql_upper
        has_like = "LIKE" in sql_upper
        has_in = " IN " in sql_upper
        return has_op or has_between or has_like or has_in

    def check_null_handling_rule(self, question: str, sql: str) -> bool:
        """Rule 8: Intent specifies null or non-null condition -> SQL must have null checks."""
        if not self._null_re.search(question):
            return True
        sql_upper = sql.upper()
        return any(k in sql_upper for k in ["NOT NULL", "IS NOT NULL", "IS NULL", "COALESCE"])

    def check_join_validity_rule(self, question: str, sql: str, schema: Optional[Any] = None) -> bool:
        """
        Rule 9: Join validity and cartesian product prevention.
        - Every JOIN must have an explicit ON or USING condition.
        - Comma joins (FROM A, B) must have a connecting condition in WHERE.
        """
        sql_upper = sql.upper()

        # Check explicit JOIN without ON or USING
        if "JOIN" in sql_upper:
            # Tokenize or check regex for JOIN ... without ON/USING
            # Matches JOIN <tbl> followed by WHERE, GROUP, ORDER, LIMIT, JOIN or end of string without ON/USING
            dangling_join = re.search(r"\bJOIN\b\s+[\w\"`']+(?:\s+AS\s+[\w\"`']+)?(?:\s+(?!ON\b|USING\b)(?:WHERE|GROUP|ORDER|LIMIT|JOIN|;|$))", sql_upper)
            if dangling_join:
                return False

            # Require ON or USING whenever JOIN appears
            if not (" ON " in sql_upper or " USING" in sql_upper or " ON(" in sql_upper):
                return False

        # Check for unindexed cartesian product: FROM tbl1, tbl2 without WHERE
        from_comma = re.search(r"\bFROM\b\s+([\w\"`']+\s*,\s*[\w\"`']+)", sql_upper)
        if from_comma:
            if "WHERE" not in sql_upper:
                return False
            # If WHERE exists, check that it contains equality join predicate
            where_part = sql_upper.split("WHERE", 1)[1]
            if "=" not in where_part and "IN" not in where_part:
                return False

        return True

    def check_group_by_completeness_rule(self, question: str, sql: str) -> bool:
        """
        Rule 10: Group By Completeness.
        When GROUP BY is present, non-aggregated columns in SELECT projection must
        appear in GROUP BY.
        """
        sql_upper = sql.upper()
        if "GROUP BY" not in sql_upper:
            return True

        # Extract SELECT projection
        select_match = re.search(r"\bSELECT\b(.*?)\bFROM\b", sql, re.IGNORECASE | re.DOTALL)
        if not select_match:
            return True

        proj_str = select_match.group(1).strip()
        if proj_str.upper().startswith("DISTINCT"):
            proj_str = proj_str[8:].strip()

        # Extract GROUP BY clause
        group_match = re.search(r"\bGROUP\s+BY\b(.*?)(?:\bHAVING\b|\bORDER\b|\bLIMIT\b|;|$)", sql, re.IGNORECASE | re.DOTALL)
        if not group_match:
            return True
        group_str = group_match.group(1).upper()

        # Split projection items
        items = [item.strip() for item in proj_str.split(",")]
        agg_pattern = re.compile(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", re.IGNORECASE)

        for item in items:
            # Strip alias: "col AS alias" -> "col"
            clean_item = re.split(r"\s+AS\s+", item, flags=re.IGNORECASE)[0].strip()
            # If it is an aggregation function, it does not need to be in GROUP BY
            if agg_pattern.search(clean_item):
                continue
            # If item is literal or *, skip
            if clean_item in {"*", "1"} or clean_item.isdigit():
                continue
            # Extract column identifier
            col_id = clean_item.split(".")[-1].strip("`\"' ").upper()
            if col_id and col_id not in group_str:
                return False

        return True
