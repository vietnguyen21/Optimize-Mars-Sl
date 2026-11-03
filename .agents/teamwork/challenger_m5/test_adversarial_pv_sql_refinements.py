#!/usr/bin/env python3
"""
Adversarial Test Suite for PV-SQL Rules Refinements & Boundary Conditions
Milestone M5 - Empirical Challenger

Verifies:
1. Rule 1: Distinct Rule regex word boundary (\\bDISTINCT\\b) preventing false positives
   on identifiers containing 'distinct' (e.g. distinct_clients, DISTINCT_CLIENTS).
2. Rule 8: Null Handling Rule keyword enforcement, blocking WHERE-only bypass
   (e.g. WHERE age > 20) when question asks for null/unknown/missing.
3. Edge case coverage across all remaining 8 PV-SQL rules.
"""

import os
import sys
import unittest

# Add repository root to sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.validation.pv_sql_rules import PVSQLValidator


class TestPVSQLAdversarialRefinements(unittest.TestCase):

    def setUp(self):
        self.validator = PVSQLValidator()

    # ---------------- Rule 1: Distinct Rule Adversarial Tests ----------------

    def test_rule_1_false_positive_prevention_on_table_name(self):
        """Distinct intent: 'distinct_clients' in FROM clause must NOT satisfy Rule 1."""
        question = "List the unique client names from the system."
        sql_fail = "SELECT client_name FROM distinct_clients;"
        self.assertFalse(self.validator.check_distinct_rule(question, sql_fail))

        # Full validation pipeline check
        is_valid, violations = self.validator.validate(question, sql_fail)
        self.assertFalse(is_valid)
        self.assertIn("Distinct Rule", violations)

    def test_rule_1_false_positive_prevention_on_column_name(self):
        """Distinct intent: column containing 'distinct' must NOT satisfy Rule 1."""
        question = "What are the distinct countries represented?"
        sql_fail = "SELECT distinct_country_code FROM locations;"
        self.assertFalse(self.validator.check_distinct_rule(question, sql_fail))

    def test_rule_1_false_positive_prevention_uppercase_underscore_identifier(self):
        """Distinct intent: identifier 'DISTINCT_CLIENTS' must NOT satisfy \\bDISTINCT\\b."""
        question = "Show different employee departments."
        sql_fail = "SELECT DISTINCT_DEPARTMENTS FROM employees;"
        self.assertFalse(self.validator.check_distinct_rule(question, sql_fail))

    def test_rule_1_valid_distinct_keyword_passes(self):
        """Distinct intent: explicit DISTINCT keyword passes, even with distinct_clients table."""
        question = "List the unique client names."
        sql_pass = "SELECT DISTINCT client_name FROM distinct_clients;"
        self.assertTrue(self.validator.check_distinct_rule(question, sql_pass))

    def test_rule_1_group_by_passes(self):
        """Distinct intent: GROUP BY alternative satisfies Rule 1."""
        question = "List the individual regions where we operate."
        sql_pass = "SELECT region FROM distinct_clients GROUP BY region;"
        self.assertTrue(self.validator.check_distinct_rule(question, sql_pass))

    def test_rule_1_case_insensitivity_and_multiline(self):
        """Distinct intent: multiline lowercase 'distinct' passes."""
        question = "List unique categories."
        sql_pass = """
        select distinct
            category_id
        from
            categories
        """
        self.assertTrue(self.validator.check_distinct_rule(question, sql_pass))

    def test_rule_1_non_distinct_question_ignores_distinct_clause(self):
        """Non-distinct question: passes regardless of whether SQL has DISTINCT or not."""
        question = "What is the phone number of client 101?"
        sql = "SELECT phone FROM distinct_clients WHERE client_id = 101;"
        self.assertTrue(self.validator.check_distinct_rule(question, sql))

    # ---------------- Rule 8: Null Handling Rule Adversarial Tests ----------------

    def test_rule_8_where_clause_without_null_check_rejected(self):
        """Null intent: WHERE clause with simple comparison (e.g. age > 20) must be REJECTED."""
        question = "Find all users whose email address is missing or unknown."
        sql_fail = "SELECT user_id FROM users WHERE age > 20;"
        self.assertFalse(self.validator.check_null_handling_rule(question, sql_fail))

        # Full validation pipeline check
        is_valid, violations = self.validator.validate(question, sql_fail)
        self.assertFalse(is_valid)
        self.assertIn("Null Handling Rule", violations)

    def test_rule_8_is_null_passes(self):
        """Null intent: IS NULL check passes."""
        question = "List records with missing phone numbers."
        sql_pass = "SELECT record_id FROM records WHERE phone IS NULL;"
        self.assertTrue(self.validator.check_null_handling_rule(question, sql_pass))

    def test_rule_8_is_not_null_passes(self):
        """Null intent: IS NOT NULL check passes."""
        question = "List customers where secondary email is known."
        sql_pass = "SELECT customer_id FROM customers WHERE secondary_email IS NOT NULL;"
        self.assertTrue(self.validator.check_null_handling_rule(question, sql_pass))

    def test_rule_8_not_null_passes(self):
        """Null intent: NOT NULL check passes."""
        question = "Find entries without null comments."
        sql_pass = "SELECT id FROM feedback WHERE comments NOT NULL;"
        self.assertTrue(self.validator.check_null_handling_rule(question, sql_pass))

    def test_rule_8_coalesce_passes(self):
        """Null intent: COALESCE function passes."""
        question = "List products with missing descriptions."
        sql_pass = "SELECT product_name FROM products WHERE COALESCE(description, '') = '';"
        self.assertTrue(self.validator.check_null_handling_rule(question, sql_pass))

    def test_rule_8_string_literal_null_rejected(self):
        """Null intent: comparing to literal string 'NULL' must NOT satisfy Rule 8."""
        question = "Find orders with unknown shipping date."
        sql_fail = "SELECT order_id FROM orders WHERE ship_date = 'NULL';"
        self.assertFalse(self.validator.check_null_handling_rule(question, sql_fail))

    def test_rule_8_non_null_question_ignores_rule(self):
        """Non-null question: simple WHERE comparison passes."""
        question = "Show clients older than 25."
        sql = "SELECT client_id FROM clients WHERE age > 25;"
        self.assertTrue(self.validator.check_null_handling_rule(question, sql))

    # ---------------- Edge Cases for Remaining 8 PV-SQL Rules ----------------

    def test_rule_3_average_aggregation_enforcement(self):
        """Rule 3: Average question requires AVG(), not column named average_score."""
        question = "What is the average transaction amount?"
        sql_fail = "SELECT average_amount FROM transactions;"
        self.assertFalse(self.validator.check_average_rule(question, sql_fail))
        sql_pass = "SELECT AVG(amount) FROM transactions;"
        self.assertTrue(self.validator.check_average_rule(question, sql_pass))

    def test_rule_4_count_aggregation_enforcement(self):
        """Rule 4: Count question requires COUNT(), not column named counter."""
        question = "How many accounts are currently active?"
        sql_fail = "SELECT active_counter FROM stats;"
        self.assertFalse(self.validator.check_count_rule(question, sql_fail))
        sql_pass = "SELECT COUNT(*) FROM accounts WHERE status = 'active';"
        self.assertTrue(self.validator.check_count_rule(question, sql_pass))

    def test_rule_5_sum_vs_count_intent(self):
        """Rule 5: 'total revenue' requires SUM, but 'total number of orders' is count."""
        q_sum = "What is the total revenue in 2023?"
        self.assertFalse(self.validator.check_sum_rule(q_sum, "SELECT total_revenue FROM finance;"))
        self.assertTrue(self.validator.check_sum_rule(q_sum, "SELECT SUM(revenue) FROM finance;"))

        q_count = "What is the total number of orders placed?"
        # total number -> count intent, check_sum_rule returns True (handled by count rule)
        self.assertTrue(self.validator.check_sum_rule(q_count, "SELECT COUNT(*) FROM orders;"))

    def test_rule_6_extreme_aggregation_enforcement(self):
        """Rule 6: Maximum/minimum question requires MAX/MIN or ORDER BY + LIMIT."""
        question = "What is the maximum salary in IT?"
        sql_max = "SELECT MAX(salary) FROM employees WHERE dept = 'IT';"
        sql_order = "SELECT salary FROM employees WHERE dept = 'IT' ORDER BY salary DESC LIMIT 1;"
        sql_fail = "SELECT salary FROM employees WHERE dept = 'IT';"
        self.assertTrue(self.validator.check_extreme_rule(question, sql_max))
        self.assertTrue(self.validator.check_extreme_rule(question, sql_order))
        self.assertFalse(self.validator.check_extreme_rule(question, sql_fail))

    def test_rule_7_comparison_operators(self):
        """Rule 7: Comparison question requires comparison operators in WHERE."""
        question = "Find items priced greater than 100 dollars."
        sql_fail = "SELECT item_name FROM items WHERE category = 'electronics';"
        sql_pass = "SELECT item_name FROM items WHERE price > 100;"
        sql_between = "SELECT item_name FROM items WHERE price BETWEEN 100 AND 500;"
        self.assertFalse(self.validator.check_comparison_rule(question, sql_fail))
        self.assertTrue(self.validator.check_comparison_rule(question, sql_pass))
        self.assertTrue(self.validator.check_comparison_rule(question, sql_between))

    def test_rule_9_join_validity_and_cartesian_product(self):
        """Rule 9: Join validity blocks dangling JOIN and unindexed cartesian product."""
        # Dangling JOIN without ON/USING
        sql_dangling = "SELECT * FROM clients c JOIN accounts a WHERE c.id = 1;"
        self.assertFalse(self.validator.check_join_validity_rule("Any question", sql_dangling))

        # Proper JOIN with ON
        sql_valid_join = "SELECT * FROM clients c JOIN accounts a ON c.id = a.client_id;"
        self.assertTrue(self.validator.check_join_validity_rule("Any question", sql_valid_join))

        # Comma join without WHERE (unindexed cartesian product)
        sql_cartesian = "SELECT * FROM clients, accounts;"
        self.assertFalse(self.validator.check_join_validity_rule("Any question", sql_cartesian))

        # Comma join with WHERE join condition
        sql_comma_where = "SELECT * FROM clients c, accounts a WHERE c.id = a.client_id;"
        self.assertTrue(self.validator.check_join_validity_rule("Any question", sql_comma_where))

    def test_rule_10_group_by_completeness(self):
        """Rule 10: Non-aggregated columns in SELECT must appear in GROUP BY."""
        question = "Show client id and average balance per client."
        sql_pass = "SELECT client_id, AVG(balance) FROM accounts GROUP BY client_id;"
        self.assertTrue(self.validator.check_group_by_completeness_rule(question, sql_pass))

        sql_fail = "SELECT client_id, account_type, AVG(balance) FROM accounts GROUP BY client_id;"
        self.assertFalse(self.validator.check_group_by_completeness_rule(question, sql_fail))


if __name__ == "__main__":
    unittest.main()
