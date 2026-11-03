"""
Adversarial Challenge: PV-SQL 10 AST Semantic Validation Rules
Tests corner cases, trick questions, missing conditions, join validity,
and group-by completeness.
"""

import sys
import os
import unittest

REPO_ROOT = "/mnt/c/Workspace/optimize"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.validation.pv_sql_rules import PVSQLValidator


class TestChallengePVSQLRules(unittest.TestCase):

    def setUp(self):
        self.validator = PVSQLValidator()

    def test_rule_1_distinct_adversarial(self):
        """Rule 1 Distinct: test subtle triggers (unique, different, individual) and evasion."""
        # Violations (Missing DISTINCT / GROUP BY)
        fail_pairs = [
            ("List all unique branch names.", "SELECT branch_name FROM branches;"),
            ("Find distinct customer types.", "SELECT cust_type FROM customers;"),
            ("What different account frequencies exist?", "SELECT frequency FROM accounts;"),
            ("Show individual district codes.", "SELECT district_id FROM districts;"),
        ]
        for q, sql in fail_pairs:
            passed, violations, details = self.validator.validate_rules(q, sql)
            self.assertFalse(passed, f"Expected failure for: {q}")
            self.assertIn("Distinct Rule", violations)
            self.assertFalse(details["Distinct Rule"])

        # Successes (Compliant)
        pass_pairs = [
            ("List unique branch names.", "SELECT DISTINCT branch_name FROM branches;"),
            ("Find distinct customer types.", "SELECT cust_type FROM customers GROUP BY cust_type;"),
            ("Show all account frequencies.", "SELECT frequency FROM accounts;"),  # No distinct intent
        ]
        for q, sql in pass_pairs:
            passed, violations, details = self.validator.validate_rules(q, sql)
            self.assertTrue(details["Distinct Rule"], f"Expected pass for Distinct Rule: {q}")

    def test_rule_2_top_k_adversarial(self):
        """Rule 2 Top-K: test ranking keywords and queries missing ORDER BY / LIMIT."""
        fail_pairs = [
            ("Which school has the highest SAT score?", "SELECT school_name FROM schools;"),
            ("Find the lowest temperature recorded.", "SELECT temp FROM weather;"),
            ("What is the top performing district?", "SELECT district_name FROM districts;"),
            ("Show the least populated county.", "SELECT county FROM counties;"),
            ("Who was the first student registered?", "SELECT name FROM students;"),
            ("Which client has the worst credit rating?", "SELECT client_id FROM clients;"),
        ]
        for q, sql in fail_pairs:
            passed, violations, details = self.validator.validate_rules(q, sql)
            self.assertFalse(passed, f"Expected failure for: {q}")
            self.assertIn("Top-K Rule", violations)

        pass_pairs = [
            ("Which school has the highest SAT score?", "SELECT school_name FROM schools ORDER BY score DESC LIMIT 1;"),
            ("What is the top performing district?", "SELECT district_name FROM districts LIMIT 1;"),
            ("Show all schools.", "SELECT school_name FROM schools;"),  # No ranking intent
        ]
        for q, sql in pass_pairs:
            passed, violations, details = self.validator.validate_rules(q, sql)
            self.assertTrue(details["Top-K Rule"], f"Expected pass for Top-K Rule: {q}")

    def test_rule_3_average_adversarial(self):
        """Rule 3 Average: test average/mean/avg intent vs missing AVG()."""
        fail_pairs = [
            ("What is the average salary of teachers?", "SELECT salary FROM teachers;"),
            ("Calculate the mean score in math.", "SELECT math_score FROM exams;"),
            ("What is the avg balance of gold accounts?", "SELECT balance FROM accounts;"),
        ]
        for q, sql in fail_pairs:
            passed, violations, details = self.validator.validate_rules(q, sql)
            self.assertFalse(passed, f"Expected failure for: {q}")
            self.assertIn("Average Rule", violations)

        pass_pairs = [
            ("What is the average salary of teachers?", "SELECT AVG(salary) FROM teachers;"),
            ("Calculate the mean score in math.", "SELECT AVG (math_score) FROM exams;"),
            ("List teacher salaries.", "SELECT salary FROM teachers;"),  # No average intent
        ]
        for q, sql in pass_pairs:
            passed, violations, details = self.validator.validate_rules(q, sql)
            self.assertTrue(details["Average Rule"], f"Expected pass for Average Rule: {q}")

    def test_rule_4_count_adversarial(self):
        """Rule 4 Count: test how many/number of/count of intent vs missing COUNT()."""
        fail_pairs = [
            ("How many accounts are registered in Prague?", "SELECT account_id FROM accounts;"),
            ("What is the number of male clients?", "SELECT client_id FROM clients WHERE gender = 'M';"),
            ("Give the count of transactions in 1997.", "SELECT trans_id FROM trans;"),
            ("What is the quantity of orders processed?", "SELECT order_id FROM orders;"),
        ]
        for q, sql in fail_pairs:
            passed, violations, details = self.validator.validate_rules(q, sql)
            self.assertFalse(passed, f"Expected failure for: {q}")
            self.assertIn("Count Rule", violations)

        pass_pairs = [
            ("How many accounts are registered in Prague?", "SELECT COUNT(*) FROM accounts;"),
            ("What is the number of male clients?", "SELECT COUNT (client_id) FROM clients;"),
        ]
        for q, sql in pass_pairs:
            passed, violations, details = self.validator.validate_rules(q, sql)
            self.assertTrue(details["Count Rule"], f"Expected pass for Count Rule: {q}")

    def test_rule_5_sum_adversarial(self):
        """Rule 5 Sum: test total/sum vs SUM(), and distinction from total count."""
        fail_pairs = [
            ("What is the total balance across all accounts?", "SELECT balance FROM accounts;"),
            ("Calculate the sum of all payments.", "SELECT payment FROM payments;"),
            ("What is the sum of loan debts?", "SELECT amount FROM loans;"),
        ]
        for q, sql in fail_pairs:
            passed, violations, details = self.validator.validate_rules(q, sql)
            self.assertFalse(passed, f"Expected failure for: {q}")
            self.assertIn("Sum Rule", violations)

        # "total number of" should not require SUM()
        ok_total_num, violations_tn, details_tn = self.validator.validate_rules(
            "What is the total number of accounts?",
            "SELECT COUNT(*) FROM accounts;"
        )
        self.assertTrue(details_tn["Sum Rule"], "total number should be handled by Count rule, not require SUM")

        pass_pairs = [
            ("What is the total balance across all accounts?", "SELECT SUM(balance) FROM accounts;"),
            ("Calculate the sum of all payments.", "SELECT SUM (payment) FROM payments;"),
        ]
        for q, sql in pass_pairs:
            passed, violations, details = self.validator.validate_rules(q, sql)
            self.assertTrue(details["Sum Rule"], f"Expected pass for Sum Rule: {q}")

    def test_rule_6_extreme_adversarial(self):
        """Rule 6 Extreme: test maximum/minimum/max/min/oldest/latest."""
        fail_pairs = [
            ("Find the maximum credit card limit.", "SELECT card_limit FROM cards;"),
            ("What is the minimum loan term?", "SELECT duration FROM loans;"),
            ("Who is the oldest client?", "SELECT birth_date FROM clients;"),
            ("What is the latest transaction date?", "SELECT date FROM trans;"),
        ]
        for q, sql in fail_pairs:
            passed, violations, details = self.validator.validate_rules(q, sql)
            self.assertFalse(passed, f"Expected failure for: {q}")
            self.assertIn("Extreme Rule", violations)

        pass_pairs = [
            ("Find the maximum credit card limit.", "SELECT MAX(card_limit) FROM cards;"),
            ("What is the minimum loan term.", "SELECT MIN(duration) FROM loans;"),
            ("Who is the oldest client?", "SELECT birth_date FROM clients ORDER BY birth_date ASC LIMIT 1;"),
        ]
        for q, sql in pass_pairs:
            passed, violations, details = self.validator.validate_rules(q, sql)
            self.assertTrue(details["Extreme Rule"], f"Expected pass for Extreme Rule: {q}")

    def test_rule_7_comparison_adversarial(self):
        """Rule 7 Comparison: test comparison intent vs comparison operators in WHERE."""
        fail_pairs = [
            ("List accounts with balance greater than 10000.", "SELECT account_id FROM accounts;"),
            ("Find transactions occurring after 1996.", "SELECT trans_id FROM trans;"),
            ("Show clients younger than 30 years old.", "SELECT client_id FROM clients;"),
            ("List loans exceeding 50000.", "SELECT loan_id FROM loans;"),
        ]
        for q, sql in fail_pairs:
            passed, violations, details = self.validator.validate_rules(q, sql)
            self.assertFalse(passed, f"Expected failure for: {q}")
            self.assertIn("Comparison Rule", violations)

        pass_pairs = [
            ("List accounts with balance greater than 10000.", "SELECT account_id FROM accounts WHERE balance > 10000;"),
            ("Find transactions occurring after 1996.", "SELECT trans_id FROM trans WHERE date >= '1996-01-01';"),
            ("Show loans between 1000 and 5000.", "SELECT loan_id FROM loans WHERE amount BETWEEN 1000 AND 5000;"),
        ]
        for q, sql in pass_pairs:
            passed, violations, details = self.validator.validate_rules(q, sql)
            self.assertTrue(details["Comparison Rule"], f"Expected pass for Comparison Rule: {q}")

    def test_rule_8_null_handling_adversarial(self):
        """Rule 8 Null Handling: test null/missing/known intent vs IS NULL / IS NOT NULL."""
        fail_pairs = [
            ("List clients with missing birth date.", "SELECT client_id FROM clients;"),
            ("Show cards where status is not null.", "SELECT card_id FROM cards;"),
            ("Which districts have unknown unemployment rate?", "SELECT district_name FROM districts;"),
        ]
        for q, sql in fail_pairs:
            passed, violations, details = self.validator.validate_rules(q, sql)
            self.assertFalse(passed, f"Expected failure for: {q}")
            self.assertIn("Null Handling Rule", violations)

        pass_pairs = [
            ("List clients with missing birth date.", "SELECT client_id FROM clients WHERE birth_date IS NULL;"),
            ("Show cards where status is not null.", "SELECT card_id FROM cards WHERE status IS NOT NULL;"),
        ]
        for q, sql in pass_pairs:
            passed, violations, details = self.validator.validate_rules(q, sql)
            self.assertTrue(details["Null Handling Rule"], f"Expected pass for Null Handling Rule: {q}")

    def test_rule_9_join_validity_adversarial(self):
        """Rule 9 Join Validity: test dangling JOIN without ON/USING, and Cartesian FROM A, B without WHERE."""
        fail_queries = [
            "SELECT * FROM accounts JOIN clients;",
            "SELECT * FROM accounts JOIN clients WHERE accounts.id = 1;",  # JOIN without ON
            "SELECT * FROM accounts, clients;",  # Unindexed Cartesian product
            "SELECT * FROM accounts, clients WHERE accounts.id > 10;",  # No join predicate linking them
        ]
        for sql in fail_queries:
            passed, violations, details = self.validator.validate_rules("Dummy question", sql)
            self.assertFalse(details["Join Validity Rule"], f"Expected Join Validity failure for: {sql}")
            self.assertIn("Join Validity Rule", violations)

        pass_queries = [
            "SELECT * FROM accounts JOIN clients ON accounts.client_id = clients.client_id;",
            "SELECT * FROM accounts LEFT JOIN clients USING (client_id);",
            "SELECT * FROM accounts, clients WHERE accounts.client_id = clients.client_id;",
            "SELECT * FROM accounts;",  # Single table, no join
        ]
        for sql in pass_queries:
            passed, violations, details = self.validator.validate_rules("Dummy question", sql)
            self.assertTrue(details["Join Validity Rule"], f"Expected Join Validity pass for: {sql}")

    def test_rule_10_group_by_completeness_adversarial(self):
        """Rule 10 Group By Completeness: non-aggregated columns in SELECT must appear in GROUP BY."""
        fail_queries = [
            "SELECT client_id, district_id, COUNT(*) FROM clients GROUP BY client_id;",
            "SELECT accounts.account_id, clients.gender, SUM(balance) FROM accounts JOIN clients ON accounts.client_id = clients.client_id GROUP BY accounts.account_id;",
            "SELECT district_name, county, AVG(salary) FROM schools GROUP BY district_name;",
        ]
        for sql in fail_queries:
            passed, violations, details = self.validator.validate_rules("Dummy question", sql)
            self.assertFalse(details["Group By Completeness Rule"], f"Expected Group By Completeness failure for: {sql}")
            self.assertIn("Group By Completeness Rule", violations)

        pass_queries = [
            "SELECT client_id, district_id, COUNT(*) FROM clients GROUP BY client_id, district_id;",
            "SELECT district_id, COUNT(*) FROM clients GROUP BY district_id;",
            "SELECT COUNT(*), AVG(balance) FROM accounts GROUP BY district_id;",  # Pure aggregations
            "SELECT 1, COUNT(*) FROM accounts GROUP BY district_id;",  # Numeric literal constant
            "SELECT 1, SUM(amount) FROM loans GROUP BY status;",
            "SELECT client_id FROM clients;",  # No GROUP BY
        ]
        for sql in pass_queries:
            passed, violations, details = self.validator.validate_rules("Dummy question", sql)
            self.assertTrue(details["Group By Completeness Rule"], f"Expected Group By Completeness pass for: {sql}")

    def test_rule_10_string_literal_edge_case(self):
        """Document edge case: string literal in projection without GROUP BY."""
        sql = "SELECT 'Constant', COUNT(*) FROM accounts GROUP BY district_id;"
        passed, violations, details = self.validator.validate_rules("Dummy question", sql)
        # Empirical finding: string literal 'Constant' is parsed as column identifier 'CONSTANT'
        # which fails because 'CONSTANT' is not in GROUP BY clause.
        self.assertFalse(details["Group By Completeness Rule"])


if __name__ == "__main__":
    unittest.main()
