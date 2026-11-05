"""Tests QueryPlan extraction and its optional place in the prompt."""

import unittest

try:
    import sqlglot  # noqa: F401
    HAS_SQLGLOT = True
except ImportError:
    HAS_SQLGLOT = False

from src.evaluation.candidate_generator import build_prompt


@unittest.skipUnless(HAS_SQLGLOT, "needs sqlglot")
class TestExtractPlan(unittest.TestCase):
    def _plan(self, sql):
        from src.common.query_plan import extract_plan
        return extract_plan(sql)

    def test_document_example(self):
        plan = self._plan("SELECT c.id, COUNT(o.id) AS n FROM customers AS c JOIN orders AS o ON o.customer_id = c.id "
                          "GROUP BY c.id ORDER BY n DESC LIMIT 5;")
        self.assertEqual(plan, {"aggregation": "COUNT", "group_by": True, "having": False, "order_by": True,
                                "direction": "DESC", "limit": True, "distinct": False, "subquery": False})

    def test_inner_aggregation_is_not_outer(self):
        plan = self._plan("SELECT name FROM t WHERE x > (SELECT AVG(x) FROM t);")
        self.assertEqual(plan["aggregation"], "NONE")
        self.assertTrue(plan["subquery"])

    def test_distinct_and_having(self):
        self.assertTrue(self._plan("SELECT DISTINCT a FROM t;")["distinct"])
        self.assertTrue(self._plan("SELECT COUNT(DISTINCT a) FROM t;")["distinct"])
        self.assertTrue(self._plan("SELECT a FROM t GROUP BY a HAVING COUNT(*) > 1;")["having"])


class TestPlanPrompt(unittest.TestCase):
    def test_plan_is_optional(self):
        self.assertNotIn("<plan>", build_prompt("q", "", "ddl"))
        self.assertIn('<plan>{"limit":true}</plan>', build_prompt("q", "", "ddl", '{"limit":true}'))


if __name__ == "__main__":
    unittest.main()
