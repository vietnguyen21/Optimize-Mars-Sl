"""
T23: PV-SQL rules and probe timeouts are soft signals by default; hard_gate=True keeps the strict verdict.
"""

import os
import sqlite3
import tempfile
import unittest

from src.common.schemas import SQLCandidate
from src.validation import DBProbeEngine, validate_candidate

# Never terminates on its own, so only the probe deadline can stop it.
ENDLESS_SQL = (
    "SELECT COUNT(*) FROM (WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) "
    "SELECT x FROM c)"
)


class _FixedScorer:
    threshold = 0.75

    def __init__(self, score):
        self.score = score

    def score_candidate(self, question, schema_ddl, sql):
        return self.score


class _ViolatingValidator:
    def validate(self, question, sql, schema=None):
        return False, ["Rule 3 (aggregation mismatch)"]


class _PassingValidator:
    def validate(self, question, sql, schema=None):
        return True, []


class TestValidatorSoftGate(unittest.TestCase):
    def setUp(self):
        handle = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
        handle.close()
        self.db_path = handle.name
        conn = sqlite3.connect(self.db_path)
        conn.execute("CREATE TABLE t (a TEXT)")
        conn.execute("INSERT INTO t VALUES ('1,234')")
        conn.commit()
        conn.close()

    def tearDown(self):
        os.unlink(self.db_path)

    def _validate(self, sql, validator, score, probe=None, **kwargs):
        return validate_candidate(
            candidate=SQLCandidate(candidate_id=0, sql=sql),
            question="How many rows are in t?",
            db_path=self.db_path,
            validator=validator,
            probe_engine=probe or DBProbeEngine(),
            scorer=_FixedScorer(score),
            **kwargs,
        )

    def test_violation_with_high_score_accepted_by_default(self):
        res = self._validate("SELECT a FROM t", _ViolatingValidator(), 0.9)
        self.assertEqual(res.final_verdict, "ACCEPT")
        self.assertFalse(res.pv_sql_passed)
        self.assertIn("PV-SQL violations", res.rejection_reason)

    def test_violation_rejected_with_hard_gate(self):
        res = self._validate("SELECT a FROM t", _ViolatingValidator(), 0.9, hard_gate=True)
        self.assertEqual(res.final_verdict, "REJECT")
        self.assertIn("PV-SQL violations", res.rejection_reason)

    def test_low_score_still_rejected_by_default(self):
        res = self._validate("SELECT a FROM t", _PassingValidator(), 0.1)
        self.assertEqual(res.final_verdict, "REJECT")

    def test_real_execution_error_still_rejected_by_default(self):
        res = self._validate("SELECT missing_col FROM t", _PassingValidator(), 0.9)
        self.assertEqual(res.final_verdict, "REJECT")
        self.assertIn("DB probe error", res.rejection_reason)

    def test_probe_timeout_reported_as_timeout_kind(self):
        ok, rows, err = DBProbeEngine(timeout=0.05).probe_sql(ENDLESS_SQL, self.db_path)
        self.assertFalse(ok)
        self.assertTrue(err.startswith(DBProbeEngine.TIMEOUT_ERROR_PREFIX), err)

    def test_probe_timeout_not_rejected_by_default(self):
        res = self._validate(ENDLESS_SQL, _PassingValidator(), 0.9, probe=DBProbeEngine(timeout=0.05))
        self.assertEqual(res.final_verdict, "ACCEPT")
        self.assertFalse(res.probe_passed)
        self.assertIn("timeout", res.rejection_reason)

    def test_probe_timeout_rejected_with_hard_gate(self):
        res = self._validate(ENDLESS_SQL, _PassingValidator(), 0.9,
                             probe=DBProbeEngine(timeout=0.05), hard_gate=True)
        self.assertEqual(res.final_verdict, "REJECT")

    def test_replace_function_passes_probe(self):
        ok, rows, err = DBProbeEngine().probe_sql("SELECT CAST(REPLACE(a, ',', '') AS INT) FROM t", self.db_path)
        self.assertTrue(ok, err)
        self.assertEqual(rows, [(1234,)])
        ok, _, err = DBProbeEngine().probe_sql("SELECT replace (a, ',', '') FROM t", self.db_path)
        self.assertTrue(ok, err)

    def test_statement_replace_still_blocked(self):
        for sql in ("REPLACE INTO t VALUES (1)", "replace into t values (1)", "INSERT OR REPLACE INTO t VALUES (1)"):
            ok, _, err = DBProbeEngine().probe_sql(sql, self.db_path)
            self.assertFalse(ok, sql)
            self.assertIn("Destructive operation blocked", err)


if __name__ == "__main__":
    unittest.main()
