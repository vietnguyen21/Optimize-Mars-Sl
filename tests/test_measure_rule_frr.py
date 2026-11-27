"""Tests for scripts/measure_rule_frr.py. Run:
    cd /content/Optimize-Mars-Sl && python -m unittest -v tests.test_measure_rule_frr
"""
import json
import os
import sqlite3
import tempfile
import unittest

from scripts import measure_rule_frr as m
from src.validation.db_probe import DBProbeEngine

MINI = "data/mini/mini_bird.jsonl"
MINI_DB = "data/mini"


def _make_db(path):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE t (id INTEGER, name TEXT)")
    con.executemany("INSERT INTO t VALUES (?, ?)", [(1, "alpha"), (2, "beta"), (3, "gamma")])
    con.commit()
    con.close()


def _write_jsonl(path, rows):
    with open(path, "wb") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False).encode("utf-8") + b"\n")


class TestMini(unittest.TestCase):
    """Numbers from commit e48441b: 24 questions, rules reject 18, 19, 23; probe rejects 0."""

    def test_mini_numbers(self):
        res, details = m._run(MINI, MINI_DB)
        self.assertEqual(res["n"] + res["n_gold_failed"], 24)
        rule_ids = {str(d["question_id"]) for d in details if d["rule_rejected"]}
        self.assertEqual(rule_ids, {"18", "19", "23"})
        self.assertEqual(res["false_rejections"], 3)
        self.assertEqual(res["probe"]["false_rejections"], 0)
        self.assertAlmostEqual(res["frr"], 3 / res["n"])
        self.assertIn("by_rule", res)
        self.assertIn("by_error", res["probe"])


class TestSynthetic(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        _make_db(os.path.join(self.dir, "d1.sqlite"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_gold_executor_distinguishes_good_and_bad(self):
        ex = m.default_gold_executor()
        db = os.path.join(self.dir, "d1.sqlite")
        self.assertTrue(ex(db, "SELECT name FROM t"))
        self.assertFalse(ex(db, "SELECT nope FROM missing_table"))

    def test_six_consecutive_not_hit_by_limit_5(self):
        # premise: without reset_counter the engine blocks after 5 queries
        p = DBProbeEngine()
        db = os.path.join(self.dir, "d1.sqlite")
        results = [p.probe_sql("SELECT name FROM t", db)[0] for _ in range(6)]
        self.assertFalse(results[5], "premise: 6th query without reset should be blocked")
        # measure_rule_frr resets before every question -> no probe rejections
        rows = [{"question_id": f"s:{i}", "db_id": "d1",
                 "question": "What are the names?", "SQL": "SELECT name FROM t"} for i in range(8)]
        path = os.path.join(self.dir, "six.jsonl")
        _write_jsonl(path, rows)
        res = m.measure_rule_frr(path, self.dir)
        self.assertEqual(res["n"], 8)
        self.assertEqual(res["probe"]["false_rejections"], 0)

    def test_replace_is_blocked_keyword(self):
        rows = [{"question_id": "r:0", "db_id": "d1", "question": "Replace letters in names?",
                 "SQL": "SELECT REPLACE(name, 'a', 'b') FROM t"}]
        path = os.path.join(self.dir, "replace.jsonl")
        _write_jsonl(path, rows)
        res = m.measure_rule_frr(path, self.dir)
        self.assertEqual(res["probe"]["false_rejections"], 1)
        self.assertEqual(res["probe"]["by_error"].get("blocked_keyword"), 1)

    def test_failed_gold_not_in_n(self):
        rows = [{"question_id": "g:0", "db_id": "d1", "question": "Bad?", "SQL": "SELECT x FROM nothing"},
                {"question_id": "g:1", "db_id": "d1", "question": "Names?", "SQL": "SELECT name FROM t"}]
        path = os.path.join(self.dir, "gold.jsonl")
        _write_jsonl(path, rows)
        res = m.measure_rule_frr(path, self.dir)
        self.assertEqual(res["n_gold_failed"], 1)
        self.assertEqual(res["n"], 1)

    def test_rows_split_on_lf_only(self):
        # U+2028 inside a string must not split a row (QUC §3.4)
        path = os.path.join(self.dir, "u2028.jsonl")
        _write_jsonl(path, [{"question_id": "u:0", "db_id": "d1", "question": "a\u2028b", "SQL": "SELECT 1"}])
        self.assertEqual(len(m.read_rows(path)), 1)

    def test_blank_middle_line_stops(self):
        path = os.path.join(self.dir, "blank.jsonl")
        with open(path, "wb") as f:
            f.write(b'{"a":1}\n\n{"a":2}\n')
        with self.assertRaises(ValueError):
            m.read_rows(path)


if __name__ == "__main__":
    unittest.main()
