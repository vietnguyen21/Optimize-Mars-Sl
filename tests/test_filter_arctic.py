"""Tests for scripts/filter_arctic.py on a tiny SQLite database (no BIRD needed)."""

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import filter_arctic as fa

SLOW_SQL = ("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 50000000) "
            "SELECT count(*) FROM c")
ROWS = [
    {"db_id": "mini", "question": "kept", "evidence": "", "SQL": "SELECT name FROM t"},
    {"db_id": "mini", "question": "empty", "evidence": "", "SQL": "SELECT name FROM t WHERE id < 0"},
    {"db_id": "mini", "question": "null row", "evidence": "", "SQL": "SELECT NULL"},
    {"db_id": "mini", "question": "error", "evidence": "", "SQL": "SELECT nope FROM missing_table"},
    {"db_id": "mini", "question": "slow", "evidence": "", "SQL": SLOW_SQL},
    {"db_id": "mini", "question": "kept 2", "evidence": "", "SQL": "SELECT count(*) FROM t"},
]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


class FilterArcticTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.db_dir = d / "db"
        self.db_dir.mkdir()
        conn = sqlite3.connect(self.db_dir / "mini.sqlite")
        conn.executescript("CREATE TABLE t (id INTEGER, name TEXT); INSERT INTO t VALUES (1, 'a'), (2, NULL);")
        conn.commit()
        conn.close()
        self.train = d / "train.jsonl"
        write_jsonl(self.train, ROWS)
        self.out = d / "filter" / "arctic_filter.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def run_filter(self, timeout_s=0.05, **kw):
        summary = fa.filter_arctic(str(self.train), str(self.db_dir), str(self.out), timeout_s=timeout_s, **kw)
        return summary, {d["question_id"]: d for d in fa.read_rows(str(self.out))}

    def test_reasons_and_counts(self):
        summary, dropped = self.run_filter()
        self.assertEqual(dropped["train:1"]["reason"], "empty")
        self.assertEqual(dropped["train:1"]["n_rows"], 0)
        self.assertEqual(dropped["train:3"]["reason"], "error")
        self.assertIsNone(dropped["train:3"]["n_rows"])
        self.assertEqual(dropped["train:4"]["reason"], "slow")
        self.assertNotIn("train:0", dropped)
        self.assertNotIn("train:2", dropped, "one row of NULL is not empty (U12)")
        self.assertEqual(summary, dict(summary, n_total=6, n_empty=1, n_slow=1, n_error=1, n_kept=3))
        self.assertEqual(summary["n_kept"] + len(dropped), summary["n_total"])
        for d in dropped.values():
            self.assertEqual(list(d), list(fa.FIELDS), "exactly the QUC 11 fields, in order")
        errors = fa.read_rows(fa.errors_path(str(self.out)))
        self.assertEqual([e["question_id"] for e in errors], ["train:3"])
        self.assertIn("missing_table", errors[0]["error"])
        self.assertFalse(Path(str(self.out) + ".progress.jsonl").exists())

    def test_slow_needs_two_interruptions(self):
        _, dropped = self.run_filter(timeout_s=0.001)
        slow = dropped["train:4"]
        self.assertEqual(slow["reason"], "slow")
        self.assertIsNotNone(slow["gold_runtime_s"])
        self.assertIsNotNone(slow["gold_runtime_rerun_s"], "slow only after a rerun (U33)")
        for qid in ("train:1", "train:3"):
            if dropped[qid]["reason"] != "slow":
                self.assertIsNone(dropped[qid]["gold_runtime_rerun_s"])

    def test_interrupted_once_then_ok_is_kept(self):
        calls = []

        def fake(sql, db_path, timeout_s):
            calls.append(sql)
            if sql == ROWS[0]["SQL"] and calls.count(sql) == 1:
                return False, "interrupted", timeout_s
            return fa.execute_query_safely(sql, db_path, timeout=5.0) + (0.001,)

        with mock.patch.object(fa, "_run", side_effect=fake):
            summary, dropped = self.run_filter(timeout_s=5.0, limit=1)
        self.assertEqual(calls.count(ROWS[0]["SQL"]), 2)
        self.assertEqual(dropped, {})
        self.assertEqual(summary["n_kept"], 1)
        self.assertEqual(summary["n_rerun"], 1)
        self.assertEqual(summary["n_rerun_kept"], 1)

    def test_rerun_error_is_error(self):
        side = [(False, "interrupted", 5.0), (False, "no such table: x", 0.1)]
        with mock.patch.object(fa, "_run", side_effect=side):
            _, dropped = self.run_filter(timeout_s=5.0, limit=1)
        self.assertEqual(dropped["train:0"]["reason"], "error")
        self.assertEqual(dropped["train:0"]["gold_runtime_rerun_s"], 0.1)

    def test_row_ids_kept_after_filter(self):
        self.run_filter()
        out = Path(self.tmp.name) / "train_filtered.jsonl"
        self.assertEqual(fa.write_filtered(str(self.train), str(self.out), str(out)), 3)
        kept = fa.read_rows(str(out))
        self.assertEqual([r["question_id"] for r in kept], ["train:0", "train:2", "train:5"])
        self.assertEqual([r["question"] for r in kept], ["kept", "null row", "kept 2"])
        sft = Path(self.tmp.name) / "sft_train.jsonl"
        write_jsonl(sft, [{"db_id": r["db_id"], "question": r["question"], "answer": r["SQL"]} for r in ROWS])
        sft_out = Path(self.tmp.name) / "sft_filtered.jsonl"
        fa.write_filtered(str(sft), str(self.out), str(sft_out))
        self.assertEqual([r["question_id"] for r in fa.read_rows(str(sft_out))], ["sft_train:0", "sft_train:2", "sft_train:5"])

    def test_write_filtered_rejects_other_line_order(self):
        self.run_filter()
        other = Path(self.tmp.name) / "other.jsonl"
        write_jsonl(other, [dict(r, db_id="x") for r in ROWS])
        with self.assertRaises(ValueError):
            fa.write_filtered(str(other), str(self.out), str(Path(self.tmp.name) / "o.jsonl"))

    def test_resume_after_interruption(self):
        real = fa.check_gold
        n = {"calls": 0}

        def dies_after_two(*a):
            n["calls"] += 1
            if n["calls"] > 2:
                raise KeyboardInterrupt
            return real(*a)

        with mock.patch.object(fa, "check_gold", side_effect=dies_after_two):
            with self.assertRaises(KeyboardInterrupt):
                self.run_filter()
        progress = Path(str(self.out) + ".progress.jsonl")
        self.assertTrue(progress.exists())
        with open(progress, "ab") as f:
            f.write(b'{"i": 2, "reas')  # line cut by a disconnect
        with mock.patch.object(fa, "check_gold", wraps=real) as again:
            summary, _ = self.run_filter()
        self.assertEqual(again.call_count, 4)
        self.assertEqual(summary["n_kept"], 3)

    def test_row_ids_split_on_lf_only(self):
        rows = [dict(ROWS[0], question="a\u2028b"), dict(ROWS[5], question="c\rd")]
        write_jsonl(self.train, rows)
        self.assertEqual(len(fa.read_rows(str(self.train))), 2)
        with open(self.train, "ab") as f:
            f.write(b"\n" + json.dumps(ROWS[0]).encode() + b"\n")
        with self.assertRaises(ValueError):
            fa.read_rows(str(self.train))

    def test_missing_database_stops(self):
        write_jsonl(self.train, [dict(ROWS[0], db_id="absent")])
        with self.assertRaises(FileNotFoundError):
            self.run_filter()


if __name__ == "__main__":
    unittest.main()
