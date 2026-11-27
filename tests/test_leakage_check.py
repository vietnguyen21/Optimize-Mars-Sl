"""Tests for scripts/leakage_check.py on tiny JSONL files."""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts import leakage_check as lc

QUC_FIELDS = {"pairs_checked", "exact_question_dup", "exact_sql_dup", "near_dup_threshold", "near_dups",
              "gold_in_prompt_hits"}


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


class LeakageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.train, self.cal, self.test = d / "train.jsonl", d / "cal_b.jsonl", d / "test.jsonl"
        write_jsonl(self.train, [
            {"db_id": "a", "question": "How many  Books are there?", "evidence": "", "SQL": "SELECT count(*) FROM book"},
            {"db_id": "a", "question": "list the names of all the red cars in the city of hanoi today",
             "evidence": "", "SQL": "SELECT name FROM car WHERE color = 'red'"},
            {"db_id": "a", "question": "Leaky one", "evidence": "use SELECT 1 FROM t", "SQL": "select 1  from t"},
        ])
        write_jsonl(self.cal, [
            {"db_id": "b", "question": "how many books are there?", "evidence": "", "SQL": "SELECT title FROM b"},
            {"db_id": "b", "question": "list the names of all the red cars in the city of hanoi", "evidence": "",
             "SQL": "SELECT  COUNT(*) FROM book"},
            {"db_id": "b", "question": "nothing alike here", "evidence": "", "SQL": "SELECT 2"},
        ])
        write_jsonl(self.test, [{"question_id": 0, "db_id": "c", "question": "unrelated", "evidence": "", "SQL": "SELECT 3"}])

    def tearDown(self):
        self.tmp.cleanup()

    def report(self, **kw):
        return lc.leakage_report(str(self.train), [str(self.cal), str(self.test)], **kw)

    def test_fields_and_counts(self):
        r = self.report()
        self.assertTrue(QUC_FIELDS <= set(r))
        self.assertEqual(r["pairs_checked"], 3 * 3 + 3 * 1)
        self.assertEqual(r["exact_question_dup"], 1)
        self.assertEqual(r["exact_question_examples"][0]["train_id"], "train:0")
        self.assertEqual(r["exact_question_examples"][0]["eval_id"], "cal_b:0")
        self.assertEqual(r["exact_sql_dup"], 1)
        self.assertEqual(r["gold_in_prompt_hits"], 1)
        self.assertEqual(r["gold_in_prompt_examples"][0]["question_id"], "train:2")
        self.assertEqual(r["near_dup_threshold"], 0.9)

    def test_near_dup_jaccard(self):
        pairs = {(x["train_id"], x["eval_id"]): x["jaccard"] for x in self.report()["near_dups"]}
        self.assertEqual(pairs[("train:0", "cal_b:0")], 1.0)
        self.assertAlmostEqual(pairs[("train:1", "cal_b:1")], 10 / 11, places=4)  # 10 shared of 11 words
        strict = {(x["train_id"], x["eval_id"]) for x in self.report(near_dup_threshold=0.95)["near_dups"]}
        self.assertEqual(strict, {("train:0", "cal_b:0")})

    def test_length_filter_matches_brute_force(self):
        train, ev = lc.read_rows(str(self.train)), lc.read_rows(str(self.cal))
        for t in (0.3, 0.5, 0.9, 1.0):
            fast = {(x["train_id"], x["eval_id"]) for x in lc._near_dups(
                train, [f"train:{i}" for i in range(3)], ev, [f"cal_b:{i}" for i in range(3)], t)}
            slow = set()
            for i, a in enumerate(train):
                for k, b in enumerate(ev):
                    A, B = lc.tokens(a["question"]), lc.tokens(b["question"])
                    if len(A & B) / len(A | B) >= t:
                        slow.add((f"train:{i}", f"cal_b:{k}"))
            self.assertEqual(fast, slow, t)

    def test_schema_from_db_dir(self):
        db_dir = Path(self.tmp.name) / "db"
        db_dir.mkdir()
        conn = sqlite3.connect(db_dir / "c.sqlite")
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.commit()
        conn.close()
        write_jsonl(self.cal, [{"db_id": "c", "question": "q", "evidence": "", "SQL": "CREATE TABLE t (x INTEGER)"}])
        self.assertEqual(self.report()["eval_files"]["cal_b.jsonl"]["gold_in_prompt_hits"], 0)
        r = self.report(db_dir=str(db_dir))
        self.assertTrue(r["schema_checked"])
        self.assertEqual(r["eval_files"]["cal_b.jsonl"]["gold_in_prompt_hits"], 1)

    def test_bird_dev_text_only(self):
        write_jsonl(self.test, [{"db_id": "c", "question": "q", "evidence": "SELECT 3", "SQL": "SELECT 3"}])
        r = self.report()
        self.assertIsNone(r["eval_files"]["test.jsonl"]["gold_in_prompt_hits"], "BIRD-dev: hash and text comparison only")
        self.assertEqual(r["gold_in_prompt_hits"], 1)  # train:2 only

    def test_shared_db_ids_listed(self):
        self.assertEqual(self.report()["eval_files"]["cal_b.jsonl"]["shared_db_ids"], [])


if __name__ == "__main__":
    unittest.main()
