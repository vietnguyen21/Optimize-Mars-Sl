"""Tests for src/evaluation/generator_only.py (CPU only, mini databases, fake generators).

  python -m unittest -v tests.test_generator_only
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation import generator_only as go
from src.training.runtime import load_records

DB_DIR = "data/mini"
MINI = "data/mini/mini_bird.jsonl"
N = 8  # the first 8 mini rows are all financial_mini


def full_fn(row, row_id):
    return go.full_schema(row, DB_DIR)


def meta(**extra):
    return dict({"baseline": "b0", "split": "mini", "checkpoint": "fake", "commit": "abc", "seed": 42,
                 "prompt_format": "v2", "gpu": None}, **extra)


def result(sql, overlength=False):
    return {"sql": sql, "tokens_in": 10, "tokens_out": 5, "overlength": overlength}


class Cal300Test(unittest.TestCase):
    def test_reference_values_from_quc(self):
        idx = go.cal_a_300_indices()
        self.assertEqual(len(idx), 300)
        self.assertEqual(idx[:5], [0, 1, 3, 6, 7])
        self.assertEqual(idx[-5:], [657, 658, 659, 660, 661])
        text = go.cal_a_300_ids_text(idx)
        self.assertTrue(text.endswith("\n") and text.startswith("retrieval_dev:0\n"))
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), go.CAL_A_300_IDS_SHA256)
        go.check_cal_a_300_reference(idx, text)

    def test_reference_mismatch_stops(self):
        idx = go.cal_a_300_indices()
        with self.assertRaises(ValueError):
            go.check_cal_a_300_reference(idx[:-1] + [660], go.cal_a_300_ids_text(idx))

    def test_lf_only_split_keeps_u2028_and_rejects_blank_middle(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a.jsonl"
            p.write_bytes('{"q": "a\u2028b"}\n{"q": "c"}\n'.encode())
            rows = go.read_jsonl_lf(p)
            self.assertEqual([r["q"] for r in rows], ["a\u2028b", "c"])
            p.write_bytes(b'{"q": 1}\n\n{"q": 2}\n')
            with self.assertRaises(ValueError):
                go.read_jsonl_lf(p)


class SchemaTest(unittest.TestCase):
    def test_sprime_is_verbatim_subset(self):
        db = f"{DB_DIR}/financial_mini.sqlite"
        sub = go.sprime_schema(db, ["clients", "districts"])
        self.assertIn("CREATE TABLE", sub)
        self.assertNotIn("transactions", sub)
        row = {"db_id": "financial_mini"}
        self.assertTrue(set(sub.split("\n\n")) <= set(go.full_schema(row, DB_DIR).split("\n\n")))

    def test_sprime_unknown_table_raises(self):
        with self.assertRaises(ValueError):
            go.sprime_schema(f"{DB_DIR}/financial_mini.sqlite", ["no_such_table"])


class ExtractionTest(unittest.TestCase):
    def test_extract_sql_formats(self):
        self.assertEqual(go.extract_sql_from_generation("reasoning\n<answer>SELECT 1</answer>"), "SELECT 1")
        self.assertEqual(go.extract_sql_from_generation("<answer>SELECT * FROM t"), "SELECT * FROM t")
        self.assertEqual(go.extract_sql_from_generation("Here is SQL:\n```sql\nSELECT 2;\n```"), "SELECT 2;")
        self.assertEqual(go.extract_sql_from_generation("No valid tags here"), "")


class StatsTest(unittest.TestCase):
    def test_p90_is_ceil_09n_th_value(self):
        self.assertEqual(go.percentile_ceil(list(range(1, 11))), 9)   # ceil(9.0) = 9th
        self.assertEqual(go.percentile_ceil(list(range(1, 12))), 10)  # ceil(9.9) = 10th
        self.assertEqual(go.percentile_ceil([5.0]), 5.0)


class EvaluateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = str(Path(self.tmp.name) / "run")
        rows = load_records(MINI)
        self.records = rows[:N]
        self.gold = {r["question"]: r["SQL"] for r in self.records}

    def tearDown(self):
        self.tmp.cleanup()

    def run_eval(self, fn, out=None, schema_fn=full_fn, indices=None, **kw):
        return go.evaluate_generator_only(MINI, indices if indices is not None else list(range(N)), DB_DIR, fn,
                                          schema_fn, out or self.out, kw.pop("meta", meta()), log_every=0, **kw)

    def test_oracle_scores_100_and_files_follow_quc_s11(self):
        m = self.run_eval(lambda q, s, e: result(self.gold[q]))
        self.assertEqual(m["ex"], 100.0)
        self.assertEqual(m["n"], N)
        self.assertEqual(m["timeout_s"], 30.0)
        self.assertIsNone(m["ves"])
        self.assertEqual(m["n_overlength"], 0)
        self.assertEqual(set(m), {"baseline", "split", "n", "ex", "ves", "timeout_s", "checkpoint", "commit",
                                  "seed", "n_overlength"})
        rows = [json.loads(l) for l in (Path(self.out) / "predictions.jsonl").read_text().splitlines()]
        self.assertEqual(list(rows[0]), go.PREDICTION_FIELDS)
        self.assertEqual([r["question_id"] for r in rows], [f"mini_bird:{i}" for i in range(N)])  # stem:line
        self.assertTrue(all(r["peak_vram_bytes"] is None for r in rows))  # CPU: null, not 0
        cost = json.loads((Path(self.out) / "cost.json").read_text())
        self.assertEqual(set(cost), {"calls_per_q", "tokens_per_q", "latency_median_s", "latency_p90_s",
                                     "peak_vram_gb", "ex_per_gb", "ex_per_s"})
        self.assertIsNone(cost["peak_vram_gb"])
        self.assertIsNone(cost["ex_per_gb"])
        self.assertEqual(cost["tokens_per_q"], 15)
        self.assertEqual(cost["calls_per_q"], 1)

    def test_row_id_uses_original_file_line_not_position(self):
        self.run_eval(lambda q, s, e: result(""), indices=[5, 2, 7])
        rows = [json.loads(l) for l in (Path(self.out) / "predictions.jsonl").read_text().splitlines()]
        self.assertEqual([r["question_id"] for r in rows], ["mini_bird:5", "mini_bird:2", "mini_bird:7"])

    def test_limit_takes_first_n_of_the_indices(self):
        m = self.run_eval(lambda q, s, e: result(""), limit=3)
        self.assertEqual(m["n"], 3)

    def test_empty_sql_is_wrong_but_n_is_kept(self):
        m = self.run_eval(lambda q, s, e: result(""))
        self.assertEqual((m["ex"], m["n"]), (0.0, N))

    def test_overlength_is_counted_wrong_and_not_dropped(self):
        first = self.records[0]["question"]
        m = self.run_eval(lambda q, s, e: result("SELECT 1", True) if q == first else result(self.gold[q]))
        self.assertEqual(m["n_overlength"], 1)
        self.assertEqual(m["n"], N)
        self.assertAlmostEqual(m["ex"], 100.0 * (N - 1) / N)
        first_row = json.loads((Path(self.out) / "predictions.jsonl").read_text().splitlines()[0])
        self.assertEqual((first_row["pred_sql"], first_row["ex"], first_row["n_calls"]), ("", False, 0))

    def test_execution_error_is_wrong(self):
        m = self.run_eval(lambda q, s, e: result("SELECT nope FROM nowhere"))
        self.assertEqual(m["ex"], 0.0)
        rows = [json.loads(l) for l in (Path(self.out) / "predictions.jsonl").read_text().splitlines()]
        self.assertTrue(all(r["exec_ok"] is False for r in rows))

    def test_generator_never_sees_gold(self):
        seen = []

        def fn(q, s, e):
            seen.append((q, s, e))
            return result("")

        self.run_eval(fn)
        self.assertEqual(len(seen), N)
        for (q, s, e), rec in zip(seen, self.records):
            self.assertNotIn(rec["SQL"], s + e + q)
            self.assertEqual((q, e), (rec["question"], rec.get("evidence", "")))

    def test_partial_file_holds_no_gold_and_no_ex(self):
        self.run_eval(lambda q, s, e: result(self.gold[q]))
        line = json.loads((Path(self.out) / "generations.partial.jsonl").read_text().splitlines()[0])
        self.assertNotIn("ex", line)
        self.assertNotIn("SQL", line)
        self.assertEqual(set(line), {"question_id", "db_id", "pred_sql", "n_calls", "tokens_in", "tokens_out",
                                     "latency_s", "peak_vram_bytes", "overlength"})

    def test_bad_generator_result_keys_raise(self):
        with self.assertRaises(ValueError):
            self.run_eval(lambda q, s, e: {"sql": "SELECT 1"})
        with self.assertRaises(ValueError):  # the old n_calls key is no longer part of the contract
            self.run_eval(lambda q, s, e: dict(result(""), n_calls=1), out=self.out + "2")

    def test_resume_after_crash_skips_done_questions(self):
        calls = []

        def crashing(q, s, e):
            if len(calls) == 3:
                raise RuntimeError("session cut")
            calls.append(q)
            return result(self.gold[q])

        with self.assertRaises(RuntimeError):
            self.run_eval(crashing)
        self.assertFalse((Path(self.out) / "predictions.jsonl").exists())
        again = []

        def fresh(q, s, e):
            again.append(q)
            return result(self.gold[q])

        m = self.run_eval(fresh)
        self.assertEqual(len(again), N - 3)
        self.assertFalse(set(again) & set(calls))
        self.assertEqual(m["ex"], 100.0)
        self.assertEqual(len((Path(self.out) / "predictions.jsonl").read_text().splitlines()), N)

    def test_torn_last_partial_line_is_cut_and_redone(self):
        calls = []

        def crashing(q, s, e):
            if len(calls) == 2:
                raise RuntimeError("cut")
            calls.append(q)
            return result(self.gold[q])

        with self.assertRaises(RuntimeError):
            self.run_eval(crashing)
        with open(Path(self.out) / "generations.partial.jsonl", "ab") as f:
            f.write(b'{"question_id": "mini_bird:2", "db_')  # crash in the middle of a write
        again = []
        m = self.run_eval(lambda q, s, e: (again.append(q), result(self.gold[q]))[1])
        self.assertIn(self.records[2]["question"], again)
        self.assertEqual(m["ex"], 100.0)

    def test_meta_mismatch_on_resume_raises(self):
        with self.assertRaises(RuntimeError):
            self.run_eval(lambda q, s, e: (_ for _ in ()).throw(RuntimeError("cut")))
        with self.assertRaises(ValueError):
            self.run_eval(lambda q, s, e: result(""), meta=meta(seed=43))

    def test_meta_must_carry_phieu_fields(self):
        bad = meta()
        del bad["gpu"]
        with self.assertRaises(ValueError):
            self.run_eval(lambda q, s, e: result(""), meta=bad)

    def test_completed_run_is_never_overwritten(self):
        self.run_eval(lambda q, s, e: result(self.gold[q]))
        with self.assertRaises(FileExistsError):
            self.run_eval(lambda q, s, e: result(self.gold[q]))

    def test_sprime_missing_entry_raises_instead_of_falling_back(self):
        with self.assertRaises(ValueError):
            self.run_eval(lambda q, s, e: result(""), schema_fn=go.make_sprime_schema_fn({}, DB_DIR))

    def test_sprime_prompt_uses_only_listed_tables(self):
        seen = []
        tables = {f"mini_bird:{i}": ["clients"] for i in range(N)}
        self.run_eval(lambda q, s, e: (seen.append(s), result(""))[1],
                      schema_fn=go.make_sprime_schema_fn(tables, DB_DIR))
        self.assertEqual(len(seen), N)
        self.assertTrue(all("clients" in s and "transactions" not in s for s in seen))

    def test_no_decision_or_grounding_checkpoint_needed(self):
        import os
        for key in ("DECISION_CHECKPOINT", "GROUNDING_CHECKPOINT"):
            self.assertNotIn(key, os.environ)
        self.assertEqual(self.run_eval(lambda q, s, e: result(self.gold[q]))["ex"], 100.0)


if __name__ == "__main__":
    unittest.main()
