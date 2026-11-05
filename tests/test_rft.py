"""Tests RFT filtering and resumable dataset building with a stub generator."""

import json
import os
import tempfile
import unittest

from src.common.schemas import SQLCandidate
from src.evaluation.candidate_generator import CandidateGenerator
from src.training.train_rft import build_rft_dataset, filter_candidates
from src.training.train_sft_generator import load_sft_records
from src.validation import PVSQLValidator
from tests.stub_lm import StubModel, StubTokenizer

DB = "data/mini/financial_mini.sqlite"
GOLD = "SELECT COUNT(*) FROM clients;"


class TestFilterCandidates(unittest.TestCase):
    def _cands(self, *sqls):
        return [SQLCandidate(candidate_id=i, sql=s) for i, s in enumerate(sqls)]

    def test_keeps_correct_distinct_up_to_limit(self):
        cands = self._cands("SELECT 1;", GOLD, "select count(*)  from clients;",
                            "SELECT COUNT(client_id) FROM clients;", "SELECT COUNT(1) FROM clients;")
        kept = filter_candidates(cands, "How many clients are there?", GOLD, DB, PVSQLValidator(), 2)
        self.assertEqual([c.candidate_id for c in kept], [1, 3])

    def test_rejects_ast_violation_even_if_result_matches(self):
        kept = filter_candidates(self._cands("SELECT COUNT(*) FROM clients;"),
                                 "What is the average amount?", GOLD, DB, PVSQLValidator())
        self.assertEqual(kept, [])

    def test_broken_gold_keeps_nothing(self):
        kept = filter_candidates(self._cands(GOLD), "How many clients?", "SELECT * FROM missing;", DB, PVSQLValidator())
        self.assertEqual(kept, [])


class TestBuildDataset(unittest.TestCase):
    def test_output_is_sft_data_and_resumes(self):
        records = [{"db_id": "financial_mini", "question": "How many clients are there?", "evidence": "", "SQL": GOLD}] * 2
        tok = StubTokenizer(f"why</think>\n<answer>\n{GOLD}\n</answer>")
        gen = CandidateGenerator(model=StubModel(), tokenizer=tok)
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "rft.jsonl")
            res = build_rft_dataset(gen, records, "data/mini", out, num_samples=4)
            self.assertEqual(res["samples"], 2)  # duplicates collapse to one per question
            again = build_rft_dataset(gen, records, "data/mini", out, num_samples=4)
            self.assertEqual(again["questions"], 0)
            sft = load_sft_records(out, db_dir="data/mini")
            self.assertEqual(len(sft), 2)
            self.assertIn(GOLD, sft[0]["completion"])
            self.assertIn("why", sft[0]["completion"])
            self.assertNotIn(GOLD, tok.prompts[0])
            with open(out, encoding="utf-8") as f:
                self.assertEqual(json.loads(f.readline())["think_steps"], "why")


if __name__ == "__main__":
    unittest.main()
