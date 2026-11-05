"""Tests the database-level calibration split used by scripts/prepare_bird.py."""

import unittest

from scripts.prepare_bird import split_by_db


class TestSplitByDb(unittest.TestCase):
    def setUp(self):
        self.records = [{"db_id": f"db{i % 20}", "question": str(i)} for i in range(200)]

    def test_split_is_disjoint_and_complete(self):
        train, cal, cal_dbs = split_by_db(self.records)
        self.assertEqual(len(cal_dbs), 2)
        self.assertEqual(len(train) + len(cal), 200)
        self.assertFalse({r["db_id"] for r in train} & {r["db_id"] for r in cal})

    def test_split_is_reproducible(self):
        self.assertEqual(split_by_db(self.records, seed=7)[2], split_by_db(self.records, seed=7)[2])


if __name__ == "__main__":
    unittest.main()
