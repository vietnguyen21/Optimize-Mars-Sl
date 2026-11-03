"""
Adversarial Stress Test: XGrammar PDA & Fallback Engine
Tests malformed SQL strings, unbalanced brackets, unclosed quotes, duplicate semicolons,
dangling operators, scrambled clauses, extreme inputs, and crash-freedom guarantee.
"""

import sys
import os
import unittest
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

REPO_ROOT = "/mnt/c/Workspace/optimize"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.grammar.fallback_engine import RegexASTFallbackEngine
from src.grammar.xgrammar_engine import XGrammarPDAEngine
from src.grammar import get_grammar_engine


class TestStressGrammarPDAAndFallback(unittest.TestCase):

    def setUp(self):
        self.sample_ddl = """
        CREATE TABLE accounts (
            account_id INTEGER PRIMARY KEY,
            district_id INTEGER,
            frequency TEXT,
            parsed_date DATE
        );
        CREATE TABLE clients (
            client_id INTEGER PRIMARY KEY,
            gender TEXT,
            birth_date DATE,
            district_id INTEGER
        );
        """
        self.fallback = RegexASTFallbackEngine(schema_ddl=self.sample_ddl)
        self.pda = XGrammarPDAEngine(schema_ddl=self.sample_ddl)

    def test_unbalanced_parentheses_fuzzing(self):
        """Stress-test unbalanced parentheses across deep nesting and erratic patterns."""
        cases = [
            "((SELECT 1)",
            "SELECT 1))",
            "SELECT (1 + (2 * (3 - 4)))",  # valid
            "SELECT (1 + (2 * (3 - 4))))",  # extra close
            "SELECT (((((1)))))",  # valid
            "SELECT (((((1))))",  # unclosed
            "SELECT )",
            "SELECT (",
            ")(",
            "()())(",
            "SELECT * FROM accounts WHERE (account_id = 1 AND (district_id = 2);",
            "SELECT * FROM accounts WHERE ((account_id = 1);",
            "SELECT COUNT(DISTINCT(account_id)) FROM accounts;",  # valid
            "SELECT COUNT(DISTINCT(account_id) FROM accounts;",  # missing 1 close
            "SELECT " + "(" * 100 + "1" + ")" * 99,  # 100 open, 99 close
            "SELECT " + "(" * 50 + "1" + ")" * 50,  # 50 open, 50 close (valid)
        ]

        for sql in cases:
            for engine in [self.fallback, self.pda]:
                try:
                    is_valid, err = engine.validate_sql_syntax(sql)
                    open_c = sql.count("(")
                    close_c = sql.count(")")
                    if open_c != close_c or sql.startswith(")") or sql in [")(", "()())(", "SELECT )", "SELECT ("]:
                        self.assertFalse(is_valid, f"Engine {engine.__class__.__name__} should reject unbalanced parens: {sql}")
                        self.assertIsNotNone(err)
                except Exception as ex:
                    self.fail(f"Engine {engine.__class__.__name__} CRASHED on unbalanced parens '{sql}': {ex}")

    def test_unclosed_quotes_fuzzing(self):
        """Stress-test unclosed single and double quotes and escaped quotes."""
        cases = [
            "SELECT 'unclosed string FROM accounts;",
            'SELECT "unclosed identifier FROM accounts;',
            "SELECT 'valid string' FROM accounts;",  # valid
            'SELECT "accounts"."account_id" FROM accounts;',  # valid
            "SELECT ''' FROM accounts;",
            "SELECT 'hello''world' FROM accounts;",
            "SELECT '",
            'SELECT "',
            "SELECT 'abc\\'def' FROM accounts;",
            "SELECT 'abc\\' FROM accounts;",
            "SELECT \"\" FROM accounts;",
            "SELECT '' FROM accounts;",
        ]

        for sql in cases:
            for engine in [self.fallback, self.pda]:
                try:
                    is_valid, err = engine.validate_sql_syntax(sql)
                    if sql in ["SELECT 'unclosed string FROM accounts;", 'SELECT "unclosed identifier FROM accounts;', "SELECT ''' FROM accounts;", "SELECT '", 'SELECT "']:
                        self.assertFalse(is_valid, f"Engine {engine.__class__.__name__} should reject unclosed quote: {sql}")
                except Exception as ex:
                    self.fail(f"Engine {engine.__class__.__name__} CRASHED on quote case '{sql}': {ex}")

    def test_duplicate_semicolons_and_empty_queries(self):
        """Stress-test double semicolons, chained semicolons, and empty strings."""
        cases = [
            ";;",
            ";;;;",
            "SELECT 1;;",
            "",
            "   ",
            "\n\t\r",
            ";",
            "-- only comment\n",
            "-- comment 1\n-- comment 2",
        ]

        for sql in cases:
            for engine in [self.fallback, self.pda]:
                try:
                    is_valid, err = engine.validate_sql_syntax(sql)
                    # All these should be rejected as invalid standalone queries
                    self.assertFalse(is_valid, f"Engine {engine.__class__.__name__} should reject empty/duplicate semicolon: {repr(sql)}")
                except Exception as ex:
                    self.fail(f"Engine {engine.__class__.__name__} CRASHED on semicolon/empty query '{repr(sql)}': {ex}")

    def test_dangling_operators_at_end(self):
        """Stress-test trailing dangling operators and incomplete clauses."""
        dangling = [
            "SELECT account_id FROM accounts WHERE",
            "SELECT account_id FROM accounts WHERE account_id =",
            "SELECT account_id FROM accounts WHERE account_id +",
            "SELECT account_id FROM accounts WHERE account_id -",
            "SELECT account_id FROM accounts WHERE account_id *",
            "SELECT account_id FROM accounts WHERE account_id /",
            "SELECT account_id FROM accounts WHERE account_id >",
            "SELECT account_id FROM accounts WHERE account_id <",
            "SELECT account_id FROM accounts WHERE account_id >=",
            "SELECT account_id FROM accounts WHERE account_id <=",
            "SELECT account_id FROM accounts WHERE account_id !=",
            "SELECT account_id FROM accounts WHERE account_id AND",
            "SELECT account_id FROM accounts WHERE account_id OR",
            "SELECT account_id FROM accounts JOIN",
            "SELECT account_id FROM accounts JOIN clients ON",
            "SELECT account_id,",
        ]

        for sql in dangling:
            for engine in [self.fallback, self.pda]:
                try:
                    is_valid, err = engine.validate_sql_syntax(sql)
                    self.assertFalse(is_valid, f"Engine {engine.__class__.__name__} should reject dangling op: '{sql}'")
                except Exception as ex:
                    self.fail(f"Engine {engine.__class__.__name__} CRASHED on dangling op '{sql}': {ex}")

    def test_scrambled_clauses(self):
        """Stress-test queries with inverted or scrambled SQL clause orders."""
        scrambled = [
            "FROM accounts SELECT account_id;",
            "WHERE account_id = 1 SELECT account_id FROM accounts;",
            "SELECT account_id WHERE account_id = 1 FROM accounts;",
            "LIMIT 5 SELECT account_id FROM accounts;",
            "ORDER BY account_id SELECT account_id FROM accounts;",
            "GROUP BY district_id SELECT count(*) FROM accounts;",
            "SELECT account_id FROM accounts LIMIT 5 WHERE account_id > 10;",
            "SELECT account_id FROM accounts ORDER BY account_id WHERE account_id > 10;",
        ]

        for sql in scrambled:
            for engine in [self.fallback, self.pda]:
                try:
                    is_valid, err = engine.validate_sql_syntax(sql)
                    self.assertFalse(is_valid, f"Engine {engine.__class__.__name__} should reject scrambled clause order: '{sql}'")
                except Exception as ex:
                    self.fail(f"Engine {engine.__class__.__name__} CRASHED on scrambled clause '{sql}': {ex}")

    def test_prefix_validation_and_bitmask(self):
        """Stress-test prefix validation and PDA bitmask generation at edge contexts."""
        prefixes = [
            "",
            "SELECT",
            "SELECT ",
            "SELECT * FROM",
            "SELECT * FROM accounts WHERE (",
            "SELECT * FROM accounts WHERE (account_id = 1",
            "SELECT * FROM accounts WHERE (account_id = 1)",
            "SELECT * FROM accounts WHERE account_id >",
            ") SELECT",  # invalid starter
            "DELETE FROM",  # invalid starter for SELECT decoder
            "12345",
            None,
        ]

        for pref in prefixes:
            # Fallback
            try:
                ok, err = self.fallback.validate_prefix(pref)
                if pref in [") SELECT", "DELETE FROM", "12345", None]:
                    self.assertFalse(ok, f"Prefix '{pref}' should be marked invalid")
            except Exception as ex:
                self.fail(f"Fallback validate_prefix crashed on '{pref}': {ex}")

            # PDA bitmask
            if pref is not None:
                try:
                    bitmask = self.pda.generate_bitmask(pref)
                    self.assertIsInstance(bitmask, list)
                    self.assertEqual(len(bitmask), len(self.pda.vocab))
                    # Check that all elements are 0 or 1
                    for b in bitmask:
                        self.assertIn(b, [0, 1])

                    # At start (""), only SELECT or WITH allowed
                    if pref == "":
                        sel_idx = self.pda.vocab_to_id.get("SELECT")
                        self.assertEqual(bitmask[sel_idx], 1)
                        from_idx = self.pda.vocab_to_id.get("FROM")
                        self.assertEqual(bitmask[from_idx], 0)

                    # When no open parens, ')' must be masked out (0)
                    if pref == "SELECT * FROM accounts WHERE (account_id = 1)":
                        close_p = self.pda.vocab_to_id.get(")")
                        self.assertEqual(bitmask[close_p], 0)

                    # When open parens > 0, ';' must be masked out (0)
                    if pref == "SELECT * FROM accounts WHERE (":
                        semi_idx = self.pda.vocab_to_id.get(";")
                        self.assertEqual(bitmask[semi_idx], 0)
                except Exception as ex:
                    self.fail(f"PDA generate_bitmask crashed on '{pref}': {ex}")

    def test_logits_filtering_zero_crash(self):
        """Stress-test logits filtering on python lists, PyTorch tensors, CPU/GPU structures."""
        contexts = ["SELECT * FROM accounts", ") INVALID (", "SELECT * FROM accounts WHERE ("]

        # 1. PyTorch Tensor (if available)
        if HAS_TORCH:
            tensor_logits = torch.randn(1, len(self.pda.vocab))
            for ctx in contexts:
                res = self.pda.filter_logits(tensor_logits, ctx)
                self.assertEqual(res.shape, tensor_logits.shape)
                self.assertFalse(torch.isnan(res).any())

        # 2. Python list
        list_logits = [0.5] * len(self.pda.vocab)
        for ctx in contexts:
            res = self.pda.filter_logits(list_logits, ctx)
            self.assertEqual(len(res), len(list_logits))

        # 3. Arbitrary non-list object
        non_list = "unexpected_logits"
        res = self.pda.filter_logits(non_list, "SELECT")
        self.assertEqual(res, non_list)

    def test_compilation_cache_speed(self):
        """Test grammar compilation cache hits are < 20 ms."""
        # First call: cache miss
        c1 = self.pda.compile_grammar(self.sample_ddl)
        # Second call: cache hit
        c2 = self.pda.compile_grammar(self.sample_ddl)
        self.assertTrue(c2["cache_hit"])
        self.assertLess(c2["compile_time_ms"], 20.0, f"Compilation cache took {c2['compile_time_ms']} ms, expected < 20 ms")


if __name__ == "__main__":
    unittest.main()
