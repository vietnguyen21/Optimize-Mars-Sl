"""Tests the BIRD split logic and the Colab runner's environment mapping (no BIRD, no GPU)."""

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.prepare_bird import build_splits, dedupe, split_by_db, split_cal
from pathlib import Path

from scripts.colab.run_stage import (apply_overlay, build_env, check_resume, code_patch, code_provenance,
                                    overlay_target, parse_overrides)


class TestSplits(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.records = []
        for i in range(20):
            db = f"db{i}"
            conn = sqlite3.connect(os.path.join(self.tmp.name, f"{db}.sqlite"))
            conn.execute('CREATE TABLE "Free Meal" (id INTEGER PRIMARY KEY, `Count (K-12)` REAL)')
            conn.commit()
            conn.close()
            for q in range(5):
                self.records.append({"db_id": db, "question": f"q{q} of {db}", "evidence": "", "SQL": "SELECT 1"})

    def tearDown(self):
        self.tmp.cleanup()

    def test_databases_never_cross_splits(self):
        dev = [{"db_id": "devdb", "question": "d", "SQL": "SELECT 1"}]
        files, meta = build_splits(dev, self.records, self.tmp.name)
        dbs = {name: {r["db_id"] for r in rows} for name, rows in files.items()}
        held = set(meta["cal_a_db_ids"]) | set(meta["cal_b_db_ids"])
        self.assertEqual(len(held), 2)
        self.assertFalse(dbs["train.jsonl"] & held)
        self.assertFalse(dbs["retrieval_dev.jsonl"] & dbs["cal_b.jsonl"])
        self.assertEqual(dbs["retrieval_dev.jsonl"], set(meta["cal_a_db_ids"]))
        self.assertEqual(files["test.jsonl"], dev)

    def test_sft_schema_is_verbatim_ddl_and_answer_is_sql(self):
        files, _ = build_splits([], self.records, self.tmp.name)
        row = files["sft_train.jsonl"][0]
        self.assertIn('"Free Meal"', row["schema"])
        self.assertIn("`Count (K-12)`", row["schema"])
        self.assertEqual(row["answer"], "SELECT 1")
        self.assertNotIn("SQL", row)

    def test_same_split_as_earlier_runs(self):
        a = split_by_db(self.records, seed=42)[2]
        self.assertEqual(a, split_by_db(self.records, seed=42)[2])
        cal_a, cal_b = split_cal(["a", "b", "c", "d", "e", "f", "g"])
        self.assertEqual((len(cal_a), len(cal_b)), (4, 3))

    def test_dedupe_by_normalized_question(self):
        rows = [{"db_id": "x", "question": "How  many?"}, {"db_id": "x", "question": "how many?"},
                {"db_id": "y", "question": "how many?"}]
        kept, dropped = dedupe(rows)
        self.assertEqual((len(kept), dropped), (2, 1))


class TestRunnerEnv(unittest.TestCase):
    def test_paths_and_resume(self):
        env = build_env("grounding", "/d", "r1", {}, run_dir_has_checkpoint=False, base={})
        self.assertEqual(env["KAGGLE_DATA_DIR"], "/d/data")
        self.assertEqual(env["KAGGLE_OUTPUT_DIR"], "/d/runs/r1")
        self.assertEqual(env["RESUME_FROM_CHECKPOINT"], "none")
        self.assertEqual(env["CHECKPOINT_SYNC_ENABLED"], "0")
        self.assertNotIn("GENERATOR_MODEL", env)
        self.assertEqual(build_env("grounding", "/d", "r1", {}, True, base={})["RESUME_FROM_CHECKPOINT"], "latest")

    def test_generator_default_and_overrides(self):
        env = build_env("sft", "/d", "r", parse_overrides(["MAX_STEPS=50", "LAMBDA_JEPA=0"]), False, base={})
        self.assertEqual(env["GENERATOR_MODEL"], "Qwen/Qwen2.5-Coder-3B-Instruct")
        self.assertEqual((env["MAX_STEPS"], env["LAMBDA_JEPA"]), ("50", "0"))
        with self.assertRaises(ValueError):
            parse_overrides(["MAX_STEPS"])


class TestRunnerProvenance(unittest.TestCase):
    """Temporary git repos only: no Drive, no GPU."""

    def setUp(self):
        if shutil.which("git") is None:
            self.skipTest("git not available")
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.git(self.repo, "init", "-q")
        self.git(self.repo, "config", "core.autocrlf", "false")
        (self.repo / ".gitignore").write_text("ignored/\n")
        (self.repo / "a.py").write_bytes(b"print('a')\n")
        (self.repo / "gone.py").write_bytes(b"x = 1\n")
        self.git(self.repo, "add", "-A")
        self.git(self.repo, "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
                 "commit", "-q", "-m", "init")

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def git(repo, *args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    def test_clean_vs_dirty(self):
        clean = code_provenance(self.repo)
        self.assertFalse(clean["dirty"])
        self.assertRegex(clean["commit"], "^[0-9a-f]{40}$")
        (self.repo / "ignored").mkdir()
        (self.repo / "ignored" / "out.txt").write_text("run output")
        self.assertEqual(code_provenance(self.repo), clean)
        (self.repo / "a.py").write_bytes(b"print('b')\n")
        dirty = code_provenance(self.repo)
        self.assertTrue(dirty["dirty"])
        self.assertNotEqual(dirty["code_hash"], clean["code_hash"])

    def test_untracked_file_changes_hash(self):
        clean = code_provenance(self.repo)
        (self.repo / "new.py").write_text("y = 2\n")
        dirty = code_provenance(self.repo)
        self.assertTrue(dirty["dirty"])
        self.assertNotEqual(dirty["code_hash"], clean["code_hash"])

    def test_patch_reproduces_working_tree(self):
        (self.repo / "a.py").write_bytes(b"print('changed')\n")
        (self.repo / "gone.py").unlink()
        (self.repo / "pkg").mkdir()
        (self.repo / "pkg" / "new.py").write_bytes(b"z = 3\n")
        (self.repo / "blob.bin").write_bytes(bytes(range(256)))
        before = code_provenance(self.repo)
        patch = self.root / "code.patch"
        patch.write_bytes(code_patch(self.repo, before["commit"]))
        self.assertEqual(code_provenance(self.repo), before)  # Real index and working tree untouched.
        staged = subprocess.run(["git", "-C", str(self.repo), "diff", "--cached", "--name-only"],
                                check=True, capture_output=True).stdout
        self.assertEqual(staged, b"")
        copy = self.root / "copy"
        subprocess.run(["git", "-c", "core.autocrlf=false", "clone", "-q", str(self.repo), str(copy)],
                       check=True, capture_output=True)
        self.git(copy, "config", "core.autocrlf", "false")
        self.git(copy, "checkout", "-q", before["commit"])
        self.git(copy, "apply", "--binary", str(patch))
        self.assertEqual(code_provenance(copy)["code_hash"], before["code_hash"])
        self.assertEqual(code_provenance(copy)["commit"], before["commit"])

    def test_overlay_copies_and_hash_covers_it(self):
        clean = code_provenance(self.repo)
        overlay = self.root / "work"
        (overlay / "pkg").mkdir(parents=True)
        (overlay / "a.py").write_bytes(b"print('member')\n")
        (overlay / "pkg" / "b.py").write_bytes(b"b = 1\n")
        apply_overlay(overlay, self.repo)
        self.assertEqual((self.repo / "a.py").read_bytes(), b"print('member')\n")
        self.assertEqual((self.repo / "pkg" / "b.py").read_bytes(), b"b = 1\n")
        self.assertNotEqual(code_provenance(self.repo)["code_hash"], clean["code_hash"])

    def test_overlay_escape_refused(self):
        root = self.repo.resolve()
        for rel in ("../evil.py", "pkg/../../evil.py", str(self.root.resolve() / "evil.py"), ".git/config"):
            with self.assertRaises(SystemExit, msg=rel):
                overlay_target(root, rel)
        overlay = self.root / "work"
        (overlay / ".git").mkdir(parents=True)
        (overlay / "a.py").write_bytes(b"print('x')\n")
        (overlay / ".git" / "config").write_text("[core]")
        with self.assertRaises(SystemExit):
            apply_overlay(overlay, self.repo)
        self.assertEqual((self.repo / "a.py").read_bytes(), b"print('a')\n")  # Nothing copied on refusal.

    def test_overlay_symlink_refused(self):
        outside = self.root / "secret.txt"
        outside.write_text("secret")
        overlay = self.root / "work"
        overlay.mkdir()
        try:
            os.symlink(outside, overlay / "link.py")
            os.symlink(self.root, self.repo / "out")
        except (OSError, NotImplementedError):
            self.skipTest("symlinks cannot be created here")
        with self.assertRaises(SystemExit):
            apply_overlay(overlay, self.repo)
        (overlay / "link.py").unlink()
        (overlay / "out").mkdir()
        (overlay / "out" / "x.py").write_text("x")  # Repo symlink "out" points outside the repo.
        with self.assertRaises(SystemExit):
            apply_overlay(overlay, self.repo)
        self.assertFalse((self.root / "x.py").exists())

    def test_resume_refused_on_other_code(self):
        run_dir = self.root / "run"
        (run_dir / "checkpoint-25").mkdir(parents=True)
        with self.assertRaises(SystemExit) as ctx:
            check_resume(run_dir, "new")
        self.assertIn("none", str(ctx.exception.code))
        (run_dir / "launch-20260101-000000.json").write_text(json.dumps({"code_hash": "new"}))
        (run_dir / "launch-20260102-000000.json").write_text(json.dumps({"code_hash": "old"}))
        with self.assertRaises(SystemExit) as ctx:
            check_resume(run_dir, "new")
        self.assertIn("old", str(ctx.exception.code))
        self.assertIn("new", str(ctx.exception.code))
        check_resume(run_dir, "old")
        check_resume(self.root / "fresh", "anything")  # No checkpoint: nothing to resume.

    def test_unknown_commit_is_an_error(self):
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaises(SystemExit) as ctx:
            code_provenance(empty)  # Not a git repo.
        self.assertIn("Cannot resolve the code commit", str(ctx.exception.code))
        self.git(empty, "init", "-q")
        with self.assertRaises(SystemExit):
            code_provenance(empty)  # Repo without HEAD.


class TestPipCheckFilter(unittest.TestCase):
    IBIS = "ibis-framework 9.5.0 has requirement sqlglot<25.21,>=23.4, but you have sqlglot 26.16.4."
    PEFT = "peft 0.15.2 has requirement transformers>=4.99, but you have transformers 4.51.3."

    def test_only_listed_requirers_are_ignored(self):
        from src.training.kaggle import pip_check_problems
        blocking, ignored = pip_check_problems(self.IBIS + "\nipython 7.34.0 requires jedi, which is not installed.\n")
        self.assertEqual(blocking, [])
        self.assertEqual(len(ignored), 2)
        blocking, _ = pip_check_problems(self.IBIS + "\n" + self.PEFT)
        self.assertEqual(blocking, [self.PEFT])

    def test_never_ignores_a_project_package(self):
        from src.training.kaggle import pip_check_problems
        blocking, _ = pip_check_problems(self.IBIS, profile_packages=["ibis-framework"])
        self.assertEqual(blocking, [self.IBIS])


class TestImplicitForeignKeyColumn(unittest.TestCase):
    def test_references_without_column_targets_primary_key(self):
        from src.grounding.graph_builder import build_graph_from_sqlite
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.sqlite")
            conn = sqlite3.connect(path)
            conn.executescript("CREATE TABLE author (aid INTEGER PRIMARY KEY, name TEXT);"
                               "CREATE TABLE book (bid INTEGER PRIMARY KEY, aid INTEGER REFERENCES author);")
            conn.close()
            graph = build_graph_from_sqlite(path, "x")
            self.assertIn(("book", "aid", "author", "aid"), [f.as_tuple() for f in graph.foreign_keys])

    def test_composite_key_order_and_table_case(self):
        from src.grounding.graph_builder import build_graph_from_sqlite
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "y.sqlite")
            conn = sqlite3.connect(path)
            conn.executescript("CREATE TABLE customers (a INTEGER, b INTEGER, PRIMARY KEY (b, a));"
                               "CREATE TABLE orders (id INTEGER PRIMARY KEY, cb INTEGER, ca INTEGER,"
                               " FOREIGN KEY (cb, ca) REFERENCES Customers);")
            conn.close()
            edges = [f.as_tuple() for f in build_graph_from_sqlite(path, "y").foreign_keys]
            self.assertIn(("orders", "cb", "customers", "b"), edges)
            self.assertIn(("orders", "ca", "customers", "a"), edges)


if __name__ == "__main__":
    unittest.main()
