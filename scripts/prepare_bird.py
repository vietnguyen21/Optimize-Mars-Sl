"""
Prepares BIRD in the data layout the notebooks expect (see KAGGLE.md, "Data contracts").

Splits by db_id with seed 42, so no database appears in two splits:
  train.jsonl / sft_train.jsonl / grpo_train.jsonl  BIRD train minus the held-out databases
  retrieval_dev.jsonl   cal-A databases: Notebook 01 retrieval eval and System-1 calibration
  cal_b.jsonl           cal-B databases: System-1 test (decision_test is built from it later)
  test.jsonl            BIRD dev, for Notebook 05 only
decision_*.jsonl are not written here: they need candidates sampled by a trained generator.

Databases are hard-linked flat into --db_dir as <db_id>.sqlite (a local disk, not Drive).
manifest.json records counts, database ids and SHA-256 of every output file.

Usage:
  python scripts/prepare_bird.py --root /content/bird --data_dir <Drive>/data --db_dir /content/bird_db
"""

import argparse
import hashlib
import json
import os
import random
import re
import sqlite3
import zipfile
from typing import Dict, List, Tuple


def split_by_db(records: List[Dict], cal_fraction: float = 0.1, seed: int = 42) -> Tuple[List[Dict], List[Dict], List[str]]:
    """Returns (train, held_out, held_out_db_ids). At least one database is held out."""
    db_ids = sorted({r["db_id"] for r in records})
    random.Random(seed).shuffle(db_ids)
    held = set(db_ids[:max(1, round(len(db_ids) * cal_fraction))])
    train = [r for r in records if r["db_id"] not in held]
    held_out = [r for r in records if r["db_id"] in held]
    assert not {r["db_id"] for r in train} & {r["db_id"] for r in held_out}
    return train, held_out, sorted(held)


def split_cal(cal_dbs: List[str], seed: int = 42) -> Tuple[List[str], List[str]]:
    """Splits the held-out databases into cal-A (larger half) and cal-B, deterministically."""
    dbs = sorted(cal_dbs)
    random.Random(seed + 1).shuffle(dbs)
    cut = (len(dbs) + 1) // 2
    return sorted(dbs[:cut]), sorted(dbs[cut:])


def dedupe(records: List[Dict]) -> Tuple[List[Dict], int]:
    """Drops repeated (db_id, normalized question) pairs; keeps the first occurrence."""
    seen, kept = set(), []
    for r in records:
        key = (r["db_id"], re.sub(r"\s+", " ", r["question"]).strip().lower())
        if key not in seen:
            seen.add(key)
            kept.append(r)
    return kept, len(records) - len(kept)


def schema_ddl(db_path: str) -> str:
    """CREATE TABLE statements read from sqlite_master, opened read-only; identifiers kept verbatim."""
    conn = sqlite3.connect(f"file:{os.path.abspath(db_path)}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL").fetchall()
    finally:
        conn.close()
    return "\n\n".join(r[0].strip() + ";" for r in rows)


def write_jsonl(path: str, records: List[Dict]) -> str:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def unzip_databases(zip_path: str, folder_name: str) -> str:
    """Extracts next to the archive once; returns the folder holding <db_id>/<db_id>.sqlite."""
    base = os.path.dirname(zip_path)
    target = os.path.join(base, folder_name)
    if not os.path.isdir(target):
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(base, [n for n in z.namelist() if not n.startswith("__MACOSX")])
    if not os.path.isdir(target):
        raise FileNotFoundError(f"{zip_path} did not produce {target}")
    return target


def flatten_databases(db_root: str, flat_dir: str) -> int:
    """Hard-links <db_id>/<db_id>.sqlite to flat_dir/<db_id>.sqlite."""
    os.makedirs(flat_dir, exist_ok=True)
    count = 0
    for db_id in sorted(os.listdir(db_root)):
        src = os.path.join(db_root, db_id, f"{db_id}.sqlite")
        dst = os.path.join(flat_dir, f"{db_id}.sqlite")
        if os.path.exists(src):
            if not os.path.exists(dst):
                os.link(src, dst)
            count += 1
    return count


def prepare_databases(root: str, db_dir: str) -> int:
    dev = unzip_databases(os.path.join(root, "dev_20240627", "dev_databases.zip"), "dev_databases")
    train = unzip_databases(os.path.join(root, "train", "train_databases.zip"), "train_databases")
    return flatten_databases(dev, db_dir) + flatten_databases(train, db_dir)


def build_splits(dev: List[Dict], train_all: List[Dict], db_dir: str, seed: int = 42,
                 cal_fraction: float = 0.1) -> Tuple[Dict[str, List[Dict]], Dict]:
    """Pure split logic, so it can be tested without BIRD."""
    train, held_out, cal_dbs = split_by_db(train_all, cal_fraction, seed)
    cal_a, cal_b = split_cal(cal_dbs, seed)
    train, train_dups = dedupe(train)
    ddl_cache: Dict[str, str] = {}

    def ddl(db_id: str) -> str:
        if db_id not in ddl_cache:
            ddl_cache[db_id] = schema_ddl(os.path.join(db_dir, f"{db_id}.sqlite"))
        return ddl_cache[db_id]

    sft = [{"db_id": r["db_id"], "question": r["question"], "evidence": r.get("evidence", ""),
            "schema": ddl(r["db_id"]), "answer": r["SQL"]} for r in train]
    grpo = [{"db_id": r["db_id"], "question": r["question"], "evidence": r.get("evidence", ""),
             "schema": ddl(r["db_id"]), "SQL": r["SQL"]} for r in train]
    files = {
        "train.jsonl": train,
        "sft_train.jsonl": sft,
        "grpo_train.jsonl": grpo,
        "retrieval_dev.jsonl": [r for r in held_out if r["db_id"] in cal_a],
        "cal_b.jsonl": [r for r in held_out if r["db_id"] in cal_b],
        "test.jsonl": dev,
    }
    meta = {"seed": seed, "cal_fraction": cal_fraction, "cal_a_db_ids": cal_a, "cal_b_db_ids": cal_b,
            "train_duplicates_dropped": train_dups,
            "note": "Not filtered with the MARS-SQL 8,036-example list; report this difference."}
    return files, meta


def main():
    parser = argparse.ArgumentParser(description="Prepare BIRD splits in the notebook data layout")
    parser.add_argument("--root", default="/content/bird", help="Folder with dev_20240627/ and train/ unzipped")
    parser.add_argument("--data_dir", required=True, help="Output folder for JSONL files (Drive)")
    parser.add_argument("--db_dir", default="/content/bird_db", help="Flat <db_id>.sqlite folder (local disk)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--databases_only", action="store_true", help="Only rebuild --db_dir")
    args = parser.parse_args()

    n_db = prepare_databases(args.root, args.db_dir)
    print(f"databases: {n_db}")
    if args.databases_only:
        return

    with open(os.path.join(args.root, "dev_20240627", "dev.json"), encoding="utf-8") as f:
        dev = json.load(f)
    with open(os.path.join(args.root, "train", "train.json"), encoding="utf-8") as f:
        train_all = json.load(f)

    files, meta = build_splits(dev, train_all, args.db_dir, args.seed)
    os.makedirs(args.data_dir, exist_ok=True)
    meta["counts"] = {name: len(rows) for name, rows in files.items()}
    meta["sha256"] = {name: write_jsonl(os.path.join(args.data_dir, name), rows) for name, rows in files.items()}
    with open(os.path.join(args.data_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta["counts"]), "cal-A:", meta["cal_a_db_ids"], "cal-B:", meta["cal_b_db_ids"])


if __name__ == "__main__":
    main()
