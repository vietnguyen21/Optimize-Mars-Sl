"""
Prepares BIRD for training and evaluation:
  1. Unzips the nested dev/train database archives.
  2. Converts dev.json / train.json to JSONL.
  3. Holds out 10% of training databases as cal-DB, split by db_id, never by question.
  4. Writes manifest.json with counts, seed and file hashes.

Usage: python scripts/prepare_bird.py --root /content/bird --out /content/bird/processed
"""

import argparse
import hashlib
import json
import os
import random
import zipfile
from typing import Dict, List, Tuple


def split_by_db(records: List[Dict], cal_fraction: float = 0.1, seed: int = 42) -> Tuple[List[Dict], List[Dict], List[str]]:
    """Returns (train, cal, cal_db_ids). At least one database goes to cal-DB."""
    db_ids = sorted({r["db_id"] for r in records})
    random.Random(seed).shuffle(db_ids)
    cal_dbs = set(db_ids[:max(1, round(len(db_ids) * cal_fraction))])
    train = [r for r in records if r["db_id"] not in cal_dbs]
    cal = [r for r in records if r["db_id"] in cal_dbs]
    assert not {r["db_id"] for r in train} & {r["db_id"] for r in cal}
    return train, cal, sorted(cal_dbs)


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
    """Hard-links <db_id>/<db_id>.sqlite to flat_dir/<db_id>.sqlite, the layout the harness expects."""
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


def main():
    parser = argparse.ArgumentParser(description="Prepare BIRD JSONL splits and database folders")
    parser.add_argument("--root", default="/content/bird")
    parser.add_argument("--out", default="/content/bird/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cal_fraction", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev_dir = os.path.join(args.root, "dev_20240627")
    train_dir = os.path.join(args.root, "train")

    dev_dbs = unzip_databases(os.path.join(dev_dir, "dev_databases.zip"), "dev_databases")
    train_dbs = unzip_databases(os.path.join(train_dir, "train_databases.zip"), "train_databases")
    n_dev_db = flatten_databases(dev_dbs, os.path.join(args.out, "db"))
    n_train_db = flatten_databases(train_dbs, os.path.join(args.out, "db"))

    with open(os.path.join(dev_dir, "dev.json"), encoding="utf-8") as f:
        dev = json.load(f)
    with open(os.path.join(train_dir, "train.json"), encoding="utf-8") as f:
        train_all = json.load(f)

    train, cal, cal_dbs = split_by_db(train_all, args.cal_fraction, args.seed)
    manifest = {
        "seed": args.seed,
        "cal_fraction": args.cal_fraction,
        "cal_db_ids": cal_dbs,
        "counts": {"dev": len(dev), "train": len(train), "cal": len(cal), "dev_dbs": n_dev_db, "train_dbs": n_train_db},
        "sha256": {
            "dev.jsonl": write_jsonl(os.path.join(args.out, "dev.jsonl"), dev),
            "train.jsonl": write_jsonl(os.path.join(args.out, "train.jsonl"), train),
            "cal.jsonl": write_jsonl(os.path.join(args.out, "cal.jsonl"), cal),
        }
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest["counts"]), "cal_dbs:", cal_dbs)


if __name__ == "__main__":
    main()
