"""Leakage check, step 0 (S2, W1). Reports only; nothing is dropped (U13).

  exact_question_dup  train and eval questions equal after dedupe's normalization
                      (whitespace collapsed, lower case; scripts/prepare_bird.py dedupe)
  exact_sql_dup       gold SQL equal after the same normalization
  near_dups           Jaccard >= threshold on set(re.findall(r"\\w+", q.lower()))
  gold_in_prompt_hits rows whose normalized gold SQL is a substring of question + evidence + schema

Train is compared with every eval file (cal-A, cal-B, BIRD-dev). BIRD-dev is read as text only.

  python scripts/leakage_check.py --train <DATA>/train.jsonl \
      --eval <DATA>/retrieval_dev.jsonl <DATA>/cal_b.jsonl <DATA>/test.jsonl \
      --db_dir /content/bird_db --out runs/m2-data-seed42/leakage/leakage_report.json
"""

import argparse
import json
import math
import os
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filter_arctic import read_rows, row_id  # noqa: E402

MAX_EXAMPLES = 50


def norm_text(s: str) -> str:
    """Same normalization as dedupe in scripts/prepare_bird.py."""
    return re.sub(r"\s+", " ", s or "").strip().lower()


def tokens(q: str) -> frozenset:
    return frozenset(re.findall(r"\w+", (q or "").lower()))


def gold_sql(r: Dict) -> str:
    return r.get("SQL", r.get("answer", ""))


def schema_ddl(db_path: str) -> str:
    """CREATE TABLE text as in scripts/prepare_bird.py schema_ddl (the schema put in the prompt)."""
    conn = sqlite3.connect(f"file:{os.path.abspath(db_path)}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL").fetchall()
    finally:
        conn.close()
    return "\n\n".join(r[0].strip() + ";" for r in rows)


def _near_dups(train: List[Dict], train_ids: List[str], ev: List[Dict], ev_ids: List[str], t: float) -> List[Dict]:
    """All pairs with Jaccard >= t. Only lengths that can reach t are compared: |A|/|B| >= t."""
    by_len = defaultdict(list)
    for j, r in enumerate(train):
        by_len[len(tokens(r["question"]))].append(j)
    out = []
    for k, e in enumerate(ev):
        b = tokens(e["question"])
        lo, hi = math.ceil(t * len(b) - 1e-9), math.floor(len(b) / t + 1e-9) if t > 0 else 10 ** 9
        for n in range(lo, hi + 1):
            for j in by_len.get(n, ()):
                a = tokens(train[j]["question"])
                union = len(a | b)
                jac = len(a & b) / union if union else 1.0
                if jac >= t:
                    out.append({"train_id": train_ids[j], "eval_id": ev_ids[k], "train_db": train[j]["db_id"],
                                "eval_db": e["db_id"], "jaccard": round(jac, 4),
                                "train_question": train[j]["question"], "eval_question": e["question"]})
    return out


def _gold_in_prompt(rows: List[Dict], ids: List[str], db_dir: Optional[str], cache: Dict[str, str]) -> List[Dict]:
    hits = []
    for r, qid in zip(rows, ids):
        schema = r.get("schema")
        if schema is None and db_dir:
            db = r["db_id"]
            if db not in cache:
                path = os.path.join(db_dir, f"{db}.sqlite")
                cache[db] = schema_ddl(path) if os.path.isfile(path) else ""
            schema = cache[db]
        sql = norm_text(gold_sql(r))
        prompt = norm_text(" ".join([r.get("question", ""), r.get("evidence", "") or "", schema or ""]))
        if sql and sql in prompt:
            hits.append({"question_id": qid, "db_id": r["db_id"], "sql": gold_sql(r)})
    return hits


def leakage_report(train_path, eval_paths, near_dup_threshold=0.9, db_dir=None) -> dict:
    """Fields of QUC 11: pairs_checked, exact_question_dup, exact_sql_dup, near_dup_threshold,
    near_dups[], gold_in_prompt_hits; plus per-file counts and examples.

    db_dir is optional: rows without a "schema" field get the DDL of their database from it.
    Without db_dir such rows are checked on question + evidence only (schema_checked = false)."""
    train = read_rows(train_path)
    train_ids = [row_id(train_path, i) for i in range(len(train))]
    q_index, s_index = defaultdict(list), defaultdict(list)
    for j, r in enumerate(train):
        q_index[norm_text(r["question"])].append(j)
        s_index[norm_text(gold_sql(r))].append(j)

    cache: Dict[str, str] = {}
    report = {"pairs_checked": 0, "exact_question_dup": 0, "exact_sql_dup": 0,
              "near_dup_threshold": near_dup_threshold, "near_dups": [], "gold_in_prompt_hits": 0,
              "train_file": Path(train_path).name, "n_train": len(train), "eval_files": {},
              "near_dup_method": 'Jaccard on set(re.findall(r"\\w+", q.lower())), train vs each eval file',
              "normalization": "re.sub(r'\\s+', ' ', s).strip().lower() (dedupe, prepare_bird.py)",
              "schema_checked": db_dir is not None,
              "exact_question_examples": [], "exact_sql_examples": [], "gold_in_prompt_examples": []}

    hits = _gold_in_prompt(train, train_ids, db_dir, cache)
    report["eval_files"][Path(train_path).name] = {"n": len(train), "gold_in_prompt_hits": len(hits)}
    report["gold_in_prompt_hits"] += len(hits)
    report["gold_in_prompt_examples"] += hits[:MAX_EXAMPLES]

    for path in eval_paths:
        ev = read_rows(path)
        ev_ids = [row_id(path, i) for i in range(len(ev))]
        qd = [(j, k) for k, e in enumerate(ev) for j in q_index.get(norm_text(e["question"]), ())]
        sd = [(j, k) for k, e in enumerate(ev) for j in s_index.get(norm_text(gold_sql(e)), ())]
        near = _near_dups(train, train_ids, ev, ev_ids, near_dup_threshold)
        # BIRD-dev (test.jsonl) is only hashed and compared as text (PHIEU W1, QUC 3.1): no gold-in-prompt, no schema.
        is_test = Path(path).stem == "test"
        hits = [] if is_test else _gold_in_prompt(ev, ev_ids, db_dir, cache)
        report["pairs_checked"] += len(train) * len(ev)
        report["exact_question_dup"] += len(qd)
        report["exact_sql_dup"] += len(sd)
        report["near_dups"] += near
        report["gold_in_prompt_hits"] += len(hits)
        report["eval_files"][Path(path).name] = {
            "n": len(ev), "pairs_checked": len(train) * len(ev), "exact_question_dup": len(qd),
            "exact_sql_dup": len(sd), "near_dups": len(near),
            "gold_in_prompt_hits": None if is_test else len(hits),
            "shared_db_ids": sorted({r["db_id"] for r in train} & {r["db_id"] for r in ev})}
        for key, pairs in (("exact_question_examples", qd), ("exact_sql_examples", sd)):
            room = MAX_EXAMPLES - len(report[key])
            report[key] += [{"train_id": train_ids[j], "eval_id": ev_ids[k], "train_db": train[j]["db_id"],
                             "eval_db": ev[k]["db_id"], "question": ev[k]["question"], "sql": gold_sql(ev[k])}
                            for j, k in pairs[:max(room, 0)]]
        report["gold_in_prompt_examples"] += hits[:max(MAX_EXAMPLES - len(report["gold_in_prompt_examples"]), 0)]
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True)
    p.add_argument("--eval", nargs="+", required=True)
    p.add_argument("--db_dir")
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rep = leakage_report(a.train, a.eval, a.threshold, db_dir=a.db_dir)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: rep[k] for k in ("pairs_checked", "exact_question_dup", "exact_sql_dup",
                                          "gold_in_prompt_hits")} | {"near_dups": len(rep["near_dups"])}))


if __name__ == "__main__":
    sys.exit(main())
