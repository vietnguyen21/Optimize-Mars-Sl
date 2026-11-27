"""Arctic filter for the training split (S2, W1).

Runs every gold SQL of train.jsonl once on its database and lists the questions to drop:
  empty  the gold query returns 0 rows (one row of NULLs is not empty, U12)
  slow   interrupted after timeout_s twice in a row (one rerun after the first interruption, U33)
  error  any other failure of the gold query (U12)

Row ids follow QUC 3.4: "<file stem>:<0-based line in the ORIGINAL file>", lines split on LF only.
Filtered files keep the original row id in question_id.

  python scripts/filter_arctic.py --train <DATA>/train.jsonl --db_dir /content/bird_db \
      --out runs/m2-data-seed42/filter/arctic_filter.jsonl
  python scripts/filter_arctic.py --apply <DATA>/sft_train.jsonl --filter .../arctic_filter.jsonl --out F
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REASONS = ("empty", "slow", "error")
FIELDS = ("question_id", "db_id", "reason", "gold_runtime_s", "n_rows", "gold_runtime_rerun_s")
INTERRUPTED = "interrupted"


def _load_execute():
    """execute_query_safely from src/evaluation/metrics.py, loaded by path.

    Importing the package src.evaluation would also import the generator and its model code;
    only this one stdlib function is needed here."""
    path = Path(__file__).resolve().parents[1] / "src" / "evaluation" / "metrics.py"
    spec = importlib.util.spec_from_file_location("_s2_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.execute_query_safely


execute_query_safely = _load_execute()


def read_rows(path: str) -> List[Dict]:
    """JSON rows of a JSONL file, split on LF only (QUC 3.4). Index i is the row id number."""
    data = Path(path).read_bytes()
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    for i, line in enumerate(lines):
        if not line.strip():
            raise ValueError(f"{path}: blank line {i} in the middle of the file (QUC 3.4: stop and tell the Leader)")
    return [json.loads(line) for line in lines]


def errors_path(out_path: str) -> str:
    """arctic_filter.jsonl -> arctic_filter_errors.jsonl, next to it."""
    p = Path(str(out_path))
    return str(p.with_name(p.stem + "_errors" + p.suffix))


def row_id(path: str, i: int) -> str:
    return f"{Path(path).stem}:{i}"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _run(sql: str, db_path: str, timeout_s: float):
    """One timed call of execute_query_safely: (ok, rows_or_error, seconds)."""
    t0 = time.perf_counter()
    ok, res = execute_query_safely(sql, db_path, timeout=timeout_s)
    return ok, res, time.perf_counter() - t0


def check_gold(sql: str, db_path: str, timeout_s: float = 5.0) -> Dict:
    """Classifies one gold query. reason is None when the question is kept."""
    ok, res, t1 = _run(sql, db_path, timeout_s)
    t2 = None
    if not ok and res == INTERRUPTED:
        ok, res, t2 = _run(sql, db_path, timeout_s)
        if not ok and res == INTERRUPTED:
            return {"reason": "slow", "gold_runtime_s": t1, "n_rows": None, "gold_runtime_rerun_s": t2, "detail": None}
    if not ok:
        return {"reason": "error", "gold_runtime_s": t1, "n_rows": None, "gold_runtime_rerun_s": t2, "detail": str(res)[:300]}
    n = len(res)
    return {"reason": "empty" if n == 0 else None, "gold_runtime_s": t1, "n_rows": n, "gold_runtime_rerun_s": t2, "detail": None}


def _progress_header(train_path: str, timeout_s: float, limit: Optional[int]) -> Dict:
    return {"_header": True, "train_sha256": sha256_file(train_path), "timeout_s": timeout_s, "limit": limit}


def filter_arctic(train_path, db_dir, out_path, timeout_s=5.0, limit=None) -> dict:
    """Chạy gold từng dòng, ghi arctic_filter.jsonl chỉ gồm câu bị loại;
    trả {"n_total", "n_empty", "n_slow", "n_error", "n_kept"}.

    Progress goes to <out_path>.progress.jsonl after every question, so a disconnected runtime
    resumes where it stopped (same train file and timeout only). It is deleted when the run ends."""
    rows = read_rows(train_path)
    if limit is not None:
        rows = rows[:limit]
    missing = sorted({r["db_id"] for r in rows if not os.path.isfile(os.path.join(db_dir, f"{r['db_id']}.sqlite"))})
    if missing:
        raise FileNotFoundError(f"{len(missing)} databases missing in {db_dir}: {missing[:5]}")

    out_path = str(out_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    progress_path = out_path + ".progress.jsonl"
    header = _progress_header(train_path, timeout_s, limit)
    done: Dict[int, Dict] = {}
    if os.path.exists(progress_path):
        saved = []
        for line in Path(progress_path).read_bytes().split(b"\n"):
            try:
                saved.append(json.loads(line))
            except ValueError:
                pass  # a line cut by a disconnect is checked again
        if not saved or saved[0] != header:
            raise SystemExit(f"{progress_path} belongs to another run (train file, timeout or limit differ). "
                             "Delete it to start again.")
        done = {r["i"]: r for r in saved[1:] if isinstance(r.get("i"), int) and 0 <= r["i"] < len(rows)}
        print(f"resume: {len(done)}/{len(rows)} questions already checked")

    with open(progress_path, "w", encoding="utf-8", newline="\n") as prog:
        prog.write(json.dumps(header) + "\n")
        for i in sorted(done):
            prog.write(json.dumps(done[i], ensure_ascii=False) + "\n")
        prog.flush()
        t_start, last = time.time(), time.time()
        for i, r in enumerate(rows):
            if i in done:
                continue
            res = check_gold(r["SQL"], os.path.join(db_dir, f"{r['db_id']}.sqlite"), timeout_s)
            done[i] = dict(res, i=i)
            prog.write(json.dumps(done[i], ensure_ascii=False) + "\n")
            prog.flush()
            if time.time() - last > 60:
                last = time.time()
                print(f"  {len(done)}/{len(rows)} questions, {time.time() - t_start:.0f} s", flush=True)

    dropped, errors = [], []
    for i, r in enumerate(rows):
        res = done[i]
        if res["reason"] is not None:
            dropped.append({"question_id": row_id(train_path, i), "db_id": r["db_id"], "reason": res["reason"],
                            "gold_runtime_s": round(res["gold_runtime_s"], 6), "n_rows": res["n_rows"],
                            "gold_runtime_rerun_s": None if res["gold_runtime_rerun_s"] is None
                            else round(res["gold_runtime_rerun_s"], 6)})
            if res["detail"]:
                errors.append({"question_id": dropped[-1]["question_id"], "db_id": r["db_id"], "error": res["detail"]})
    # arctic_filter.jsonl holds exactly the QUC 11 fields; SQLite messages go to a side file for reading by hand.
    for path, items in ((out_path, dropped), (errors_path(out_path), errors)):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            for d in items:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        os.replace(tmp, path)

    summary = {"n_total": len(rows)}
    for reason in REASONS:
        summary[f"n_{reason}"] = sum(d["reason"] == reason for d in dropped)
    summary["n_kept"] = len(rows) - len(dropped)
    summary["n_rerun"] = sum(done[i]["gold_runtime_rerun_s"] is not None for i in range(len(rows)))
    summary["n_rerun_kept"] = sum(done[i]["gold_runtime_rerun_s"] is not None and done[i]["reason"] is None
                                  for i in range(len(rows)))
    summary["near_threshold_3_to_5s"] = sum(3.0 <= done[i]["gold_runtime_s"] < timeout_s and done[i]["reason"] is None
                                            for i in range(len(rows)))
    summary["timeout_s"] = timeout_s
    os.remove(progress_path)
    return summary


def write_filtered(src_path, filter_path, out_path) -> int:
    """Giữ các dòng không bị loại; thêm question_id = row id của file src (QUC §3.4).

    The filter holds train:<i>; the same i is dropped from src (train, sft_train or grpo_train,
    which share line order). db_id of every dropped line is checked against src. Returns lines kept."""
    rows = read_rows(src_path)
    drop: Dict[int, str] = {}
    for d in read_rows(filter_path):
        stem, _, idx = d["question_id"].rpartition(":")
        i = int(idx)
        if not stem or i in drop or not 0 <= i < len(rows):
            raise ValueError(f"bad or repeated question_id in filter: {d['question_id']}")
        if rows[i]["db_id"] != d["db_id"]:
            raise ValueError(f"{src_path} line {i} is {rows[i]['db_id']}, filter says {d['db_id']}: line order differs")
        drop[i] = d["db_id"]
    os.makedirs(os.path.dirname(os.path.abspath(str(out_path))), exist_ok=True)
    kept = 0
    tmp = str(out_path) + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for i, r in enumerate(rows):
            if i in drop:
                continue
            if "question_id" in r:
                raise ValueError(f"{src_path} line {i} already has question_id")
            f.write(json.dumps(dict(question_id=row_id(src_path, i), **r), ensure_ascii=False) + "\n")
            kept += 1
    os.replace(tmp, out_path)
    return kept


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", help="train.jsonl to filter")
    p.add_argument("--db_dir", default="/content/bird_db")
    p.add_argument("--apply", help="write a filtered copy of this file instead of running gold")
    p.add_argument("--filter", help="arctic_filter.jsonl, with --apply")
    p.add_argument("--out", required=True)
    p.add_argument("--timeout_s", type=float, default=5.0)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    if a.apply:
        print(json.dumps({"kept": write_filtered(a.apply, a.filter, a.out)}))
    else:
        print(json.dumps(filter_arctic(a.train, a.db_dir, a.out, a.timeout_s, a.limit)))


if __name__ == "__main__":
    sys.exit(main())
