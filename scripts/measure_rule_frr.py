"""S6 W1: false rejection rate (FRR) of the team's keyword-matching rules
("luật so khớp từ khóa của nhóm", U22) and of the DB probe, on gold-correct SQL.

Usage (from repo root):
    python scripts/measure_rule_frr.py <data.jsonl> <db_dir> <out/ast_frr.json> [--limit N]

Rules (QUC §14): no gold in prompts, no model, no cal-B, no rule edits.
Rules and probe are reported SEPARATELY.
"""
import argparse
import collections
import importlib
import inspect
import json
import os
import pkgutil
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.validation import pv_sql_rules, db_probe  # noqa: E402
from src.validation.pv_sql_rules import PVSQLValidator  # noqa: E402
from src.validation.db_probe import DBProbeEngine  # noqa: E402

GOLD_TIMEOUT_S = 30  # QUC §7 / execute_query_safely


# ----------------------------------------------------------------- reading
def read_rows(path):
    """QUC §3.4: split on LF only (binary), never splitlines(); a blank line in
    the middle stops the run (report to Leader). Returns [(line_no, dict)]."""
    raw = open(path, "rb").read().split(b"\n")
    if raw and raw[-1] == b"":
        raw = raw[:-1]
    rows = []
    for i, b in enumerate(raw):
        if not b.strip():
            raise ValueError(f"blank line at {i} in {path}: stop and report to Leader (QUC §3.4)")
        rows.append((i, json.loads(b.decode("utf-8"))))
    return rows


def _gold_sql(r):
    for k in ("SQL", "sql", "gold_sql", "query"):
        if k in r and r[k]:
            return r[k]
    raise KeyError(f"no gold SQL field in row keys {sorted(r)}")


def _db_path(db_dir, db_id):
    cands = [os.path.join(db_dir, f"{db_id}.sqlite"),
             os.path.join(db_dir, db_id, f"{db_id}.sqlite")]
    for c in cands:
        if os.path.exists(c):
            return c
    d = os.path.join(db_dir, db_id)
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith(".sqlite"):
                return os.path.join(d, f)
    raise FileNotFoundError(f"database for {db_id} not found under {db_dir} (tried {cands})")


# ------------------------------------------------------- gold executability
def _find_execute_query_safely():
    """Locate the repo's execute_query_safely (QUC §7 names metrics.py:63-71)."""
    try:
        m = importlib.import_module("src.evaluation.metrics")
        if hasattr(m, "execute_query_safely"):
            return m.execute_query_safely
    except ImportError:
        pass
    for pkg_name in ("src.evaluation", "src.validation"):
        try:
            pkg = importlib.import_module(pkg_name)
        except ImportError:
            continue
        for mi in pkgutil.iter_modules(pkg.__path__):
            try:
                m = importlib.import_module(f"{pkg_name}.{mi.name}")
            except Exception:
                continue
            if hasattr(m, "execute_query_safely"):
                return m.execute_query_safely
    raise RuntimeError("execute_query_safely not found; do not fall back to a mock (QUC §14.2). Report to Leader.")


def _interpret(res):
    """True if the call succeeded. Handles dict / (rows, err) / (err, rows) / rows."""
    if res is None:
        return False
    if isinstance(res, dict):
        if res.get("error"):
            return False
        for k in ("success", "ok"):
            if k in res:
                return bool(res[k])
        return True
    if isinstance(res, (tuple, list)) and len(res) == 2:
        a, b = res
        if isinstance(a, bool):
            return a
        if b is None:
            return True
        if isinstance(b, str) and b:
            return False
        if a is None or (isinstance(a, str) and a):
            return not (isinstance(a, str) and a)
        return True
    return True


def default_gold_executor():
    """Returns f(db_path, sql) -> bool (gold runs within 30 s)."""
    fn = _find_execute_query_safely()
    params = list(inspect.signature(fn).parameters)
    db_name = next((p for p in params if "db" in p.lower() or "path" in p.lower()), None)
    sql_name = next((p for p in params if "sql" in p.lower() or "query" in p.lower()), None)
    to_name = next((p for p in params if "timeout" in p.lower()), None)
    if db_name is None or sql_name is None:
        raise RuntimeError(f"cannot map execute_query_safely{inspect.signature(fn)}; adapt default_gold_executor")

    def run(db_path, sql):
        kw = {db_name: db_path, sql_name: sql}
        if to_name:
            kw[to_name] = GOLD_TIMEOUT_S
        try:
            return _interpret(fn(**kw))
        except Exception:
            return False

    return run


# ------------------------------------------------------- probe classification
def _blocked_words():
    words = set()
    for src in (db_probe, DBProbeEngine):
        v = getattr(src, "DESTRUCTIVE_KEYWORDS", None)
        if v:
            words |= {str(w).upper() for w in v}
    words.add("REPLACE")  # QUC/phiếu: REPLACE() is blocked too
    return words


def classify_probe_error(sql, err):
    """blocked_keyword | timeout | empty | sqlite_error (phiếu §3)."""
    toks = set(re.findall(r"[A-Za-z_]+", sql or "")) 
    if {t.upper() for t in toks} & _blocked_words():
        return "blocked_keyword"
    e = (str(err) if err else "").lower()
    if "block" in e or "forbidden" in e or "destructive" in e or "keyword" in e:
        return "blocked_keyword"
    if "timeout" in e or "timed out" in e or "interrupt" in e:
        return "timeout"
    if "empty" in e or "no row" in e or "no result" in e:
        return "empty"
    return "sqlite_error"


def _rule_names(vio):
    """Map each violation to a name from RULE_NAMES when possible (keep the exact string)."""
    names = [str(n) for n in getattr(pv_sql_rules, "RULE_NAMES", [])]
    out = []
    for v in (vio or []):
        s = str(v)
        hit = next((n for n in names if n == s), None) or next((n for n in names if n in s), None)
        out.append(hit or s)
    return out


# ----------------------------------------------------------------- measure
def _run(data_path, db_dir, limit=None, gold_executor=None):
    gold_executor = gold_executor or default_gold_executor()
    stem = os.path.splitext(os.path.basename(data_path))[0]
    v, p = PVSQLValidator(), DBProbeEngine()
    n = n_failed = rule_rej = probe_rej = 0
    by_rule, by_error = collections.Counter(), collections.Counter()
    details = []
    for i, r in read_rows(data_path):
        if limit is not None and i >= limit:
            break
        qid = r.get("question_id", f"{stem}:{i}")
        sql = _gold_sql(r)
        db = _db_path(db_dir, r["db_id"])
        if not gold_executor(db, sql):
            n_failed += 1          # not in n
            continue
        n += 1
        p.reset_counter()          # before EVERY question (limit 5 per counter)
        ok, vio, _ = v.validate_rules(r["question"], sql)   # 3 values (not pv.validate)
        good, _, err = p.probe_sql(sql, db)
        names = _rule_names(vio) if not ok else []
        ecls = None
        if not ok:
            rule_rej += 1
            for name in names:     # one question may hit several rules
                by_rule[name] += 1
        if not good:
            probe_rej += 1
            ecls = classify_probe_error(sql, err)
            by_error[ecls] += 1
        if not ok or not good:
            details.append({"question_id": qid, "db_id": r["db_id"],
                            "rule_rejected": not ok, "rules": names,
                            "probe_rejected": not good, "probe_error_class": ecls,
                            "probe_err": None if err is None else str(err),
                            "question": r["question"], "SQL": sql})
    frr = lambda k: (k / n) if n else 0.0
    result = {"n": n, "false_rejections": rule_rej, "frr": frr(rule_rej),
              "frr_pct": 100.0 * frr(rule_rej),
              "by_rule": dict(by_rule),
              "probe": {"n": n, "false_rejections": probe_rej, "frr": frr(probe_rej),
                        "frr_pct": 100.0 * frr(probe_rej), "by_error": dict(by_error)},
              "n_gold_failed": n_failed}
    return result, details


def measure_rule_frr(data_path, db_dir, limit=None, gold_executor=None) -> dict:
    """ast_frr.json: {"n", "false_rejections", "frr", "by_rule": {luật: số},
       "probe": {"n", "false_rejections", "frr", "by_error": {}}, "n_gold_failed"}"""
    return _run(data_path, db_dir, limit, gold_executor)[0]


def _commit():
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_path")
    ap.add_argument("db_dir")
    ap.add_argument("out_path")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", default="cal_a")
    a = ap.parse_args()
    res, details = _run(a.data_path, a.db_dir, a.limit)
    res.update({"split": a.split, "seed": 42, "commit": _commit(),
                "note": "measured from members/m6/work (W1 check number, not a QA-imported result)"})
    os.makedirs(os.path.dirname(os.path.abspath(a.out_path)), exist_ok=True)
    with open(a.out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    dpath = a.out_path[:-5] + "_details.json" if a.out_path.endswith(".json") else a.out_path + ".details"
    with open(dpath, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print("details ->", dpath, f"({len(details)} rejected rows)")


if __name__ == "__main__":
    main()
