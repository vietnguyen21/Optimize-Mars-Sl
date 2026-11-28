"""S5.1 generator-only evaluation (B0-B5): K=1, no grounding model, no System-1.

Contract (QUC 26/11/2026):
  * the generator receives (question, schema, evidence) only; gold SQL is read once, after
    every prediction is fixed, by the scoring pass (QUC s14 item 2);
  * EX = set(pred_rows) == set(gold_rows), 30 s per query; error, empty, overlength = wrong (s7);
  * row id = "<file stem>:<0-based line of the ORIGINAL file>", written to ``question_id`` (s3.4);
  * output files follow QUC s11 (predictions.jsonl, metrics.json, cost.json);
  * the run is resumable: generations are appended to ``generations.partial.jsonl`` per question;
  * nothing is overwritten: result files are created exclusively ('x').
"""
import hashlib
import json
import math
import os
import random
import re
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.evaluation.metrics import compare_sql_results, execute_query_safely

TIMEOUT_S = 30.0
CAL_A_SIZE = 662
CAL_A_300_SIZE = 300
CAL_A_300_SEED = 42
# QUC s3.3 reference values (computed with Python 3.10 and 3.12).
CAL_A_300_FIRST5 = [0, 1, 3, 6, 7]
CAL_A_300_LAST5 = [657, 658, 659, 660, 661]
CAL_A_300_IDS_SHA256 = "2da9a10fda38a0ea55cd6a8366d2088933812d0ec2fb9800e71cd99e643de441"
GEN_RESULT_KEYS = {"sql", "tokens_in", "tokens_out", "overlength"}  # PHIEU_W1 signature
PREDICTION_FIELDS = ["question_id", "db_id", "pred_sql", "exec_ok", "ex", "n_calls",
                     "tokens_in", "tokens_out", "latency_s", "peak_vram_bytes"]
REQUIRED_META = ("baseline", "split", "checkpoint", "commit", "seed", "prompt_format", "gpu")


# --------------------------------------------------------------------------- data

def read_jsonl_lf(path) -> List[Dict[str, Any]]:
    """Read a JSONL file splitting on LF only (QUC s3.4): never str.splitlines(), which would also
    split on U+2028/CR. A blank line in the middle is an error (report it to the Leader)."""
    parts = Path(path).read_bytes().split(b"\n")
    if parts and parts[-1] == b"":
        parts.pop()
    rows = []
    for i, part in enumerate(parts):
        if not part.strip():
            raise ValueError(f"{path}: blank line at row {i}; stop and report to the Leader")
        rows.append(json.loads(part.decode("utf-8")))
    return rows


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cal_a_300_indices() -> List[int]:
    """QUC s3.3: simple random sample, not stratified, never changed after creation."""
    return sorted(random.Random(CAL_A_300_SEED).sample(range(CAL_A_SIZE), CAL_A_300_SIZE))


def cal_a_300_ids_text(indices: List[int]) -> str:
    return "".join(f"retrieval_dev:{i}\n" for i in indices)


def check_cal_a_300_reference(indices: List[int], ids_text: str) -> None:
    """Different from the Leader's reference values: stop and report (QUC s3.3)."""
    sha = hashlib.sha256(ids_text.encode("utf-8")).hexdigest()
    if (indices[:5] != CAL_A_300_FIRST5 or indices[-5:] != CAL_A_300_LAST5
            or len(indices) != CAL_A_300_SIZE or sha != CAL_A_300_IDS_SHA256):
        raise ValueError(f"cal-A 300 differs from the QUC s3.3 reference (first5={indices[:5]}, "
                         f"last5={indices[-5:]}, sha256={sha}); stop and report to the Leader")


def verified_cal_a_300(data_path) -> Tuple[List[int], str]:
    """0-based line indices of cal-A 300 in retrieval_dev.jsonl, after the QUC s3.3 checks; plus the
    sha256 of retrieval_dev.jsonl. Stops (raises) if the file or the sample differs from the reference."""
    rows = read_jsonl_lf(data_path)
    if len(rows) != CAL_A_SIZE:
        raise ValueError(f"{data_path} has {len(rows)} rows, expected {CAL_A_SIZE}; stop and report to the Leader")
    indices = cal_a_300_indices()
    check_cal_a_300_reference(indices, cal_a_300_ids_text(indices))
    return indices, sha256_file(data_path)


# --------------------------------------------------------------------------- schema

def _table_ddls(db_path: str) -> List[Tuple[str, str]]:
    import sqlite3
    conn = sqlite3.connect(f"file:{os.path.abspath(db_path)}?mode=ro", uri=True)
    try:
        return conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL").fetchall()
    finally:
        conn.close()


def full_schema(row: Dict[str, Any], db_dir: str) -> str:
    """Full schema = every CREATE TABLE (QUC s5); the same function that built sft_train.jsonl."""
    from scripts.prepare_bird import schema_ddl
    from src.training.runtime import resolve_db
    return schema_ddl(resolve_db(db_dir, row["db_id"]))


def sprime_schema(db_path: str, tables: List[str]) -> str:
    """S' prompt = the verbatim CREATE TABLE of S1's ``tables`` (QUC s11, U14), in database order."""
    wanted = set(tables)
    found = [(name, sql) for name, sql in _table_ddls(db_path) if name in wanted]
    missing = wanted - {name for name, _ in found}
    if missing:
        raise ValueError(f"S' lists tables absent from {db_path}: {sorted(missing)}")
    return "\n\n".join(sql.strip() + ";" for _, sql in found)


def make_sprime_schema_fn(sprime_tables: Dict[str, List[str]], db_dir: str) -> Callable[[Dict[str, Any], str], str]:
    """schema_fn(row, row_id) reading S' by row id; a missing id raises (no fallback to the full schema)."""
    from src.training.runtime import resolve_db

    def schema_fn(row: Dict[str, Any], row_id: str) -> str:
        if row_id not in sprime_tables:
            raise ValueError(f"No S' entry for {row_id}; do not substitute the full schema")
        return sprime_schema(resolve_db(db_dir, row["db_id"]), sprime_tables[row_id])

    return schema_fn


def load_sprime(path) -> Dict[str, List[str]]:
    """question_id -> tables from an S1 file (s_prime_<split>.jsonl). Fields per QUC s11."""
    out: Dict[str, List[str]] = {}
    for row in read_jsonl_lf(path):
        for key in ("question_id", "db_id", "tables"):
            if key not in row:
                raise ValueError(f"{path}: row lacks {key!r}")
        if row["question_id"] in out:
            raise ValueError(f"{path}: duplicate question_id {row['question_id']}")
        out[row["question_id"]] = list(row["tables"])
    return out


# --------------------------------------------------------------------------- cuda helpers

def _torch_cuda():
    try:
        import torch
    except ImportError:
        return None
    return torch if torch.cuda.is_available() else None


def _cuda_begin(torch) -> None:
    if torch is not None:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _cuda_end(torch) -> Optional[int]:
    if torch is None:
        return None
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


# --------------------------------------------------------------------------- real generator

def extract_sql_from_generation(text: str) -> str:
    """Extract SQL from <answer>...</answer> (even if truncated before </answer>) with markdown fallback."""
    match = re.search(r"<answer>(.*?)(?:</answer>|$)", text, re.S)
    if match:
        return match.group(1).strip()
    md_match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.IGNORECASE)
    return md_match.group(1).strip() if md_match else ""


def make_hf_generate_fn(model, tokenizer, prompt_format: str = "v2", max_prompt_length: int = 2048,
                        max_new_tokens: int = 256) -> Callable[[str, str, str], Dict[str, Any]]:
    """Greedy K=1 generation with the shared prompt (src.training.runtime.build_prompt).

    Mirrors the greedy branch of CandidateGenerator._generate_with_model (do_sample=False,
    repetition_penalty=1.0). An overlong prompt is not generated: it is returned as overlength."""
    import torch
    from src.training.runtime import build_prompt
    found = getattr(model, "_prompt_format", None)
    if found is None and hasattr(model, "peft_config"):
        found = "v1"  # an untagged adapter predates v2
    if found is not None and found != prompt_format:
        raise ValueError(f"Model adapter uses prompt_format {found}, not {prompt_format}; select {found} explicitly")
    model.eval()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    target_device = next(model.parameters()).device

    def generate(question: str, schema: str, evidence: str) -> Dict[str, Any]:
        prompt = build_prompt(question, schema, evidence, prompt_format=prompt_format)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        n_in = int(inputs["input_ids"].shape[1])
        if n_in > max_prompt_length:
            return {"sql": "", "tokens_in": n_in, "tokens_out": 0, "overlength": True}
        inputs = {k: v.to(target_device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 repetition_penalty=1.0, pad_token_id=pad_id,
                                 eos_token_id=tokenizer.eos_token_id)
        generated = out[0, n_in:]
        ids = generated.tolist()
        if tokenizer.eos_token_id in ids:
            ids = ids[:ids.index(tokenizer.eos_token_id) + 1]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        sql = extract_sql_from_generation(text)
        return {"sql": sql, "tokens_in": n_in, "tokens_out": len(ids), "overlength": False}

    return generate


# --------------------------------------------------------------------------- statistics

def percentile_ceil(values: List[float], q: float = 0.9) -> float:
    """QUC s9 / U9: the ceil(q*n)-th smallest value, counted from 1."""
    ordered = sorted(values)
    return ordered[max(1, math.ceil(q * len(ordered))) - 1]


def build_cost(rows: List[Dict[str, Any]], ex: float) -> Dict[str, Any]:
    latencies = [r["latency_s"] for r in rows]
    peaks = [r["peak_vram_bytes"] for r in rows if r["peak_vram_bytes"] is not None]
    peak_gb = max(peaks) / 2 ** 30 if peaks else None
    median = statistics.median(latencies)
    return {"calls_per_q": statistics.fmean(r["n_calls"] for r in rows),
            "tokens_per_q": statistics.fmean(r["tokens_in"] + r["tokens_out"] for r in rows),
            "latency_median_s": median, "latency_p90_s": percentile_ceil(latencies),
            "peak_vram_gb": peak_gb,
            "ex_per_gb": ex / peak_gb if peak_gb else None,
            "ex_per_s": ex / median if median else None}


# --------------------------------------------------------------------------- run

def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def _load_partial(path: Path) -> List[Dict[str, Any]]:
    """Read the append-only partial file; a torn last line (crash mid-write) is cut off."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.split(b"\n")
    keep, good = [], 0
    for i, line in enumerate(lines):
        if i == len(lines) - 1 and line == b"":
            break
        try:
            keep.append(json.loads(line.decode("utf-8")))
            good += len(line) + 1
        except ValueError:
            if i != len(lines) - 1:
                raise ValueError(f"{path}: corrupt line {i} in the middle of the file; stop and report")
            break
    if good != len(raw):
        with open(path, "r+b") as f:
            f.truncate(good)
    return keep


def evaluate_generator_only(data_path, line_indices, db_dir, generate_fn, schema_fn, out_dir, meta,
                            limit=None, log_every: int = 25) -> Dict[str, Any]:
    """K=1. Writes predictions.jsonl, metrics.json, cost.json once; returns metrics.

    data_path     JSONL file; ``line_indices`` are 0-based lines of that ORIGINAL file (QUC s3.4); the row
                  id is ``<file stem>:<line>`` and goes to ``question_id``.
    generate_fn   (question, schema, evidence) -> {"sql", "tokens_in", "tokens_out", "overlength"}; no gold.
    schema_fn     (row, row_id) -> schema text (``full_schema`` or ``make_sprime_schema_fn(...)``).
    meta          baseline, split, checkpoint, commit, seed, prompt_format, gpu (gpu may be None on CPU).
    A rerun on the same ``out_dir`` resumes from generations.partial.jsonl and raises if ``meta`` differs.
    """
    from src.training.runtime import resolve_db
    from src.evaluation.metrics import compute_execution_accuracy
    missing = [k for k in REQUIRED_META if k not in meta]
    if missing:
        raise ValueError(f"meta lacks {missing}")
    rows = read_jsonl_lf(data_path)
    stem = Path(data_path).stem
    indices = list(line_indices)[:limit] if limit is not None else list(line_indices)
    if not indices:
        raise ValueError("Evaluation set is empty")
    if len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= len(rows):
        raise ValueError("line_indices must be unique and inside the data file")
    records = [dict(rows[i], question_id=f"{stem}:{i}") for i in indices]
    qids = [r["question_id"] for r in records]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "predictions.jsonl").exists():
        raise FileExistsError(f"{out / 'predictions.jsonl'} exists: this run is complete; use a new out_dir")
    meta = dict(meta, n=len(records), timeout_s=TIMEOUT_S, limit=limit,
                question_ids_sha256=hashlib.sha256("\n".join(qids).encode()).hexdigest())
    meta_path = out / "meta.json"
    if meta_path.exists():
        if _canon(json.loads(meta_path.read_text("utf-8"))) != _canon(meta):
            raise ValueError(f"{meta_path} differs from this run's meta; start a new out_dir or restore the settings")
    else:
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    partial_path = out / "generations.partial.jsonl"
    done = {row["question_id"]: row for row in _load_partial(partial_path)}
    unknown = set(done) - set(qids)
    if unknown:
        raise ValueError(f"{partial_path} holds question_ids outside this run: {sorted(unknown)[:3]}")

    torch = _torch_cuda()
    with open(partial_path, "ab") as sink:
        for index, rec in enumerate(records):
            qid = rec["question_id"]
            if qid in done:
                continue
            schema = schema_fn(rec, qid)
            # Explicit allowlist: neither gold SQL nor labels can reach generation.
            _cuda_begin(torch)
            started = time.perf_counter()
            result = generate_fn(rec["question"], schema, rec.get("evidence", "") or "")
            if torch is not None:
                torch.cuda.synchronize()
            latency = time.perf_counter() - started
            peak = _cuda_end(torch)
            if set(result) != GEN_RESULT_KEYS:
                raise ValueError(f"generate_fn must return exactly {sorted(GEN_RESULT_KEYS)}, got {sorted(result)}")
            overlength = bool(result["overlength"])
            row = {"question_id": qid, "db_id": rec["db_id"],
                   "pred_sql": "" if overlength else str(result["sql"] or ""),
                   "n_calls": 0 if overlength else 1, "tokens_in": int(result["tokens_in"]),
                   "tokens_out": int(result["tokens_out"]), "latency_s": latency,
                   "peak_vram_bytes": peak, "overlength": overlength}
            sink.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
            sink.flush()
            os.fsync(sink.fileno())
            done[qid] = row
            if log_every and (index + 1) % log_every == 0:
                print(f"[generator_only] {index + 1}/{len(records)}", flush=True)

    # Scoring pass: the only place gold SQL is read (QUC s14 item 2).
    gens = [done[q] for q in qids]
    db_paths = [resolve_db(db_dir, r["db_id"]) for r in records]
    golds = [r.get("SQL", r.get("sql", "")) for r in records]
    if any(not g for g in golds):
        raise ValueError("Evaluation records must carry gold SQL for scoring")
    _, matches = compute_execution_accuracy([g["pred_sql"] for g in gens], golds, db_paths)
    final = []
    for gen, db_path, match in zip(gens, db_paths, matches):
        exec_ok, _ = execute_query_safely(gen["pred_sql"], db_path, TIMEOUT_S)
        final.append(dict(gen, exec_ok=bool(exec_ok), ex=bool(match)))
    n = len(final)
    ex = 100.0 * sum(r["ex"] for r in final) / n
    metrics = {"baseline": meta["baseline"], "split": meta["split"], "n": n, "ex": ex, "ves": None,
               "timeout_s": TIMEOUT_S, "checkpoint": meta["checkpoint"], "commit": meta["commit"],
               "seed": meta["seed"], "n_overlength": sum(r["overlength"] for r in final)}
    cost = build_cost(final, ex)
    # Exclusive creation prevents overwriting an earlier experiment.
    with open(out / "predictions.jsonl", "x", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps({k: r[k] for k in PREDICTION_FIELDS}, ensure_ascii=False) + "\n")
    for name, payload in (("metrics.json", metrics), ("cost.json", cost)):
        with open(out / name, "x", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return metrics
