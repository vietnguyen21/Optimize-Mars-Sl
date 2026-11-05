"""
Rejection Fine-Tuning (RFT) dataset builder, the PA3 training step after JEPA-SFT:
  1. Sample K SQL per training question from the current generator.
  2. Keep SQL that matches the gold result and passes the 10 PV-SQL AST rules.
  3. Deduplicate and keep at most 2 distinct SQL per question.
  4. The output JSONL is SFT data: train on it with train_sft_generator.

Gold SQL is used only to judge samples, never in the prompt. Output is appended per question,
so an interrupted Colab session resumes from the last finished question.
"""

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.common.schemas import SQLCandidate
from src.evaluation.candidate_generator import CandidateGenerator, load_generator
from src.evaluation.evaluate_end_to_end import load_schema_ddl
from src.evaluation.metrics import compare_sql_results, execute_query_safely
from src.validation import PVSQLValidator


def filter_candidates(
    candidates: List[SQLCandidate],
    question: str,
    gold_sql: str,
    db_path: str,
    validator: Any,
    max_per_question: int = 2
) -> List[SQLCandidate]:
    """Keeps correct, rule-passing, distinct candidates, at most max_per_question."""
    gold_ok, gold_rows = execute_query_safely(gold_sql, db_path)
    if not gold_ok:
        return []
    kept, seen = [], set()
    for cand in candidates:
        key = re.sub(r"\s+", " ", cand.sql).strip().rstrip(";").lower()
        if not key or key in seen:
            continue
        seen.add(key)
        ok, rows = execute_query_safely(cand.sql, db_path)
        if ok and compare_sql_results(rows, gold_rows) and validator.validate(question, cand.sql)[0]:
            kept.append(cand)
            if len(kept) == max_per_question:
                break
    return kept


def build_rft_dataset(
    generator: Any,
    records: List[Dict[str, Any]],
    db_dir: str,
    output_path: str,
    num_samples: int = 8,
    max_per_question: int = 2
) -> Dict[str, int]:
    """Appends filtered samples to output_path; a .done file stores how many questions finished."""
    done_path = output_path + ".done"
    start = int(open(done_path).read()) if os.path.exists(done_path) else 0
    validator = PVSQLValidator()
    written = answered = 0
    for idx in range(start, len(records)):
        item = records[idx]
        db_path = os.path.join(db_dir, f"{item['db_id']}.sqlite")
        candidates = generator.generate(
            item["question"],
            num_candidates=num_samples,
            evidence=item.get("evidence", ""),
            schema_ddl=load_schema_ddl(db_path)
        )
        kept = filter_candidates(candidates, item["question"], item["SQL"], db_path, validator, max_per_question)
        answered += bool(kept)
        with open(output_path, "a", encoding="utf-8") as f:
            for cand in kept:
                f.write(json.dumps({
                    "db_id": item["db_id"],
                    "question": item["question"],
                    "evidence": item.get("evidence", ""),
                    "SQL": cand.sql,
                    "think_steps": cand.reasoning
                }, ensure_ascii=False) + "\n")
        written += len(kept)
        with open(done_path, "w") as f:
            f.write(str(idx + 1))
    return {"questions": len(records) - start, "questions_with_sample": answered, "samples": written}


def main():
    parser = argparse.ArgumentParser(description="Build an RFT dataset from a trained generator")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True, help="BIRD-style JSONL (question, evidence, SQL, db_id)")
    parser.add_argument("--db_dir", required=True)
    parser.add_argument("--output_path", required=True, help="Output JSONL; use a Drive path to survive disconnects")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--max_per_question", type=int, default=2)
    args = parser.parse_args()

    with open(args.data_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    model, tokenizer = load_generator(args.model_path)
    generator = CandidateGenerator(model=model, tokenizer=tokenizer)
    print(json.dumps(build_rft_dataset(generator, records, args.db_dir, args.output_path,
                                       args.num_samples, args.max_per_question)))


if __name__ == "__main__":
    main()
