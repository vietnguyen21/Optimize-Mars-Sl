"""
Predicted-plan experiment (B1): a prompt-only planner replaces the gold plan of B2.
Reuses the generator trained on gold plans, so no training is needed. This is a quick check: a
generator trained on predicted plans could do better. Compares against the stored B0 and B2 runs
on the same dev sample, and reports plan quality per field.

Usage: python scripts/plan_predicted_experiment.py --data_dir /content/bird/processed
"""

import argparse
import json
import os
import random
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.plan_oracle_experiment import load_with_plans
from src.common.query_plan import PLAN_FIELDS, format_plan, parse_plan
from src.evaluation import EvaluationHarness, compute_execution_accuracy, bootstrap_paired_ci, mcnemar_exact_p
from src.evaluation.evaluate_end_to_end import load_schema_ddl

PLANNER_PROMPT = """You plan SQL queries. Given a database schema and a question, output only a JSON object \
describing the outermost query, with exactly these keys:
aggregation: "NONE", "COUNT", "SUM", "AVG", "MIN" or "MAX"; group_by, having, order_by, limit, distinct, subquery: \
true or false; direction: "ASC", "DESC" or "NONE".

Example question: List the 5 customers with the most orders.
Example output: {{"aggregation":"COUNT","group_by":true,"having":false,"order_by":true,"direction":"DESC",\
"limit":true,"distinct":false,"subquery":false}}

Schema:
{schema}

Evidence: {evidence}
Question: {question}"""


def load_dev_sample(data_dir, train_size, dev_size, seed):
    """Rebuilds the exact dev sample of plan_oracle_experiment: same seed, same RNG call order."""
    train, _ = load_with_plans(os.path.join(data_dir, "train.jsonl"))
    dev, _ = load_with_plans(os.path.join(data_dir, "dev.jsonl"))
    rng = random.Random(seed)
    rng.sample(train, train_size)
    return rng.sample(dev, dev_size)


def predict_plans(records, db_dir, output_path, model_name, max_prompt_tokens):
    """Appends one prediction per line, so a restart continues where it stopped."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    done = []
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            done = [json.loads(line) for line in f if line.strip()]
    if len(done) >= len(records):
        return done

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="auto").eval()
    for item in records[len(done):]:
        ddl = load_schema_ddl(os.path.join(db_dir, f"{item['db_id']}.sqlite"))
        ddl_ids = tokenizer(ddl, add_special_tokens=False)["input_ids"][:max_prompt_tokens]
        content = PLANNER_PROMPT.format(schema=tokenizer.decode(ddl_ids), evidence=item.get("evidence", ""),
                                        question=item["question"])
        text = tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=96, do_sample=False)
        raw = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        row = {"raw": raw, "plan": parse_plan(raw)}
        done.append(row)
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    del model
    torch.cuda.empty_cache()
    return done


def plan_quality(predicted, gold):
    """JSON validity, per-field accuracy over valid plans, and share of fully correct plans."""
    valid = [(p, g) for p, g in zip(predicted, gold) if p is not None]
    per_field = {f: round(sum(p[f] == g[f] for p, g in valid) / max(1, len(valid)), 4) for f in PLAN_FIELDS}
    return {"json_validity": round(len(valid) / len(gold), 4),
            "per_field_accuracy": per_field,
            "exact_plan_rate": round(sum(p == g for p, g in valid) / len(gold), 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/content/bird/processed")
    parser.add_argument("--drive_dir", default="/content/drive/MyDrive/Text2SQL_Colab/plan_exp")
    parser.add_argument("--out", default="/content/plan_b1")
    parser.add_argument("--planner", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--train_size", type=int, default=500)
    parser.add_argument("--dev_size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_schema_tokens", type=int, default=3000)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    db_dir = os.path.join(args.data_dir, "db")
    dev = load_dev_sample(args.data_dir, args.train_size, args.dev_size, args.seed)

    with open(os.path.join(args.drive_dir, "B2_gold_plan", "finished.json")) as f:
        b2 = json.load(f)
    with open(os.path.join(args.drive_dir, "B0_no_plan", "finished.json")) as f:
        b0 = json.load(f)
    assert len(b2["outcomes"]) == len(dev)

    plans_path = os.path.join(args.drive_dir, "B1_predicted_plans.jsonl")
    rows = predict_plans(dev, db_dir, plans_path, args.planner, args.max_schema_tokens)
    predicted = [r["plan"] for r in rows]
    gold_plans = [json.loads(r["plan"]) for r in dev]
    quality = plan_quality(predicted, gold_plans)
    print("PLAN_QUALITY", json.dumps(quality), flush=True)

    # An invalid plan becomes an empty plan: the system must still answer.
    eval_records = [{**r, "plan": format_plan(p) if p else ""} for r, p in zip(dev, predicted)]
    dev_path = os.path.join(args.out, "dev_b1.jsonl")
    with open(dev_path, "w", encoding="utf-8") as f:
        for r in eval_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    harness = EvaluationHarness(db_dir=db_dir, num_candidates=2,
                                model_path=os.path.join(args.drive_dir, "B2_gold_plan", "best_model"),
                                output_log_dir=args.out, output_report_dir=args.out, use_record_plan=True)
    res = harness.evaluate_dataset(dev_path)
    with open(os.path.join(args.out, "evaluation_results.jsonl"), encoding="utf-8") as f:
        logs = [json.loads(line) for line in f]
    b1 = compute_execution_accuracy([l["predicted_sql"] for l in logs], [l["gold_sql"] for l in logs],
                                    [os.path.join(db_dir, f"{l['db_id']}.sqlite") for l in logs])[1]

    def compare(a, b):
        diff, low, high = bootstrap_paired_ci(a, b)
        return {"ex_diff_points": diff, "ci95": [low, high], "mcnemar_p": mcnemar_exact_p(a, b)}

    results = {"plan_quality": quality, "planner": args.planner, "generator": "trained on gold plans",
               "b1": {"ex": res["execution_accuracy"], "vsr": res["valid_sql_rate"], "latency": res["latency"]},
               "b1_minus_b0": compare(b1, b0["outcomes"]), "b2_minus_b1": compare(b2["outcomes"], b1), "n": len(b1)}
    with open(os.path.join(args.drive_dir, "b1_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("COMPARISON", json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
