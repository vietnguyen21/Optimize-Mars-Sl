"""
Oracle experiment: is a QueryPlan worth adding?
Trains two generators under identical conditions, without a plan (B0) and with the gold-derived
plan (B2), then compares them on BIRD-dev. B2 is an upper bound, not a system result. If even a
perfect plan does not help, a predicted plan will not either.

Checkpoints and finished arms are stored in --drive_dir, so rerunning the same command resumes
after a disconnect: finished arms are skipped and training continues from the last checkpoint.

Usage: python scripts/plan_oracle_experiment.py --data_dir /content/bird/processed --out /content/plan_exp
"""

import argparse
import json
import os
import random
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.common.query_plan import extract_plan, format_plan
from src.evaluation import EvaluationHarness, compute_execution_accuracy, bootstrap_paired_ci, mcnemar_exact_p
from src.training.train_sft_generator import run_sft_training


def load_with_plans(path):
    """Returns (records with a `plan` field, number of unparsable SQL)."""
    kept, failed = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            try:
                item["plan"] = format_plan(extract_plan(item["SQL"]))
                kept.append(item)
            except Exception:
                failed += 1
    return kept, failed


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/content/bird/processed")
    parser.add_argument("--out", default="/content/plan_exp")
    parser.add_argument("--drive_dir", default="/content/drive/MyDrive/Text2SQL_Colab/plan_exp")
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--train_size", type=int, default=800)
    parser.add_argument("--dev_size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    args = parser.parse_args()

    if not os.path.isdir(os.path.dirname(args.drive_dir)):
        raise FileNotFoundError(f"Drive folder missing, mount Drive first: {args.drive_dir}")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.drive_dir, exist_ok=True)
    db_dir = os.path.join(args.data_dir, "db")
    train, train_failed = load_with_plans(os.path.join(args.data_dir, "train.jsonl"))
    dev, dev_failed = load_with_plans(os.path.join(args.data_dir, "dev.jsonl"))
    print(json.dumps({"train_parsed": len(train), "train_failed": train_failed, "dev_parsed": len(dev), "dev_failed": dev_failed}), flush=True)

    rng = random.Random(args.seed)
    train = rng.sample(train, args.train_size)
    dev = rng.sample(dev, args.dev_size)
    dev_path = os.path.join(args.out, "dev_sample.jsonl")
    write_jsonl(dev_path, dev)

    results = {}
    outcomes = {}
    for arm, use_plan in (("B0_no_plan", False), ("B2_gold_plan", True)):
        arm_dir = os.path.join(args.out, arm)
        arm_drive = os.path.join(args.drive_dir, arm)
        finished_path = os.path.join(arm_drive, "finished.json")
        if os.path.exists(finished_path):
            with open(finished_path) as f:
                finished = json.load(f)
            results[arm], outcomes[arm] = finished["result"], finished["outcomes"]
            print(arm, "already finished, skipping", flush=True)
            continue

        start = time.time()
        model_path = os.path.join(arm_drive, "best_model")
        train_loss = None
        if not os.path.isdir(model_path):
            train_path = os.path.join(args.out, f"train_{arm}.jsonl")
            write_jsonl(train_path, [r if use_plan else {k: v for k, v in r.items() if k != "plan"} for r in train])
            summary = run_sft_training(data_path=train_path, output_dir=arm_dir, drive_dir=arm_drive, epochs=1,
                                       batch_size=1, learning_rate=2e-4, dry_run=False, model_name_or_path=args.model,
                                       db_dir=db_dir, seed=args.seed, save_steps=args.save_steps,
                                       resume_from_checkpoint="latest")
            model_path = summary["saved_to"]
            train_loss = summary["train_loss"]
        harness = EvaluationHarness(db_dir=db_dir, num_candidates=2, model_path=model_path,
                                    output_log_dir=arm_dir, output_report_dir=arm_dir, use_record_plan=use_plan)
        res = harness.evaluate_dataset(dev_path)
        with open(os.path.join(arm_dir, "evaluation_results.jsonl"), encoding="utf-8") as f:
            logs = [json.loads(line) for line in f]
        preds = [l["predicted_sql"] for l in logs]
        golds = [l["gold_sql"] for l in logs]
        paths = [os.path.join(db_dir, f"{l['db_id']}.sqlite") for l in logs]
        outcomes[arm] = compute_execution_accuracy(preds, golds, paths)[1]
        results[arm] = {"ex": res["execution_accuracy"], "vsr": res["valid_sql_rate"], "latency": res["latency"],
                        "train_loss": train_loss, "minutes": round((time.time() - start) / 60, 1)}
        print(arm, json.dumps(results[arm]), flush=True)
        with open(finished_path, "w") as f:
            json.dump({"result": results[arm], "outcomes": outcomes[arm], "predictions": preds}, f)

    diff, low, high = bootstrap_paired_ci(outcomes["B2_gold_plan"], outcomes["B0_no_plan"])
    results["comparison"] = {"ex_diff_points": diff, "ci95": [low, high],
                             "mcnemar_p": mcnemar_exact_p(outcomes["B2_gold_plan"], outcomes["B0_no_plan"]),
                             "n": len(outcomes["B0_no_plan"])}
    for folder in (args.out, args.drive_dir):
        with open(os.path.join(folder, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
    print("COMPARISON", json.dumps(results["comparison"]), flush=True)


if __name__ == "__main__":
    main()
