"""Run one generator-only baseline (B0-B5) on cal-A 300. Needs a GPU runtime and the BIRD databases.

B0  python scripts/eval_generator_only.py --run-dir <Drive>/runs/m5-eval-b0-seed42 --baseline b0 \\
        --checkpoint Qwen/Qwen2.5-Coder-3B-Instruct --schema-mode full --data-dir <Drive>/data
B2  ... --baseline b2 --checkpoint <SFT100> --schema-mode sprime --sprime-file <s_prime_cal_a.jsonl> --prompt-format v1

Results go to <run-dir>/cal_a300/{full,sprime}/ (QUC U4). Rerunning the same command resumes.
Provenance (commit, dirty, code_hash) is written to launch-<stamp>.json like run_stage; a number from a
dirty tree or from work/ code is a check number, not a main number (QUC s2.4 item 4).
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.colab.run_stage import (DEFAULT_GENERATOR, LOCAL_DB_DIR, code_provenance,  # noqa: E402
                                     ensure_databases)
from src.evaluation.generator_only import (evaluate_generator_only, full_schema,  # noqa: E402
                                           load_sprime, make_hf_generate_fn, make_sprime_schema_fn,
                                           verified_cal_a_300)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="runs/<run_id> assigned by the Leader (U27)")
    parser.add_argument("--baseline", required=True, choices=["b0", "b1", "b2", "b3", "b4", "b5"])
    parser.add_argument("--checkpoint", default=DEFAULT_GENERATOR, help="base model id or adapter folder")
    parser.add_argument("--schema-mode", required=True, choices=["full", "sprime"])
    parser.add_argument("--sprime-file", help="s_prime_cal_a.jsonl from S1 (required for sprime)")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--db-dir", default=LOCAL_DB_DIR)
    parser.add_argument("--prompt-format", choices=["v1", "v2"], default="v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, help="first N of the 300 (smoke check only; not a result)")
    args = parser.parse_args()
    if args.schema_mode == "sprime" and not args.sprime_file:
        raise SystemExit("--sprime-file is required for --schema-mode sprime")

    from src.training.runtime import check_prompt_format, load_causal_model, seed_everything
    check_prompt_format(args.checkpoint, args.prompt_format)  # refuse an adapter trained on another prompt
    seed_everything(args.seed)

    run_dir = Path(args.run_dir)
    # A --limit smoke check never shares a folder with the real run (its meta would block resuming it).
    out_dir = run_dir / ("smoke_limit" if args.limit else "cal_a300") / args.schema_mode
    provenance = code_provenance()
    partial = out_dir / "generations.partial.jsonl"
    launches = sorted(run_dir.glob("launch-*.json"))
    if partial.exists() and launches:
        previous = json.loads(launches[-1].read_text()).get("code_hash")
        if previous != provenance["code_hash"]:
            raise SystemExit(f"Refusing to resume {run_dir}: code_hash {previous} != {provenance['code_hash']}. "
                             "Restore that code or use a new run id.")
    run_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (run_dir / f"launch-{stamp}.json").write_text(json.dumps(
        {"stage": "generator_only", "argv": sys.argv[1:], **provenance, "started": stamp}, indent=2))

    if Path(args.db_dir).resolve() == Path(LOCAL_DB_DIR).resolve():  # rebuild /content/bird_db from BIRD zips on Drive
        ensure_databases(str(Path(args.data_dir).resolve().parent))
    data_path = Path(args.data_dir) / "retrieval_dev.jsonl"
    indices, data_sha = verified_cal_a_300(data_path)  # stops if the sample or file differs from QUC s3.3
    if args.schema_mode == "sprime":
        schema_fn = make_sprime_schema_fn(load_sprime(args.sprime_file), args.db_dir)
    else:
        schema_fn = lambda row, row_id: full_schema(row, args.db_dir)  # noqa: E731
    model, tokenizer = load_causal_model(args.checkpoint, training=False, device=args.device)
    generate_fn = make_hf_generate_fn(model, tokenizer, args.prompt_format,
                                      args.max_prompt_length, args.max_new_tokens)
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    meta = {"baseline": args.baseline, "split": "cal_a300", "checkpoint": args.checkpoint,
            "commit": provenance["commit"], "seed": args.seed, "prompt_format": args.prompt_format, "gpu": gpu,
            "schema_mode": args.schema_mode, "max_prompt_length": args.max_prompt_length,
            "max_new_tokens": args.max_new_tokens, "decoding": "greedy",
            "retrieval_dev_sha256": data_sha, "sprime_file": args.sprime_file}
    metrics = evaluate_generator_only(str(data_path), indices, args.db_dir, generate_fn, schema_fn,
                                      str(out_dir), meta, limit=args.limit)
    print(json.dumps(metrics, indent=2))
    if provenance["dirty"] or args.limit:
        print("NOTE: dirty tree or --limit: this is a check number, not a main number.")


if __name__ == "__main__":
    main()
