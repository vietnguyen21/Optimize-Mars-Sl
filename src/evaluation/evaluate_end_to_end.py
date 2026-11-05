"""
End-to-End Evaluation Harness for Text-to-SQL SLM Architecture.
Runs the entire 4-phase pipeline on BIRD evaluation data (mini_bird.jsonl or full BIRD):
1. Schema Grounding & DDL Loading.
2. XGrammar PDA Constrained Decoding & Fallback.
3. Multi-Candidate Trajectory Generation (8-16 Rollouts).
4. PV-SQL 10 AST Validation & Safe DB Probing.
5. System-1 Laya-421M 0-Token Decision Scoring & Adaptive Early-Exit Gate (>= 0.85).
6. Metrics Computation: Execution Accuracy (EX), Valid SQL Rate (VSR), ECE, and Latency.
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
import time

# Ensure repository root is in sys.path for direct CLI execution
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from typing import Dict, Any, List, Optional, Tuple

from src.common.schemas import SQLCandidate, CompactSubSchema, ValidationResult
from src.grammar import get_grammar_engine, BaseGrammarEngine
from src.validation import (
    PVSQLValidator,
    DBProbeEngine,
    LayaScorer,
    AdaptiveEarlyExitGate,
    SecurityFilter,
    AbstentionManager,
    validate_candidate
)
from src.evaluation.candidate_generator import CandidateGenerator
from src.evaluation.metrics import (
    compute_execution_accuracy,
    compute_vsr,
    compute_ece,
    compute_latency_stats,
    measure_memory_footprint,
    format_benchmark_summary,
    execute_query_safely,
    compare_sql_results
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EvaluateEndToEnd")


def load_schema_ddl(db_path: str) -> str:
    """Full CREATE TABLE DDL of a SQLite database.

    shortcut: passes the whole schema, not Agent 1's Steiner sub-schema S'; switch to
    src.grounding output once the retriever is trained.
    """
    conn = sqlite3.connect(f"file:{os.path.abspath(db_path)}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL").fetchall()
    finally:
        conn.close()
    return "\n\n".join(r[0].strip() + ";" for r in rows)


class EvaluationHarness:
    """
    Orchestrates the entire end-to-end evaluation pipeline over a test dataset.
    """

    def __init__(
        self,
        db_dir: str = "data/mini",
        num_candidates: int = 8,
        temperature: float = 0.7,
        grammar_engine_type: str = "auto",
        early_exit_threshold: float = 0.85,
        decision_threshold: float = 0.75,
        output_log_dir: str = "logs",
        output_report_dir: str = "reports",
        model_path: Optional[str] = None,
        scorer_model_path: Optional[str] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None
    ):
        self.db_dir = db_dir
        self.num_candidates = num_candidates
        self.temperature = temperature
        self.output_log_dir = output_log_dir
        self.output_report_dir = output_report_dir
        self.model_path = model_path

        os.makedirs(self.output_log_dir, exist_ok=True)
        os.makedirs(self.output_report_dir, exist_ok=True)

        # Initialize sub-modules
        self.grammar_engine = get_grammar_engine(grammar_engine_type)

        # A load failure must stop the run: there is no fallback generator.
        if model is None and model_path:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            logger.info(f"Loading model checkpoint from {model_path}...")
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, device_map="auto")
            model.eval()

        self.candidate_generator = CandidateGenerator(
            model=model,
            tokenizer=tokenizer,
            default_num_candidates=num_candidates,
            default_temperature=temperature
        )
        self.pv_validator = PVSQLValidator()
        self.db_probe = DBProbeEngine(max_session_queries=5, row_limit=5)
        self.scorer = LayaScorer(
            model_path=scorer_model_path,
            threshold=decision_threshold,
            early_exit_threshold=early_exit_threshold
        )
        self.security_filter = SecurityFilter()
        self.abstention_manager = AbstentionManager(min_confidence=0.50)

    def evaluate_dataset(
        self,
        dataset_path: str,
        limit_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end evaluation over JSONL test records.
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")

        records = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    records.append(json.loads(line_str))

        if limit_samples:
            records = records[:limit_samples]

        logger.info(f"Loaded {len(records)} evaluation samples from {dataset_path}")

        predictions = []
        gold_queries = []
        db_paths = []
        confidences = []
        latencies_ms = []
        detailed_logs = []

        total_early_exits = 0
        total_abstentions = 0

        for idx, item in enumerate(records):
            q_start = time.perf_counter()
            question = item.get("question", "")
            gold_sql = item.get("SQL", "")
            db_id = item.get("db_id", "")
            db_file = os.path.join(self.db_dir, f"{db_id}.sqlite")

            # 1. Candidate rollouts (8 to 16 rollouts). Gold SQL is used only for scoring below.
            candidates = self.candidate_generator.generate(
                prompt_or_input=question,
                num_candidates=self.num_candidates,
                temperature=self.temperature,
                evidence=item.get("evidence", ""),
                schema_ddl=load_schema_ddl(db_file)
            )

            # 2. Candidate Selection via Grammar PDA + PV-SQL + DB Probe + System-1 Gate
            best_candidate = None
            best_score = -1.0
            best_val_result = None
            early_exited = False
            valid_candidates_count = 0
            has_security_violation = False
            has_hallucinations = False

            # Reset probe session quota for this query
            self.db_probe.reset_counter()

            for cand in candidates:
                # Grammar prefix / syntax check
                g_ok, _ = self.grammar_engine.validate_or_mask([], cand.sql)
                if not g_ok:
                    continue

                # Full candidate validation
                val_res = validate_candidate(
                    candidate=cand,
                    question=question,
                    db_path=db_file,
                    validator=self.pv_validator,
                    probe_engine=self.db_probe,
                    scorer=self.scorer,
                    security_filter=self.security_filter
                )

                if not val_res.security_safe:
                    has_security_violation = True

                if val_res.final_verdict == "ACCEPT":
                    valid_candidates_count += 1

                # Check Adaptive Early-Exit Gate (confidence >= 0.85)
                if val_res.final_verdict == "ACCEPT" and self.scorer.should_early_exit(val_res.system1_score):
                    best_candidate = cand
                    best_score = val_res.system1_score
                    best_val_result = val_res
                    early_exited = True
                    total_early_exits += 1
                    break

                # Otherwise, maintain best scoring valid candidate
                if val_res.final_verdict == "ACCEPT" and val_res.system1_score > best_score:
                    best_candidate = cand
                    best_score = val_res.system1_score
                    best_val_result = val_res

            # Check abstention
            current_conf = best_score if best_candidate is not None else 0.0
            should_ab, ab_reason = self.abstention_manager.should_abstain(
                valid_candidates_count=valid_candidates_count,
                best_confidence=current_conf,
                has_security_violation=has_security_violation,
                has_hallucinations=has_hallucinations
            )

            is_abstained = False
            if should_ab:
                is_abstained = True
                total_abstentions += 1

            # Fallback if no candidate achieved ACCEPT
            if best_candidate is None:
                best_candidate = candidates[0]
                best_score = self.scorer.score_candidate(question, "", best_candidate.sql)

            elapsed_ms = (time.perf_counter() - q_start) * 1000.0
            latencies_ms.append(elapsed_ms)

            selected_sql = best_candidate.sql
            predictions.append(selected_sql)
            gold_queries.append(gold_sql)
            db_paths.append(db_file)
            confidences.append(best_score)

            detailed_logs.append({
                "sample_idx": idx,
                "db_id": db_id,
                "question": question,
                "gold_sql": gold_sql,
                "predicted_sql": selected_sql,
                "confidence": best_score,
                "early_exited": early_exited,
                "abstained": is_abstained,
                "abstention_reason": ab_reason if is_abstained else None,
                "latency_ms": elapsed_ms
            })

        # 3. Compute benchmark metrics
        ex, matches = compute_execution_accuracy(predictions, gold_queries, db_paths)
        vsr, valids = compute_vsr(predictions, db_paths)
        ece = compute_ece(confidences, matches, num_bins=10)
        latency_stats = compute_latency_stats(latencies_ms)
        mem_stats = measure_memory_footprint()

        benchmark_results = {
            "dataset": os.path.basename(dataset_path),
            "total_queries": len(records),
            "scorer_type": self.scorer.scorer_type,
            "execution_accuracy": ex,
            "valid_sql_rate": vsr,
            "ece": ece,
            "latency": latency_stats,
            "memory": mem_stats,
            "early_exit_count": total_early_exits,
            "early_exit_rate_pct": round((total_early_exits / max(1, len(records))) * 100.0, 2),
            "total_abstentions": total_abstentions,
            "abstention_rate_pct": round((total_abstentions / max(1, len(records))) * 100.0, 2)
        }

        # 4. Save detailed results log
        log_file = os.path.join(self.output_log_dir, "evaluation_results.jsonl")
        with open(log_file, "w", encoding="utf-8") as f:
            for log_entry in detailed_logs:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        logger.info(f"Detailed evaluation log written to {log_file}")

        # 5. Save benchmark summary report
        summary_md = format_benchmark_summary(benchmark_results, title="Text-to-SQL SLM Phase 4 Benchmark")
        report_file = os.path.join(self.output_report_dir, "benchmark_summary.md")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(summary_md)
        logger.info(f"Benchmark summary report saved to {report_file}")

        return benchmark_results


def evaluate_end_to_end(
    dataset_path: str = "data/mini/mini_bird.jsonl",
    db_dir: str = "data/mini",
    num_candidates: int = 8,
    model_path: Optional[str] = None,
    scorer_model_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience functional API for end-to-end evaluation.
    """
    harness = EvaluationHarness(db_dir=db_dir, num_candidates=num_candidates, model_path=model_path,
                                scorer_model_path=scorer_model_path)
    return harness.evaluate_dataset(dataset_path)


def main():
    parser = argparse.ArgumentParser(description="End-to-End Evaluation Harness for Text-to-SQL SLM")
    parser.add_argument("--dataset", type=str, default="data/mini/mini_bird.jsonl", help="Path to evaluation JSONL")
    parser.add_argument("--db_dir", type=str, default="data/mini", help="Directory containing target SQLite DBs")
    parser.add_argument("--num_candidates", type=int, default=8, help="Number of candidate rollouts (8-16)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries to evaluate")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned generator checkpoint")
    parser.add_argument("--scorer_model_path", type=str, default=None,
                        help="Path to trained System-1 scorer; without it the heuristic baseline scorer is used and reported as such")
    args = parser.parse_args()

    harness = EvaluationHarness(
        db_dir=args.db_dir,
        num_candidates=args.num_candidates,
        model_path=args.model_path,
        scorer_model_path=args.scorer_model_path
    )
    results = harness.evaluate_dataset(args.dataset, limit_samples=args.limit)
    print("\n" + format_benchmark_summary(results))


if __name__ == "__main__":
    main()
