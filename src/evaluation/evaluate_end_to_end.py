"""Independent checkpoint evaluation; fixture runs are explicitly non-benchmarks."""
import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.common.schemas import GenerationInput
from src.evaluation.candidate_generator import CandidateGenerator
from src.evaluation.metrics import (compute_execution_accuracy, compute_vsr, compute_ece,
    compute_latency_stats, measure_memory_footprint, format_benchmark_summary)
from src.grammar import get_grammar_engine
from src.grounding import (ColumnCardExtractor, SchemaGraphBuilder,
    NodeWeightedSteinerTree, LitESQLEmbeddingRetriever)
from src.validation import PVSQLValidator, DBProbeEngine, LayaScorer, SecurityFilter, validate_candidate
from src.training.runtime import load_records, resolve_db, dataset_fingerprint, seed_everything

logger = logging.getLogger(__name__)


class EvaluationHarness:
    """Gold labels enter only the metric pass, after all predictions are fixed.

    ``dry_run=True`` exercises plumbing with independent SELECT 1 fixtures. It
    never claims model quality. ``constraint_mode='xgrammar'`` requires the real
    xgrammar extension; ``posthoc`` is an explicitly recorded validation ablation.
    """
    def __init__(self, db_dir='data/mini', num_candidates=8, temperature=0.7,
                 grammar_engine_type='auto', early_exit_threshold=0.85,
                 decision_threshold=0.75, output_log_dir='logs',
                 output_report_dir='reports', model_path=None,
                 decision_model_path=None, grounding_model_path=None,
                 dry_run=False, device='cuda:0', scorer_device='cpu',
                 grounding_device='cpu', retrieval_top_k=25,
                 max_new_tokens=256, max_prompt_length=2048,
                 constraint_mode='posthoc', seed=42, training_data_paths=None):
        if not dry_run and (not model_path or not decision_model_path):
            raise ValueError('Real evaluation requires model_path and decision_model_path; smoke runs require dry_run=True')
        if num_candidates < 1:
            raise ValueError('num_candidates must be positive')
        self.db_dir, self.num_candidates, self.temperature = db_dir, num_candidates, temperature
        self.output_log_dir, self.output_report_dir = output_log_dir, output_report_dir
        self.model_path, self.decision_model_path = model_path, decision_model_path
        self.grounding_model_path = grounding_model_path
        self.dry_run, self.seed, self.device = dry_run, seed, device
        self.constraint_mode, self.retrieval_top_k = constraint_mode, retrieval_top_k
        self.training_data_paths = training_data_paths or []
        self._schema_cache = {}
        self._dense_model = None
        if grounding_model_path:
            from sentence_transformers import SentenceTransformer
            self._dense_model = SentenceTransformer(grounding_model_path, device=grounding_device)
        model, tokenizer = None, None
        if not dry_run:
            import sqlglot  # Real evaluation requires actual AST parsing, never regex-only verification.
            from src.training.runtime import load_causal_model
            model, tokenizer = load_causal_model(model_path, training=False, device=device)
        if constraint_mode == 'xgrammar' and not dry_run:
            # Missing optional dependency must fail before a long benchmark starts.
            import xgrammar  # noqa: F401
        self.candidate_generator = CandidateGenerator(model=model, tokenizer=tokenizer,
            default_num_candidates=num_candidates, default_temperature=temperature,
            dry_run=dry_run, max_new_tokens=max_new_tokens,
            max_prompt_length=max_prompt_length, constraint_mode=constraint_mode)
        self.grammar_engine = get_grammar_engine(grammar_engine_type)
        self.pv_validator = PVSQLValidator()
        self.db_probe = DBProbeEngine(max_session_queries=5, row_limit=5)
        self.scorer = LayaScorer(model_path=decision_model_path, threshold=decision_threshold,
            early_exit_threshold=early_exit_threshold, device=scorer_device, dry_run=dry_run)
        self.security_filter = SecurityFilter()

    def _ground(self, question, db_id, db_file, evidence=''):
        if db_id not in self._schema_cache:
            cards = ColumnCardExtractor().extract_from_sqlite(db_file)
            graph = SchemaGraphBuilder.build_from_sqlite(db_file, db_id=db_id)
            retriever = LitESQLEmbeddingRetriever(cards=cards, top_k=self.retrieval_top_k)
            if self._dense_model is not None:
                retriever.model = self._dense_model
                retriever.model_loaded = True
                from src.training.train_embedding import format_column_card_text
                retriever.card_vectors = self._dense_model.encode(
                    [format_column_card_text(card.model_dump()) for card in cards],
                    convert_to_numpy=True, normalize_embeddings=True)
            self._schema_cache[db_id] = (graph, retriever)
        graph, retriever = self._schema_cache[db_id]
        _, columns, _ = retriever.retrieve(question=question, evidence=evidence)
        return NodeWeightedSteinerTree(graph).build_compact_subschema(db_id, columns, db_file)

    def predict(self, question, db_id, evidence=''):
        """Public inference contract deliberately cannot accept a gold SQL argument."""
        db_file = resolve_db(self.db_dir, db_id)
        sub_schema = self._ground(question, db_id, db_file, evidence)
        generation_input = GenerationInput(question=question, evidence=evidence,
            sub_schema=sub_schema, num_trajectories=self.num_candidates,
            temperature=self.temperature)
        candidates = self.candidate_generator.generate(generation_input,
            num_candidates=self.num_candidates, temperature=self.temperature)
        self.db_probe.reset_counter()
        best, best_score, early_exited = None, 0.0, False
        validation_logs = []
        for candidate in candidates:
            grammar_ok, grammar_error = self.grammar_engine.validate_sql_syntax(candidate.sql)
            if not grammar_ok:
                validation_logs.append({'candidate_id': candidate.candidate_id,
                    'sql': candidate.sql, 'grammar_error': grammar_error})
                continue
            result = validate_candidate(candidate=candidate, question=question,
                schema_ddl=sub_schema.ddl_statements, db_path=db_file,
                validator=self.pv_validator, probe_engine=self.db_probe,
                scorer=self.scorer, security_filter=self.security_filter)
            validation_logs.append(dict(result.model_dump(), sql=candidate.sql))
            if result.final_verdict == 'ACCEPT' and result.system1_score > best_score:
                best, best_score = candidate, result.system1_score
                if self.scorer.should_early_exit(best_score):
                    early_exited = True
                    break
        # An abstention remains an abstention; never substitute a rejected query.
        return {'predicted_sql': best.sql if best is not None else '',
                'confidence': best_score, 'abstained': best is None,
                'abstention_reason': 'No candidate passed all gates' if best is None else None,
                'early_exited': early_exited, 'candidates': validation_logs,
                'grounded_tables': sub_schema.selected_tables,
                'bridge_tables': sub_schema.bridge_tables_added}

    def evaluate_dataset(self, dataset_path, limit_samples=None):
        records = load_records(dataset_path)
        if limit_samples is not None:
            records = records[:limit_samples]
        if not records:
            raise ValueError('Evaluation set is empty')
        # Optional explicit train manifests reject overlapping natural-language examples.
        eval_keys = {(row['db_id'], row['question']) for row in records}
        for path in self.training_data_paths:
            overlap = eval_keys & {(row['db_id'], row['question']) for row in load_records(path)}
            if overlap:
                raise ValueError(f'Evaluation/train overlap: {len(overlap)} examples in {path}')
        if self.dry_run:
            import random
            random.seed(self.seed)
        else:
            seed_everything(self.seed)
        try:
            import torch
            if torch.cuda.is_available():
                for index in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(index)
        except ImportError:
            torch = None
        logs, predictions, confidences, latencies = [], [], [], []
        for index, record in enumerate(records):
            if torch is not None and torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            # Explicit allowlist: neither SQL nor labels can reach inference.
            result = self.predict(record['question'], record['db_id'], record.get('evidence', ''))
            if torch is not None and torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) * 1000
            predictions.append(result['predicted_sql'])
            confidences.append(result['confidence'])
            latencies.append(elapsed)
            logs.append(dict(result, sample_idx=index, db_id=record['db_id'],
                question=record['question'], latency_ms=elapsed))
        # Metrics are the first and only consumers of held-out gold SQL.
        golds = [record.get('SQL', record.get('sql', '')) for record in records]
        if any(not gold for gold in golds):
            raise ValueError('Held-out evaluation records must include SQL/sql labels for metrics')
        db_paths = [resolve_db(self.db_dir, record['db_id']) for record in records]
        ex, matches = compute_execution_accuracy(predictions, golds, db_paths)
        vsr, _ = compute_vsr(predictions, db_paths)
        for log, match in zip(logs, matches):
            log['execution_correct'] = match
        abstentions = sum(log['abstained'] for log in logs)
        early_exits = sum(log['early_exited'] for log in logs)
        metrics = {'dataset': Path(dataset_path).name,
            'dataset_sha256': dataset_fingerprint(dataset_path),
            'mode': 'smoke' if self.dry_run else 'evaluation',
            'is_benchmark': not self.dry_run,
            'checkpoint': self.model_path, 'decision_checkpoint': self.decision_model_path,
            'grounding_checkpoint': self.grounding_model_path,
            'grounding_mode': 'trained_dense' if self._dense_model is not None else 'lexical_ablation',
            'constraint_mode': self.constraint_mode,
            'grammar_enforced_during_generation': not self.dry_run and self.constraint_mode == 'xgrammar',
            'grammar_scope': 'SQLite SELECT subset; AST and execution checks still required',
            'seed': self.seed, 'training_overlap_checked': bool(self.training_data_paths),
            'total_queries': len(records), 'execution_accuracy': ex, 'valid_sql_rate': vsr,
            'ece': compute_ece(confidences, matches),
            'accepted_ece': compute_ece([c for c, log in zip(confidences, logs) if not log['abstained']],
                                       [m for m, log in zip(matches, logs) if not log['abstained']]) if abstentions < len(logs) else None,
            'latency': compute_latency_stats(latencies), 'memory': measure_memory_footprint(),
            'early_exit_count': early_exits, 'early_exit_rate_pct': early_exits / len(records) * 100,
            'total_abstentions': abstentions, 'abstention_rate_pct': abstentions / len(records) * 100,
            'coverage_pct': (len(records) - abstentions) / len(records) * 100}
        Path(self.output_log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_report_dir).mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents overwriting a prior experiment's results.
        with open(Path(self.output_log_dir) / 'evaluation_results.jsonl', 'x', encoding='utf-8') as f:
            for log in logs:
                f.write(json.dumps(log, ensure_ascii=False) + '\n')
        with open(Path(self.output_report_dir) / 'benchmark_summary.md', 'x', encoding='utf-8') as f:
            f.write(format_benchmark_summary(metrics))
        with open(Path(self.output_report_dir) / 'metrics.json', 'x', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        return metrics


def evaluate_end_to_end(dataset_path='data/mini/mini_bird.jsonl', db_dir='data/mini',
                        num_candidates=8, model_path=None, **kwargs):
    return EvaluationHarness(db_dir=db_dir, num_candidates=num_candidates,
        model_path=model_path, **kwargs).evaluate_dataset(dataset_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='data/mini/mini_bird.jsonl')
    parser.add_argument('--db_dir', default='data/mini')
    parser.add_argument('--model_path')
    parser.add_argument('--decision_model_path')
    parser.add_argument('--grounding_model_path')
    parser.add_argument('--num_candidates', type=int, default=8)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--training_data_paths', nargs='*', default=[])
    parser.add_argument('--scorer_device', default='cpu')
    parser.add_argument('--grounding_device', default='cpu')
    parser.add_argument('--retrieval_top_k', type=int, default=25)
    parser.add_argument('--max_prompt_length', type=int, default=2048)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--constraint_mode', choices=['xgrammar', 'posthoc'], default='posthoc')
    args = parser.parse_args()
    options = vars(args)
    dataset, limit, output = options.pop('dataset'), options.pop('limit'), options.pop('output_dir')
    harness = EvaluationHarness(**options, output_log_dir=str(Path(output) / 'logs'),
        output_report_dir=str(Path(output) / 'reports'))
    print(json.dumps(harness.evaluate_dataset(dataset, limit), indent=2))


if __name__ == '__main__':
    main()
