## 2026-11-02T06:39:33Z
You are worker_m4, a teamwork_worker subagent fulfilling the role of [AGENT-VAL-BENCH] (Kỹ sư System-1 Validation & Benchmark) with [AGENT-GEN-G2] for Milestone M4 (Phase 4: Validation, Grammar PDA & Benchmark).

Your working directory is: /mnt/c/Workspace/optimize/.agents/teamwork/worker_m4/
You must first read the original user request at:
/mnt/c/Workspace/optimize/.agents/teamwork/ORIGINAL_REQUEST.md
Also read PROJECT.md and TEST_INFRA.md at:
/mnt/c/Workspace/optimize/PROJECT.md
/mnt/c/Workspace/optimize/TEST_INFRA.md
And reference previous milestone handoffs at:
/mnt/c/Workspace/optimize/.agents/teamwork/worker_m1/handoff.md
/mnt/c/Workspace/optimize/.agents/teamwork/worker_m2/handoff.md
/mnt/c/Workspace/optimize/.agents/teamwork/worker_m3_remediate/handoff.md

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Your exclusive file ownership:
- src/grammar/__init__.py
- src/grammar/base.py
- src/grammar/xgrammar_engine.py
- src/grammar/fallback_engine.py
- src/validation/__init__.py
- src/validation/pv_sql_rules.py
- src/validation/db_probe.py
- src/validation/laya_scorer.py
- src/validation/security_filter.py
- src/evaluation/__init__.py
- src/evaluation/candidate_generator.py
- src/evaluation/evaluate_end_to_end.py
- src/evaluation/metrics.py
- tests/test_grammar_and_validation.py
- tests/test_evaluation.py

Scope and Requirements for Milestone M4 (Phase 4):
1. `src/grammar/`:
   - `base.py`: `BaseGrammarEngine` defining abstract methods (`compile_grammar(schema_ddl)`, `filter_logits(logits, context)`, `validate_sql_syntax(sql)`).
   - `xgrammar_engine.py`: `XGrammarPDAEngine` implementing Pushdown Automaton bitmask constraints using `xgrammar` when available, with compile cache (4-8 ms compilation) and dynamic BNF/EBNF for SQLite dialect.
   - `fallback_engine.py`: `RegexASTFallbackEngine` zero-crash fallback parsing tokens via regex and `sqlglot` when xgrammar binary/GPU is absent, ensuring 100% crash-free operation across all environments.
   - `__init__.py`: Factory `get_grammar_engine(engine_type="auto")` that automatically selects `XGrammarPDAEngine` if available or gracefully falls back to `RegexASTFallbackEngine`.
2. `src/validation/`:
   - `pv_sql_rules.py`: PV-SQL 10 deterministic AST rules using `sqlglot` (blocking invalid joins, missing join conditions, schema hallucinations, unindexed cartesian products, illegal aggregations, invalid window functions, type mismatches, etc.).
   - `db_probe.py`: Safe execution probing engine executing maximum 5 queries with `LIMIT` on target SQLite database (0 GB VRAM footprint, read-only mode).
   - `laya_scorer.py`: Laya-421M (ModernBERT) System-1 0-token parallel decision scorer (~1.2 ms latency, calibrated confidence score, Adaptive Early-Exit Gate at >= 0.85 confidence).
   - `security_filter.py`: Anti-hallucination Abstention logic and security filter against SQL prompt injection.
   - `__init__.py`.
3. `src/evaluation/`:
   - `candidate_generator.py`: Generates 8–16 candidate trajectories per prompt with configurable sampling (K in [8, 16], temperature=0.7, top_p=0.9, top_k=50) for Ablation Studies.
   - `evaluate_end_to_end.py`: End-to-end evaluation harness running the entire pipeline on `data/mini/mini_bird.jsonl` (or full BIRD).
   - `metrics.py`: Computes Execution Accuracy (EX), Valid SQL Rate (VSR), ECE, Latency, and Memory Footprint, outputting benchmark summary table.
   - `__init__.py`.
4. Tests:
   - `tests/test_grammar_and_validation.py`: unit tests for grammar engines (both XGrammar and Fallback), PV-SQL 10 AST rules, DB probing, Laya scorer, security filter.
   - `tests/test_evaluation.py`: unit tests for 8-16 candidate generation, metrics calculation (EX, VSR), and end-to-end evaluation runner.
5. Verification:
   - Run `python3 -m unittest discover tests/ -v`. Confirm all tests pass with 0 regressions.
   - Run standalone execution of `src/evaluation/evaluate_end_to_end.py` on `data/mini/` and confirm execution results.
6. Write your completion report in `/mnt/c/Workspace/optimize/.agents/teamwork/worker_m4/handoff.md` and send a message to orchestrator when done.
