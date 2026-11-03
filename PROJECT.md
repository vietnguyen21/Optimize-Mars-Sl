# Project: Text-to-SQL SLM Research Architecture (Director-Led SSoT)
## puppyvn :>
##
## Architecture
- **Architecture Paradigm**: Director-Led Single Source of Truth (SSoT) Multi-Agent Architecture (`AGENTS_RESEARCH.md`).
- **Core Problem Solved**: Replaces MARS-SQL 21B (41-47 GB VRAM, 22s latency, structural blindness, hallucinations, uncalibrated decisions) with an optimized Small Language Model (SLM) pipeline (2.5B–4.0B, <6s latency, 3.5–8.5 GB VRAM).
- **Subsystem Breakdown**:
  1. `src/grounding/` (LitE-SQL 0.6B + PK-FK Graph + Node-Weighted Steiner Tree): Schema linking and bridge table preservation (100% recovery across 48 complex BIRD queries).
  2. `src/training/` (FINER-SQL-3B / Qwen2.5-Coder-7B): Completion-Only Loss ($-100$ masking), LLM-JEPA causal mask alignment via $[PRED]$ token, Dense GRPO (4 reward heads: $R_{\text{format}}, R_{\text{exec}}, R_{\text{mem}}, R_{\text{atomic}}$), Google Drive checkpoint synchronization (`DriveCheckpointCallback`).
  3. `notebooks/` (Google Colab 3B & 7B Training & Eval): 5 production notebooks with hardware tiering (T4, L4, A100), automated Drive mounting, and checkpoint resume.
  4. `src/grammar/` & `src/validation/` & `src/evaluation/`:
     - Grammar: `BaseGrammarEngine` with `XGrammarPDAEngine` (pushdown automaton bitmask) and `RegexASTFallbackEngine` (100% crash-free).
     - Validation: PV-SQL 10 deterministic AST rules + lightweight DB probing (max 5 queries with LIMIT, 0 VRAM) + Laya-421M ModernBERT System-1 0-token scoring ($ECE \le 0.081$, ~1.2 ms) + Abstention + Security prompt injection filtering.
     - Evaluation: Candidate generator sampling 8–16 rollouts per question, end-to-end benchmark suite calculating Execution Accuracy (EX), Valid SQL Rate (VSR), and latency.
  5. `data/mini/` & `tests/` & `scripts/`: Self-contained smoke test suite running end-to-end in <60 seconds without heavy GPU/model downloads.

## Feature Inventory
Every feature from the survey phase mapped to its assigned milestone and source:
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | LitE-SQL Column Cards & Indexing | 6-field column card schema embedding and indexing | M1 | survey_1 & survey_2 |
| 2 | PK-FK Graph Construction | Directed/undirected schema graph with foreign key relationships | M1 | survey_1 & survey_2 |
| 3 | Node-Weighted Steiner Tree (NWST) | Minimal connecting subgraph preserving bridge tables (100% BIRD-dev) | M1 | survey_1 & survey_2 |
| 4 | Sample Mini Datasets (`data/mini/`) | Realistic miniature SQLite DBs & JSONL files for offline smoke tests | M1 | survey_3 & R2 |
| 5 | Completion-Only Cross-Entropy Loss | Custom PyTorch/TRL trainer masking prompt tokens with -100 | M2 | survey_1 & survey_2 |
| 6 | LLM-JEPA Alignment Module | 2-block causal mask, $[PRED]$ token cosine distance loss, random dropout | M2 | survey_1 & survey_2 |
| 7 | Dense GRPO 4-Head Reward Loss | Fine-grained rewards ($R_{\text{format}}, R_{\text{exec}}, R_{\text{mem}}, R_{\text{atomic}}$) preventing advantage collapse | M2 | survey_1 & survey_2 |
| 8 | DriveCheckpointCallback & Resume | Automated Google Drive checkpoint sync and `--resume_from_checkpoint` | M2 | survey_1 & survey_2 |
| 9 | Embedding & Decision Model Training | Scripts for training LitE-SQL HN-SupCon and Laya-421M RLCD | M2 | survey_1 & survey_2 |
| 10 | Colab Notebook: Data Prep & Embed | Notebook 01 for preparing schema graphs, embeddings, and cards | M3 | survey_2 & survey_3 |
| 11 | Colab Notebook: SFT & JEPA 3B/7B | Notebook 02 for SFT + LLM-JEPA with Unsloth / HuggingFace & Drive mount | M3 | survey_2 & survey_3 |
| 12 | Colab Notebook: GRPO RL 3B/7B | Notebook 03 for Group Relative Policy Optimization on T4/L4/A100 | M3 | survey_2 & survey_3 |
| 13 | Colab Notebook: System-1 Laya-421M | Notebook 04 for training ModernBERT RLCD scorer | M3 | survey_2 & survey_3 |
| 14 | Colab Notebook: End-to-End Eval | Notebook 05 running evaluation pipeline and reporting metrics | M3 | survey_2 & survey_3 |
| 15 | GrammarEngine Strategy Pattern | Unified abstraction for PDA grammar-constrained decoding and fallback | M4 | survey_3 & R1 |
| 16 | XGrammar PDA Engine | Pushdown automaton bitmask generation for SQLite grammar | M4 | survey_1 & survey_2 |
| 17 | Regex AST Fallback Engine | Deterministic syntax enforcement fallback when xgrammar C++ is absent | M4 | survey_3 & R1 |
| 18 | PV-SQL 10 Deterministic AST Rules | Syntax tree verification blocking invalid joins, predicates, aggregations | M4 | survey_1 & survey_2 |
| 19 | Lightweight DB Probing Engine | Safe execution probe (max 5 queries with LIMIT, 0 VRAM footprint) | M4 | survey_1 & survey_2 |
| 20 | System-1 Laya-421M Scorer | 0-token parallel scoring via `[MASK]` token and confidence threshold | M4 | survey_1 & survey_2 |
| 21 | Abstention & Security Filter | Anti-hallucination abstention and prompt injection defense | M4 | survey_1 & survey_2 |
| 22 | Multi-Candidate Generator (8-16) | Sampling 8–16 candidate trajectories per prompt for ablation | M4 | survey_1 & survey_2 |
| 23 | End-to-End Benchmark Harness | Full evaluation pipeline computing EX, VSR, latency, and reporting | M4 | survey_1 & survey_2 |
| 24 | Automated Smoke Test Suite | Independent pytest/unittest suite running on `data/mini/` (<60s) | M5 | survey_3 & R2 |
| 25 | Technical Handover Document | Comprehensive `HANDOVER.md` at project root covering roles and usage | M6 | survey_3 & R3 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Phase 1: Data Pipeline & Schema Grounding | `src/grounding/` (column card, graph builder, steiner tree, retriever) and `data/mini/` | None | DONE |
| M2 | Phase 2: SLM Training Scripts | `src/training/` (Completion-Only SFT, LLM-JEPA, Dense GRPO, callbacks) | M1 | DONE |
| M3 | Phase 3: Colab Training Notebooks | `notebooks/` (01 to 05, Colab T4/L4/A100, Drive sync, 3B & 7B configs) | M2 | DONE |
| M4 | Phase 4: Validation, Grammar PDA & Benchmark | `src/grammar/`, `src/validation/`, `src/evaluation/` (XGrammar PDA, Fallback, PV-SQL 10 AST, 8-16 candidates, Laya-421M 0-token, metrics) | M1, M2 | DONE |
| M5 | Automated Smoke Testing & Verification | `tests/` and `scripts/run_smoke_tests.py` verifying all phases on `data/mini/` | M1, M2, M3, M4 | DONE |
| M6 | Handover Documentation | `HANDOVER.md` at root covering 4 agent roles, file inventory, reproduction guide | M1, M2, M3, M4, M5 | DONE |

## Interface Contracts
### `src/grounding` ↔ `src/training` & `src/evaluation`
- **Output of Grounding / Input to Generator**: `CompactSubSchema`
  ```python
  class ColumnCard(BaseModel):
      table_name: str
      column_name: str
      data_type: str
      constraints: List[str]
      column_description: str
      sample_values: List[str]

  class CompactSubSchema(BaseModel):
      db_id: str
      selected_tables: List[str]
      selected_columns: Dict[str, List[str]]
      foreign_keys: List[Tuple[str, str, str, str]] # (t1, c1, t2, c2)
      ddl_statements: str
  ```

### `src/training` ↔ `src/evaluation`
- **Model Checkpoints**: HuggingFace / PEFT LoRA adapter checkpoint with config.json, adapter_model.bin/safetensors, and tokenizer.
- **Prompt Format**:
  ```text
  <system>You are an expert SQL analyst...</system>
  <schema>{ddl_statements}</schema>
  <question>{user_question}</question>
  <think>
  {reasoning_steps}
  </think>
  <answer>
  {sql_query}
  </answer>
  ```

### `src/evaluation` ↔ `src/validation`
- **Candidate Trajectories**:
  ```python
  class SQLCandidate(BaseModel):
      candidate_id: int
      sql: str
      reasoning: str
      logprob: float

  class ValidationResult(BaseModel):
      candidate_id: int
      is_valid_syntax: bool
      pv_sql_passed: bool
      probe_passed: bool
      system1_score: float
      security_safe: bool
      final_verdict: str # ACCEPT, REJECT, ABSTAIN
  ```

## Code Layout
```
/mnt/c/Workspace/optimize/
├── AGENTS_RESEARCH.md
├── GLOSSARY.md
├── HANDOVER.md                          # (M6)
├── PROJECT.md                           # Master orchestration specification
├── TEST_INFRA.md                        # E2E Test infrastructure specification
├── requirements.txt
├── data/
│   └── mini/                           # (M1) Miniature SQLite DBs and JSONL samples
│       ├── financial_mini.sqlite
│       ├── california_schools.sqlite
│       ├── mini_bird.jsonl
│       └── mini_synsql.jsonl
├── docs/
│   └── specifications/
├── notebooks/                          # (M3) Google Colab notebooks
│   ├── 01_data_preparation_and_embedding.ipynb
│   ├── 02_sft_and_jepa_training_3b.ipynb
│   ├── 03_grpo_training_3b_and_7b.ipynb
│   ├── 04_system1_laya421m_training.ipynb
│   └── 05_end_to_end_evaluation.ipynb
├── scripts/
│   ├── make_mini_dataset.py            # (M1)
│   └── run_smoke_tests.py              # (M5) Standalone test runner
├── src/
│   ├── __init__.py
│   ├── common/                         # Shared schemas and utilities
│   │   ├── __init__.py
│   │   └── schemas.py
│   ├── evaluation/                     # (M4) Evaluation and benchmarking
│   │   ├── __init__.py
│   │   ├── candidate_generator.py
│   │   ├── evaluate_end_to_end.py
│   │   └── metrics.py
│   ├── grammar/                        # (M4) Constrained decoding engines
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── fallback_engine.py
│   │   └── xgrammar_engine.py
│   ├── grounding/                      # (M1) Data pipeline and schema grounding
│   │   ├── __init__.py
│   │   ├── column_card.py
│   │   ├── embedding_retriever.py
│   │   ├── graph_builder.py
│   │   └── steiner_tree.py
│   ├── training/                       # (M2) Model training scripts
│   │   ├── __init__.py
│   │   ├── callbacks.py
│   │   ├── jepa_module.py
│   │   ├── train_decision_model.py
│   │   ├── train_embedding.py
│   │   ├── train_grpo_generator.py
│   │   └── train_sft_generator.py
│   └── validation/                     # (M4) Validation and System-1 scoring
│       ├── __init__.py
│       ├── db_probe.py
│       ├── laya_scorer.py
│       ├── pv_sql_rules.py
│       └── security_filter.py
└── tests/                              # (M5) Automated smoke test suite
    ├── __init__.py
    ├── test_evaluation.py
    ├── test_grammar_and_validation.py
    ├── test_grounding.py
    ├── test_notebooks.py
    └── test_training.py
```
