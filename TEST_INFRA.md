# E2E Test Infra: Text-to-SQL SLM Research Architecture

## Test Philosophy
- Opaque-box, requirement-driven testing independently derived from `ORIGINAL_REQUEST.md` and `docs/specifications/`.
- Zero heavy downloads and sub-60-second runtime on local CPU/developer environment without requiring multi-gigabyte HuggingFace models or GPU training sessions.
- Comprehensive 4-tier coverage validating all 4 project phases, real schema graphs, Steiner tree bridge preservation, loss computation, grammar engines, candidate generation, and validation rules.

## Feature Inventory
| # | Feature | Source (Requirement) | Tier 1 | Tier 2 | Tier 3 |
|---|---------|----------------------|:------:|:------:|:------:|
| 1 | PK-FK Graph & Bridge Tables | `docs/specifications/02` §1 | 5 | 5 | ✓ |
| 2 | Steiner Tree NWST Pruning | `docs/specifications/02` §1 | 5 | 5 | ✓ |
| 3 | Embedding Schema Grounding | `docs/specifications/02` §1 | 5 | 5 | ✓ |
| 4 | Completion-Only SFT Loss | `docs/specifications/02` §2 | 5 | 5 | ✓ |
| 5 | LLM-JEPA Cosine Distance Loss | `docs/specifications/02` §2 | 5 | 5 | ✓ |
| 6 | Dense GRPO 4-Head Rewards | `docs/specifications/02` §2 | 5 | 5 | ✓ |
| 7 | Google Drive Checkpoint Sync | `docs/specifications/05` §2 | 5 | 5 | ✓ |
| 8 | Colab Notebooks Structure & Configs | `docs/specifications/05` §1 | 5 | 5 | ✓ |
| 9 | XGrammar PDA & Fallback Strategy | `docs/specifications/02` §2 | 5 | 5 | ✓ |
| 10 | PV-SQL 10 AST Deterministic Rules | `docs/specifications/02` §3 | 5 | 5 | ✓ |
| 11 | DB Probing Engine | `docs/specifications/02` §3 | 5 | 5 | ✓ |
| 12 | System-1 Scorer & Early Exit Gate | `docs/specifications/02` §3 | 5 | 5 | ✓ |
| 13 | Multi-Candidate Generation (8-16) | `docs/specifications/02` §2 | 5 | 5 | ✓ |
| 14 | E2E Evaluation Metrics (EX, VSR) | `docs/specifications/04` §4 | 5 | 5 | ✓ |

## Test Architecture
- **Test Runner**:
  - `python -m unittest discover -s tests -p "test_*.py"` or `pytest tests/`
  - Single-command CLI runner: `python scripts/run_smoke_tests.py`
  - Pass/fail semantics: Exit code 0, all assertions pass, elapsed time < 60s.
- **Test Directories & Modules**:
  - `tests/test_grounding.py`: PK-FK graph creation, column card representation, Steiner tree bridge preservation (verifying 100% path discovery on `financial_mini.sqlite` and `california_schools.sqlite`).
  - `tests/test_training.py`: Completion-Only loss masking with -100, LLM-JEPA causal mask & $[PRED]$ token cosine distance, Dense GRPO 4-head reward computation, `DriveCheckpointCallback` file synchronization.
  - `tests/test_notebooks.py`: JSON validity of 5 `.ipynb` notebooks, cell syntax, Drive mount code inclusion, 3B & 7B configuration presence.
  - `tests/test_grammar_and_validation.py`: `BaseGrammarEngine` polymorphic behavior, `XGrammarPDAEngine` with PDA bitmask / grammar compile, `RegexASTFallbackEngine` zero-crash fallback, PV-SQL 10 AST rules, DB probing on mini SQLite DBs, System-1 early exit.
  - `tests/test_evaluation.py`: Candidate generator sampling 8–16 candidates, end-to-end evaluation harness on `data/mini/`, execution accuracy calculation.
- **Dataset**: `data/mini/` containing valid SQLite databases and sample query records.

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Complexity |
|---|----------|--------------------|------------|
| 1 | Multi-Hop Join Query across 3 tables | PK-FK Graph, Steiner Tree, Bridge Table, DDL generation | High |
| 2 | SFT + JEPA Forward-Backward Step | Prompt masking (-100), Cosine Distance, Gradient check | High |
| 3 | Dense GRPO Reward on Trajectories | Formatting, Execution, Memory Bank, Atomic clause match | High |
| 4 | Grammar PDA Enforcement & Fallback | Dynamic grammar compilation, Bitmask generation, Fallback | Medium |
| 5 | PV-SQL AST + DB Probing + System-1 Gate | 10 AST rules, SQLite execution, Early-exit thresholding | High |

## Coverage Thresholds
- Tier 1: ≥5 test cases per feature
- Tier 2: ≥5 boundary & corner test cases per feature
- Tier 3: Pairwise coverage across grounding, training, grammar, validation, and evaluation
- Tier 4: ≥5 realistic application scenarios
