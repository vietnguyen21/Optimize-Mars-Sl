# Kaggle training status — Optimize-Mars-Sl

Status recorded: 2026-11-06. This report describes the current local implementation and the validation logs retained in this repository. It does not certify a successful Kaggle run.

**Implemented and CPU-validated:** real training objectives, native tiny-model training/export/resume, and independent evaluation code. **Not yet accepted on target hardware:** full-size pretrained training, CUDA/NF4 operation, native XGrammar, and measured T4 memory/latency. Historical scaffold accuracy claims in older project documents are not benchmark evidence.

## 1. Implemented features

| Component | Implemented behavior |
|---|---|
| Shared infrastructure | Environment-configurable repository/data/output paths; explicit `train` and `smoke` modes; seeded RNGs; experiment JSONL logs; environment/package/GPU inventory; dataset fingerprints and resume contracts. |
| Memory controls | Generation uses single-device NF4 double quantization with FP16 compute, rank-16 LoRA, gradient checkpointing, small microbatches and sequential rollouts. Grounding/System-1 use gradient accumulation and CUDA AMP when applicable. These are implementation choices, not measured T4 fit guarantees. |
| Checkpoints | Native model/PEFT weights, tokenizers where applicable, optimizer/scaler/RNG state, JEPA predictor and data cursor/configuration. Hidden staging directories are published atomically as complete checkpoints. Interrupted saves can be retried; existing checkpoints are not replaced or automatically pruned. |
| Notebook 01 — Grounding | Database-scoped column cards, retained PK–FK schema graphs and Steiner linking; differentiable HN-SupCon with current-model hard-negative mining; trained embedding/model/index export; held-out retrieval recall and hit rate. |
| Notebook 02 — SFT + JEPA | Actual Llama-3.2-3B-Instruct QLoRA updates; exact prompt/completion boundaries; prompt/padding loss masking with real EOS retained; trainable latent predictor and detached SQL-span targets; SFT-only/SFT+JEPA ablations; complete resume. |
| Notebook 03 — GRPO | Loads trained SFT adapters; samples responses from the policy; uses format, execution, reasoning and atomic rewards; token-level clipped policy objective and reference KL; immutable SFT reference adapter; exact zero advantages for constant-reward groups; logging and recovery. |
| Notebook 04 — System-1 | Execution-derived candidate labels; grouped question splits; ModernBERT sequence classification; separate backbone/head learning rates; BCE+Brier calibration-aware optimization; held-out temperature fitting by NLL; test ECE; trained classifier/tokenizer/temperature export. |
| Notebook 05 — Evaluation | Loads real artifacts; generation receives question/schema rather than gold SQL; trained dense Grounding or labeled lexical ablation; optional native constrained decoding; AST validation and bounded read-only database probing; genuine abstentions; EX, VSR, ECE, accepted-only ECE, coverage, latency and VRAM reporting; optional training-overlap audit. |

Smoke mode exercises fixtures and wiring only. Training summaries explicitly identify smoke outputs as untrained; evaluation identifies them as non-benchmarks. Real-mode dependency/model-loading errors are not converted into synthetic success.

## 2. Modified and added files

Paths below are relative to the repository root. This is an implementation inventory, not a claim that unrelated local files were edited.

| Area | Modified existing files | Added files |
|---|---|---|
| Documentation/dependencies | `PROJECT.md`, `HANDOVER.md`, `requirements.txt` | `KAGGLE.md`, `requirements-kaggle.txt`, this status report |
| Shared training | `src/training/callbacks.py` | `src/training/runtime.py`, `src/training/kaggle.py` |
| Training objectives | `src/training/train_embedding.py`, `src/training/train_sft_generator.py`, `src/training/jepa_module.py`, `src/training/train_grpo_generator.py`, `src/training/train_decision_model.py` | — |
| Inference/validation | `src/evaluation/candidate_generator.py`, `src/evaluation/evaluate_end_to_end.py`, `src/evaluation/metrics.py`, `src/grammar/xgrammar_engine.py`, `src/validation/db_probe.py`, `src/validation/laya_scorer.py` | — |
| Notebooks | `notebooks/01_data_preparation_and_embedding.ipynb`, `notebooks/02_sft_and_jepa_training_3b.ipynb`, `notebooks/03_grpo_training_3b_and_7b.ipynb`, `notebooks/04_system1_laya421m_training.ipynb`, `notebooks/05_end_to_end_evaluation.ipynb` | — |
| Tests/runner | `scripts/run_smoke_tests.py`, `tests/test_evaluation.py`, `tests/test_grammar_and_validation.py`, `tests/test_training.py`, `tests/test_notebooks.py`, `tests/test_m2_stress.py`, `tests/test_m3_stress.py` | `tests/test_runtime.py`, `tests/test_generation_training.py`, `tests/test_grounding_system1_pipeline.py`, `tests/test_hf_grounding_system1.py` |
| Validation evidence | — | `notebooks/hf-suite.log`, `notebooks/cpu-suite.log`, `notebooks/smoke-validation.log` |

Legacy Colab-specific tests were updated for Kaggle configuration. Tests expecting checkpoint deletion were updated to require preservation. No Git branch, commit, push or remote-history changes were made during implementation. Existing tracked research data was preserved. Generated `__pycache__` files are interpreter caches, not trained artifacts or implementation deliverables.

## 3. Recorded test results

These are the preceding implementation session's recorded results, checked against the retained logs while preparing this report. Writing this Markdown report does not constitute a new training run.

| Check | Recorded result | Evidence |
|---|---|---|
| Complete suite with CPU PyTorch and native HF dependencies | 197 tests, all passed, no skips; 4.429 seconds | [HF suite log](notebooks/hf-suite.log) |
| Dependency-light system Python suite | 197 discovered; successful run with 18 explicitly skipped dependency-required tests; 1.246 seconds | [CPU suite log](notebooks/cpu-suite.log) |
| Standalone smoke runner | All six suites passed; 1.023 seconds; no model-quality target asserted | [Smoke log](notebooks/smoke-validation.log) |
| Notebook validation | All five passed nbformat validation and code-cell syntax checks | Recorded in [KAGGLE.md](KAGGLE.md#recorded-validation-for-this-change) |
| Python syntax | All 45 source/test/script files compiled in memory | Recorded in `KAGGLE.md` |
| Dependency consistency | `pip check`: no broken requirements in the isolated validation environment | Recorded in `KAGGLE.md` |

The native CPU environment used Python 3.14, torch 2.15.0+cpu, transformers 4.51.3, PEFT 0.15.2, sentence-transformers 4.1.0, accelerate 1.6.0, SQLGlot 26.16.4 and nbformat 5.10.4. It was not a Kaggle CUDA image.

Tests include real locally initialized tiny SentenceTransformer, ModernBERT and Llama/PEFT models; actual parameter updates; model/tokenizer export and loading; exact resumed-versus-uninterrupted parameters; JEPA predictor recovery; immutable reference adapters; checkpoint/RNG restoration; interrupted-save retries; masking; SQL-derived labels; and no-gold inference boundaries. GRPO optimizer-isolation tests mock rewards; separate reward tests exercise actual formatting, reasoning, AST and SQLite execution logic. Tiny-model success does not establish convergence or memory fit for the full models.

To repeat checks from the repository root in an environment with the desired dependencies:

```bash
python -m unittest discover -s tests -v
python scripts/run_smoke_tests.py
```

Inspect skip counts. A dependency-light pass is not equivalent to the native-model pass.

## 4. Incomplete functionality and remaining acceptance work

| Item | Current limitation / next acceptance check |
|---|---|
| Kaggle GPU execution | No notebook has been run on Kaggle during this work. Run each real stage on the target image, retain `environment.json`, measure memory, and verify exported artifacts in a fresh session. |
| Full-size training | Qwen3-Embedding-0.6B, Llama-3.2-3B-Instruct, optional 7B Generation and ModernBERT-large pretrained training have not been executed. Convergence, hyperparameter adequacy and benchmark quality remain unknown. |
| CUDA/bitsandbytes | NF4 and FP16 GPU paths are implemented but not validated locally. The dependency file is a candidate direct-version lock, not a certified complete Kaggle environment. |
| Multiple GPUs | No DDP/FSDP or pooled-memory training is implemented. The initial configuration uses one T4; two 16 GB GPUs do not become one 32 GB device. A 7B run is not a promised T4 configuration. |
| JEPA methodology | The implemented variant uses the last subtoken of literal `[PRED]`, standard causal attention and a detached SQL target from the same forward pass. It does not add a learned single vocabulary token or an EMA target encoder, and is not claimed as an exact external-paper reproduction. |
| GRPO reward quality | Reasoning reward retains lexical reasoning-bank similarity. Dense rewards cannot guarantee nonzero group variance. Constant-reward groups skip their reward-driven policy update; a run may finish with no policy updates. |
| Constrained grammar | Default `posthoc` mode rejects invalid outputs after sampling; it is not token masking. Optional `xgrammar` uses native masks for a restricted SELECT grammar, not full SQLite. Native compilation/tokenizer integration remains untested; CTEs/window functions and other valid SQL forms may be excluded. |
| Data and calibration | Real datasets, sampled candidates and scientifically appropriate splits must be supplied. No full research dataset, trained production checkpoint, target ECE or benchmark result was created. Candidate training requires both correct and incorrect examples. |
| Leakage audit | Evaluation checks supplied `TRAINING_DATA_PATHS`; without them it records that overlap was not audited. Checks are not a substitute for dataset provenance and normalized deduplication. |
| Persistence | Local checkpoints are real, but `/kaggle/working` is ephemeral. Automatic Kaggle cloud synchronization is not implemented. Save/download outputs before ending a session. |
| Re-running completed outputs | Checkpoints are immutable; Grounding/System-1 final exports and evaluation reports protect existing destinations. Use new experiment output roots, including when extending a completed training run. |

No EX, VSR, ECE, latency or VRAM research target has been demonstrated by the CPU checks.

## 5. Common Kaggle setup

Attach a repository snapshot as a Kaggle Dataset, attach the experiment's data/artifacts, and open the corresponding existing notebook. No clone or GitHub changes are required. Select the GPU runtime for real training. Enable downloads when needed or provide local wheels and authorized model weights.

Add this configuration cell **before** the notebook setup cells. Replace example dataset paths with the actual attached paths:

```python
import os
os.environ.update({
    'KAGGLE_REPO_DIR': '/kaggle/input/optimize-mars-repo/Optimize-Mars-Sl',
    'KAGGLE_DATA_DIR': '/kaggle/input/text2sql-data',
    'KAGGLE_OUTPUT_DIR': '/kaggle/working/experiment-01',
    'PIPELINE_MODE': 'train',
    'INSTALL_DEPS': '1',
    'TRAIN_DEVICE': 'cuda:0',
    'SEED': '42',
    'MAX_LENGTH': '512',
    'MAX_STEPS': '100',
    'SAVE_STEPS': '25',
    'GRAD_ACCUM': '8',
})
os.environ.pop('RESUME_FROM_CHECKPOINT', None)  # Fresh experiment only.
```

`INSTALL_DEPS=1` installs `requirements-kaggle.txt`. Preserve Kaggle's CUDA-enabled torch >=2.5; do not install the local test environment's CPU torch on Kaggle. Use a fresh kernel after changing already imported package versions. XGrammar is optional and installed separately. The notebook records the actual installed inventory.

Llama access requires the appropriate model authorization. Authenticate with an HF token obtained through Kaggle Secrets without printing it, or use an authorized attached base model. PEFT adapters do not contain base weights. Set `BASE_MODEL_PATH` when a shared adapter's stored base-model path differs in the receiving session; it must identify the same base model/vocabulary.

Use disjoint questions for training, calibration and test; prefer disjoint databases for schema-generalization experiments. JSONL and `.json` record lists are supported. Databases can be `DB_DIR/db_id.sqlite` or `DB_DIR/db_id/db_id.sqlite`. Do not train research models on the included mini smoke set.

Common output prefixes are:

```text
KAGGLE_OUTPUT_DIR/train/grounding/
KAGGLE_OUTPUT_DIR/train/sft_jepa_3b/
KAGGLE_OUTPUT_DIR/train/grpo_3b/
KAGGLE_OUTPUT_DIR/train/system1/
KAGGLE_OUTPUT_DIR/train/evaluation/
```

Smoke mode uses a separate `smoke/` prefix. `MAX_STEPS` caps optimizer updates; SFT and GRPO also stop at their configured epoch limits. `GRAD_ACCUM` is wired into notebooks 02/03; notebooks 01/04 currently use the runners' default accumulation of eight. To change other runner settings such as epochs or learning rate, edit the existing call explicitly and record the configuration. Arbitrary environment variables are not automatically forwarded.

## 6. Per-experiment instructions

Run each configuration in a fresh kernel and a distinct output root. Execute the notebook cells in order after the configuration cells below.

### Experiment A — Grounding

Notebook: `notebooks/01_data_preparation_and_embedding.ipynb`.

Input rows need `db_id`, `question`, `SQL`, with optional `evidence` and explicit `positive_columns: ["table.column", ...]`. SQL-derived labels require SQLGlot and reject ambiguous references. Provide held-out retrieval questions and their database schemas.

```python
os.environ.update({
    'KAGGLE_OUTPUT_DIR': '/kaggle/working/grounding-seed42',
    'TRAIN_DATA': '/kaggle/input/text2sql-data/train.jsonl',
    'RETRIEVAL_EVAL_DATA': '/kaggle/input/text2sql-data/retrieval_dev.jsonl',
    'DB_DIR': '/kaggle/input/text2sql-data/databases',
    'EMBEDDING_MODEL': 'Qwen/Qwen3-Embedding-0.6B',
})
```

Notebook01 builds cards/graphs from both train and retrieval-evaluation database schemas; only training query labels enter the optimizer. Optional `CARDS_PATH` supplies prebuilt cards; include database IDs when multiple databases are represented.

Save the entire `train/grounding/` directory. Downstream `GROUNDING_CHECKPOINT` must point to its `final/` model directory. Inspect retrieval metrics, index metadata, optimizer steps and logs; no recall target is guaranteed.

### Experiment B — SFT-only baseline

Notebook: `notebooks/02_sft_and_jepa_training_3b.ipynb`.

Input rows need `question`, `schema` DDL and `answer` SQL, with optional `think_steps`.

```python
os.environ.update({
    'KAGGLE_OUTPUT_DIR': '/kaggle/working/sft-only-seed42',
    'SFT_DATA': '/kaggle/input/text2sql-data/sft_train.jsonl',
    'GENERATOR_MODEL': 'meta-llama/Llama-3.2-3B-Instruct',
    'LAMBDA_JEPA': '0',
})
```

Inspect `dropped_overlength`: overlong examples are excluded with logging rather than trained with a truncated/missing answer. Adjust preprocessing/sequence budget if exclusions invalidate the experiment. Export the returned `summary['checkpoint_path']`, including adapter, tokenizer and training state.

### Experiment C — SFT + JEPA ablation

Use the same notebook, data, base model, seed, epochs and update budget as Experiment B, but change:

```python
os.environ.update({
    'KAGGLE_OUTPUT_DIR': '/kaggle/working/sft-jepa-seed42',
    'LAMBDA_JEPA': '0.1',
})
```

Start a new run, not a resume from SFT-only: the objective contract differs. Compare actual CE/JEPA logs and held-out Generation results. The output phase remains `sft_jepa_3b` for both variants; the distinct root separates experiments.

### Experiment D — Dense GRPO from SFT

Notebook: `notebooks/03_grpo_training_3b_and_7b.ipynb`.

Attach one chosen SFT artifact. Input rows need `db_id`, `question`, gold `SQL`/`answer`, and optional reasoning/evidence. Supply schema DDL where available; otherwise the runner reads it from SQLite. Gold labels enter rewards, not the generation prompt.

```python
os.environ.update({
    'KAGGLE_OUTPUT_DIR': '/kaggle/working/grpo-from-jepa-seed42',
    'SFT_CHECKPOINT': '/kaggle/input/sft-jepa-artifacts/train/sft_jepa_3b/checkpoint-100',
    'GRPO_DATA': '/kaggle/input/text2sql-data/grpo_train.jsonl',
    'DB_DIR': '/kaggle/input/text2sql-data/databases',
    'GROUP_SIZE': '2',
    'MAX_NEW_TOKENS': '128',
})
```

`checkpoint-100` is an example: use the actual exported checkpoint path. Start with the short single-T4 configuration. Inspect four-head reward logs, skipped-overlength/zero-variance groups and actual optimizer steps. Export the returned GRPO checkpoint, including its frozen `reference/` adapter. Evaluate SFT versus GRPO on the same held-out set with separate evaluation output roots.

### Experiment E — System-1 classification and calibration

Notebook: `notebooks/04_system1_laya421m_training.ipynb`.

Option 1: attach three question-disjoint labeled files. Each row needs `db_id`, `question`, `schema`, candidate `sql`, and execution-derived binary `label`.

```python
os.environ.update({
    'KAGGLE_OUTPUT_DIR': '/kaggle/working/system1-seed42',
    'DECISION_DATA': '/kaggle/input/decision-data/decision_train.jsonl',
    'CALIBRATION_DATA': '/kaggle/input/decision-data/decision_calibration.jsonl',
    'DECISION_TEST_DATA': '/kaggle/input/decision-data/decision_test.jsonl',
    'DECISION_MODEL': 'answerdotai/ModernBERT-large',
})
os.environ.pop('CANDIDATE_SOURCE', None)
```

Option 2: set `CANDIDATE_SOURCE` to JSONL containing `db_id`, `question`, `schema`, reference `SQL`, and `candidates: ["SELECT ...", ...]`, and set `DB_DIR`. The notebook executes candidates, constructs labels and creates grouped train/calibration/test files in its output directory. It does not generate candidate rollouts automatically in this branch. Provide independently sampled candidates from an existing model artifact. No reference positives are injected by default.

Both training labels must be present. Gold augmentation, if deliberately used for training, must not enter calibration/test; supply separately prepared splits in that case. Save `train/system1/final/`, including classifier weights, tokenizer and `calibration_meta.json`. Inspect measured unscaled/scaled test ECE rather than asserting the 0.081 research target.

### Experiment F — Held-out end-to-end evaluation

Notebook: `notebooks/05_end_to_end_evaluation.ipynb`.

Attach trained Grounding, Generation and System-1 outputs, held-out questions and databases. Input gold `SQL` is consumed after predictions for metrics.

```python
os.environ.update({
    'KAGGLE_OUTPUT_DIR': '/kaggle/working/eval-grpo-k4-seed42',
    'GENERATOR_CHECKPOINT': '/kaggle/input/grpo-artifacts/train/grpo_3b/checkpoint-100',
    'GROUNDING_CHECKPOINT': '/kaggle/input/grounding-artifacts/train/grounding/final',
    'DECISION_CHECKPOINT': '/kaggle/input/system1-artifacts/train/system1/final',
    'EVAL_DATA': '/kaggle/input/text2sql-data/test.jsonl',
    'DB_DIR': '/kaggle/input/text2sql-data/databases',
    'NUM_CANDIDATES': '4',
    'MAX_NEW_TOKENS': '256',
    'MAX_PROMPT_LENGTH': '2048',
    'RETRIEVAL_TOP_K': '25',
    'SCORER_DEVICE': 'cpu',
    'GROUNDING_DEVICE': 'cpu',
    'GRAMMAR_MODE': 'posthoc',
    'TRAINING_DATA_PATHS': os.pathsep.join([
        '/kaggle/input/text2sql-data/train.jsonl',
        '/kaggle/input/text2sql-data/sft_train.jsonl',
        '/kaggle/input/text2sql-data/grpo_train.jsonl',
        '/kaggle/input/decision-data/decision_train.jsonl',
    ]),
})
```

Use actual artifact paths and include all relevant training manifests. Overlap auditing expects database/question identifiers in those audit records; preserve them in training data. Without the manifests, the report explicitly marks the audit as absent.

For constrained-decoding experiments, install/verify native XGrammar and use `GRAMMAR_MODE=xgrammar` in a separate run. This exercises only the implemented SQL subset. Compare with `posthoc` under matched settings. Leaving `GROUNDING_CHECKPOINT` unset is a reported lexical Grounding ablation, not the complete trained system.

Compare SFT-only, SFT+JEPA and GRPO checkpoints with fixed held-out data and seeds. Sweep `NUM_CANDIDATES` through 1, 4, 8 and 16 using distinct output roots to study accuracy/latency. Reports are written exclusively, so re-running the same destination intentionally fails instead of replacing previous results.

Save `train/evaluation/metrics.json`, `benchmark_summary.md`, and `logs/evaluation_results.jsonl`. Interpret EX/VSR with abstention coverage and accepted-only ECE; CPU Grounding/scoring time is included in end-to-end latency. No metric should be copied from smoke outputs into a research benchmark.

### Optional — 7B Generation

A 7B experiment is not automatically selected by notebook03's filename. To attempt Qwen2.5-Coder-7B, first train its own SFT adapter using notebook02 with the authorized 7B base in `GENERATOR_MODEL`, then pass that corresponding adapter to notebook03. Do not attach a Llama 3B adapter to a 7B base. Use distinct output roots; current phase-directory names retain `3b`, so record actual model identity explicitly. Validate memory before increasing budgets. Full 7B training and T4 compatibility remain unverified.

## 7. Smoke checks, recovery and team handoff

For an offline wiring run, replace the common configuration with `PIPELINE_MODE=smoke`, `TRAIN_DEVICE=cpu`, `INSTALL_DEPS=0`, and set `KAGGLE_DATA_DIR` to the attached repository's `data/mini`. Use a new output root and remove production data/checkpoint overrides from the kernel environment. Smoke mode does not train pretrained models or establish benchmark quality.

For recovery in a new session, attach a complete exported checkpoint and set:

```python
os.environ['KAGGLE_OUTPUT_DIR'] = '/kaggle/working/recovered-experiment'
os.environ['RESUME_FROM_CHECKPOINT'] = '/kaggle/input/previous-run/train/sft_jepa_3b/checkpoint-100'
```

Use the appropriate stage checkpoint, unchanged data/objective/seed contracts and a sufficient epoch/update budget. `latest` searches the current stage output directory; it does not discover arbitrary attached datasets. Reattach base weights and original SFT artifacts for GRPO as required. Use trusted training-state files, since optimizer/RNG recovery uses serialized Python state.

Grounding, SFT and System-1 can run independently on legitimate team accounts using versioned input artifacts. GRPO depends on SFT; full evaluation depends on Grounding + Generation + System-1. Export the entire stage directory, save/download Kaggle outputs before the session ends, then attach those saved outputs downstream. Share artifacts through normal Kaggle mechanisms and record input versions, configurations and source checkpoint identities. No automatic persistent-storage transfer is performed by these notebooks.

Further reference: [KAGGLE.md](KAGGLE.md). The remaining completion gate is measured real-mode execution of every stage on the target Kaggle GPU environment, followed by independent held-out evaluation.
