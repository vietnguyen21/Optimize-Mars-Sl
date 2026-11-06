# Kaggle training and evaluation

## Status and evidence

These pipelines replace the earlier dry-run scaffolds. Historical mini-data accuracy and calibration statements in PROJECT.md/HANDOVER.md are **not pretrained model benchmark evidence**. Smoke outputs now carry `mode: smoke`, `trained: false` (training runners), and `is_benchmark: false` (evaluation). Real mode never silently substitutes synthetic training or gold-based generation. No Kaggle execution, full-model convergence, T4 memory bound, or benchmark target has been verified locally. See the validation section below for actual checks.

## Attachments and environment

Attach a snapshot of this existing repository to each notebook as a Kaggle Dataset; no clone, branch, push or GitHub access is needed. Set `KAGGLE_REPO_DIR` to the attached directory containing `src/` (for example `/kaggle/input/optimize-mars-repo/Optimize-Mars-Sl`). Run each notebook independently. No previous notebook's Python process state is required.

Set environment variables in a Python cell before the notebook's setup cell:

```python
import os
os.environ.update({
    'KAGGLE_REPO_DIR': '/kaggle/input/optimize-mars-repo/Optimize-Mars-Sl',
    'KAGGLE_DATA_DIR': '/kaggle/input/text2sql-data',
    'KAGGLE_OUTPUT_DIR': '/kaggle/working/optimize-mars-experiment-01',
    'PIPELINE_MODE': 'train',  # smoke is the explicit CPU wiring mode
    'INSTALL_DEPS': '1',      # opt in to requirements-kaggle.txt installation
    'TRAIN_DEVICE': 'cuda:0',
    'SEED': '42',
    'MAX_LENGTH': '512',
    'MAX_STEPS': '100',
    'SAVE_STEPS': '25',
    'GRAD_ACCUM': '8',
})
```

Use Internet-enabled Kaggle for first-time package/model downloads, or attach approved wheel/model datasets for offline use. `requirements-kaggle.txt` is a candidate Python 3.10/3.11 dependency lock, not a validated Kaggle image. Keep Kaggle's CUDA-enabled torch >=2.5. Optional XGrammar must be installed separately and its native/tokenizer compatibility verified. All installed package versions, Python, GPU name and GPU memory are recorded in `environment.json`.

`meta-llama/Llama-3.2-3B-Instruct` requires accepted model access terms and authenticated HF access. Obtain a token through Kaggle Secrets and authenticate using `huggingface_hub.login(token=...)` without printing it, or set `GENERATOR_MODEL` to an attached authorized local base model. Base weights remain required to load a PEFT adapter. Set `BASE_MODEL_PATH` to the corresponding attached base-model directory if the path stored in a shared adapter is unavailable in the receiving account/session. The override must refer to the same model and vocabulary used in SFT. The project does not export gated base weights.

One 16 GB T4 is the initial target: FP16 compute, NF4 double quantization for Generation, rank-16 LoRA, gradient checkpointing, one microbatch, sequential sampling. Two T4s have separate memory; the implementation does not pretend they provide pooled 32 GB VRAM. Use separate `TRAIN_DEVICE` values/processes for independent experiments if desired. No DDP is required. Start with 3B. A 7B configuration is optional and requires measured memory validation. Reduce sequence and rollout lengths on OOM. Some short-sequence datasets require a larger `MAX_LENGTH` for complete reasoning+SQL examples; SFT logs any excluded overlength examples. Inspect this count rather than assuming the full dataset was used.

## Data contracts and disjoint splits

All datasets accept UTF-8 JSONL or `.json` record lists. Databases may be `DB_DIR/db_id.sqlite` or `DB_DIR/db_id/db_id.sqlite`. Input databases are opened read-only. Dataset paths, model paths and output paths are configurable; never write output to `/kaggle/input`.

Split by database where possible for a schema generalization study, and always by question for train/calibration/test. Deduplicate normalized questions and audit split provenance before training. The tiny included dataset is suitable only for smoke tests; it is not enough for a research benchmark. Do not use retrieval evaluation questions in embedding training, calibration questions in classifier training, or evaluation questions in SFT/GRPO.

| Stage | Required record fields | Environment paths | Output/handoff |
|---|---|---|---|
| 01 Grounding | `db_id`, `question`, `SQL`; optional `positive_columns: ["table.column"]`, `evidence` | `TRAIN_DATA`, `RETRIEVAL_EVAL_DATA`, `DB_DIR`, optional `CARDS_PATH`, `EMBEDDING_MODEL` | `grounding/final/`, column index, metadata, schema graphs, retrieval metrics, checkpoints |
| 02 SFT/JEPA | `question`, `schema` DDL, `answer` SQL, optional `think_steps` | `SFT_DATA`, `GENERATOR_MODEL`, `LAMBDA_JEPA` | `sft_jepa_3b/checkpoint-N/` adapter/tokenizer/training state |
| 03 GRPO | `db_id`, `question`, `SQL` (or `answer`), schema DDL and optional reasoning/evidence | `SFT_CHECKPOINT`, `GRPO_DATA`, `DB_DIR`, `GROUP_SIZE`, `MAX_NEW_TOKENS` | `grpo_3b/checkpoint-N/` optimized adapter and complete training state |
| 04 System-1 | `db_id`, `question`, `schema`, `sql` candidate, execution-derived `label: 0 or 1` | `DECISION_DATA`, `CALIBRATION_DATA`, `DECISION_TEST_DATA`, `DECISION_MODEL` | `system1/final/` HF classifier/tokenizer/calibration metadata, checkpoints |
| 05 Evaluation | held-out `db_id`, `question`, `SQL` used for scoring only, optional `evidence` | `GENERATOR_CHECKPOINT`, `DECISION_CHECKPOINT`, `GROUNDING_CHECKPOINT`, `EVAL_DATA`, `DB_DIR`, `TRAINING_DATA_PATHS` | independent predictions, metric report, per-question latency/decision logs |

Default real input filenames under `KAGGLE_DATA_DIR`: `train.jsonl`, `retrieval_dev.jsonl`, `sft_train.jsonl`, `grpo_train.jsonl`, `decision_train.jsonl`, `decision_calibration.jsonl`, `decision_test.jsonl`, `test.jsonl`, `databases/`. Override every path independently. Notebook01 extracts cards for both train and evaluation database schemas without adding evaluation query labels to training. External cards must include `db_id` when multiple databases are involved. Prefer explicit `positive_columns` annotations. SQL-derived labels require SQLGlot and reject ambiguous column references rather than silently guessing. HN-SupCon mines current-model same-database hard negatives, retains gradients, and evaluates held-out retrieval recall/hit rate.

### Building candidate labels for System-1

Use existing independently sampled candidate artifacts, or sample with a trained Generation Agent; do not label arbitrary corruptions as wrong without execution. Candidate-source JSONL records contain `question`, `db_id`, `schema`, gold `SQL`, and `candidates: ["SELECT ...", ...]`.

```python
from src.training.train_decision_model import build_candidate_dataset, split_candidate_groups
from pathlib import Path
import json
rows = build_candidate_dataset('/kaggle/input/candidates/sampled_candidates.jsonl',
                               db_dir='/kaggle/input/text2sql-data/databases',
                               include_gold=False)
train, calibration, test = split_candidate_groups(rows, seed=42)
for name, split in [('train', train), ('calibration', calibration), ('test', test)]:
    Path(f'/kaggle/working/decision_{name}.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in split))
```

Alternatively `build_candidate_dataset(..., generator=checkpoint_backed_generator)` accepts a real `CandidateGenerator`. The builder executes candidates against gold results to derive labels and never puts gold SQL in classifier input. Gold augmentation is off by default. If used explicitly for the training split, keep calibration/test sampled-only and externally disjoint; do not auto-split a gold-augmented set into calibration/test. The classifier requires both correct and incorrect training candidates. A generator with no successful responses cannot supply positive examples; this is a data problem and is reported rather than hidden.

System-1 uses the ModernBERT sequence classification head with BCE+Brier calibration-aware loss and distinct backbone/head learning rates. Temperature is selected by held-out negative log likelihood; test ECE is measured once using this temperature. The 0.081 ECE value is a research target, never a manufactured acceptance result. Classification generates no explanation tokens; latency must still be measured on the target hardware.

## Objectives and ablations

SFT constructs prompt and completion token sequences separately, masks all prompt/padding positions with `-100`, and retains real EOS labels. `[PRED]` is a fixed textual prompt anchor (its final subtoken), not a newly learned vocabulary entry. A causal forward pass prevents the anchor from reading completion tokens. The trainable JEPA predictor aligns this prompt-only state with a stop-gradient SQL span target from the causal completion, with configurable loss dropout. This is the implemented causal latent alignment variant; it is not a claim of reproducing an external JEPA paper exactly. Predictor and optimizer state are checkpointed. Use `LAMBDA_JEPA=0` and a separate output root for SFT-only, then `0.1` for SFT+JEPA with matched seeds and budgets.

GRPO starts from the trained SFT adapter, samples groups from the model, evaluates existing format/execution/reasoning/atomic rewards, and computes token-level relative advantages and policy/KL gradients. A frozen SFT reference adapter shares the quantized backbone. Zero reward variance produces zero advantages; logs expose these groups and skip their reward policy update. Dense reward heads do not mathematically guarantee nonzero variance. The reasoning reward is the existing lexical reasoning-bank similarity, not an added pretrained semantic judge. Vary group size and reward composition only with separate logged configurations.

## Checkpoint persistence and recovery

Each checkpoint is first written to a unique hidden staging directory and published by rename only after completion. Interrupted staging directories are preserved for inspection and never chosen by automatic recovery. Every custom-loop `checkpoint-N` contains native model or PEFT weights, tokenizer where applicable, `training_state.pt` with optimizer/scaler/RNG/objective/data cursor state, and a `COMPLETE` marker written last. Resume accepts only complete checkpoints. Dataset fingerprints and training contracts prevent accidental continuation on different data/objectives. Pickle-based optimizer state must come from a trusted local or team artifact. Checkpoints are immutable and are never automatically pruned. Final Grounding/System-1 exports refuse an existing destination; use a fresh output root or preserve the old export before extending a run.

Set `RESUME_FROM_CHECKPOINT=latest` for the current output directory, or an explicit attached checkpoint directory for a new Kaggle session. The next checkpoint is written to the configurable new output directory. Download outputs or save a notebook version **before a session ends**; `/kaggle/working` is ephemeral. Saving a checkpoint locally is not cloud synchronization. Interrupted incomplete checkpoints remain for inspection and are ignored by automatic discovery.

Grounding, SFT and System-1 can run independently on separate legitimate team accounts using the same versioned inputs and configuration. Share their saved output datasets through normal Kaggle mechanisms; do not depend on another account's live session or use extra accounts to evade service limits. GRPO attaches the SFT artifact, and evaluation attaches Generation+System-1+Grounding artifacts. Record source dataset versions/checkpoint hashes in experiment records.

## Held-out evaluation

Set `SCORER_DEVICE=cpu` and `GROUNDING_DEVICE=cpu` initially to reserve GPU memory for the Generation Agent. CPU latency is part of end-to-end latency. Real evaluation requires trained Generation and System-1 checkpoints. Omitting `GROUNDING_CHECKPOINT` is explicitly reported as a **lexical Grounding ablation**, not the complete trained system.

`GRAMMAR_MODE=posthoc` performs grammar/AST rejection after generation; it is not token-constrained decoding. Optional `GRAMMAR_MODE=xgrammar` uses a genuine HF logits processor and native XGrammar. Its documented SELECT subset is not the full SQLite language (CTEs/window functions and some valid expressions are excluded); AST verification and SQLite probing remain required. No grammar guarantees execution correctness or schema/semantic correctness. Reports record the actual constraint mode/scope.

Set `TRAINING_DATA_PATHS` to all SFT, GRPO, embedding, and candidate training data paths separated by `os.pathsep` (`:` on Kaggle). Evaluation rejects overlapping question/database pairs and records whether this audit was performed; no paths means **not audited**, not proof of no leakage. Preserve training data attachments to make the audit possible. Held-out gold SQL is accessed only by metric computation after prediction. Read-only execution has deadlines; timeouts are recorded as failures. Rejected candidates are not relabeled as accepted; abstentions stay explicit. EX uses result comparison (ordered when gold contains ORDER BY); VSR measures executable selected predictions, not a promised 100%. Report coverage and accepted-only ECE alongside overall ECE to interpret abstentions.

## Local validation

Run from the repository root:

```bash
python -m unittest discover -s tests -v
python scripts/run_smoke_tests.py
```

Notebook tests execute all cells with `PIPELINE_MODE=smoke` and temporary output directories, validate syntax/internal imports and stale outputs. Additional tests cover no-gold generation, strict real-versus-smoke dispatch, label construction, grouped calibration splits, checkpoint completion/discovery, gradient losses and checkpoint/RNG roundtrips (PyTorch-required tests skip honestly when unavailable). Real tensor checks can run with CPU PyTorch; pretrained model loading, bitsandbytes/CUDA, native XGrammar token masks, real convergence and measured T4 memory/latency require target hardware/dependencies. No benchmark numbers are inferred from these checks.

### Recorded validation for this change

Local CPU validation on Python 3.14 used torch 2.15.0+cpu, transformers 4.51.3, PEFT 0.15.2, sentence-transformers 4.1.0, accelerate 1.6.0, SQLGlot 26.16.4 and nbformat 5.10.4 in an isolated temporary environment. This is not the target Kaggle CUDA image.

| Phase / modified components | Validation and remaining limit |
|---|---|
| Shared `runtime.py`, `kaggle.py`, callbacks and notebook setup | Mode/path tests, actual optimizer/RNG checkpoint roundtrip, interrupted-save retry, complete-checkpoint discovery; no automatic deletion. CUDA NF4 loader remains untested. |
| Grounding / Notebook01 / `train_embedding.py` | Real native tiny SentenceTransformer training, same-database hard negatives, held-out retrieval, native embedding/index export and exact resumed parameters. Full 0.6B GPU run untested. |
| SFT+JEPA / Notebook02 / Generation modules | Native tiny Llama+PEFT training, completion/EOS/padding masks, SQL-only latent targets, SFT-only ablation and exact epoch-boundary adapter/predictor resume. Gated 3B QLoRA GPU execution untested. |
| GRPO / Notebook03 / `train_grpo_generator.py` | Native tiny Llama sampled updates, immutable reference adapter export/load, token policy objective, zero-variance handling and exact resumed policy parameters. Optimizer-isolation tests mock rewards; separate tests exercise all four real reward heads. Full-model rollouts/VRAM untested. |
| System-1 / Notebook04 / `train_decision_model.py`, `laya_scorer.py` | Native tiny ModernBERT training, execution-derived labels (including ordered results), calibration/test splits, temperature, actual exported scorer inference and exact resumed parameters. Full-size classifier GPU run untested. |
| Evaluation / Notebook05 / evaluation, grammar and DB probe modules | Independent inference boundary, actual tensor generation/parsing/log probabilities, SQLGlot ASTs, safe SQLite deadlines, abstentions/metrics, overlap audit and all five notebook smoke executions. Native XGrammar compilation and full-system pretrained benchmarking untested. |

The complete suite passes **197 tests with no skips** in the isolated HF/CPU environment. The dependency-light system Python suite passes with **18 explicitly skipped dependency-required tests**. The standalone smoke runner passes all six suites without asserting a benchmark target. All five notebooks validate with nbformat and all 45 Python source/test/script files compile in memory. `pip check` reports no broken dependencies in the isolated environment. Final test output is retained in `notebooks/hf-suite.log`, `notebooks/cpu-suite.log` and `notebooks/smoke-validation.log`.

No full pretrained-model training, Kaggle run, CUDA/bitsandbytes operation, T4 memory bound, native XGrammar compilation or research benchmark result is asserted. Those are the remaining target-environment acceptance checks.
