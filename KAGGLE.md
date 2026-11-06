# Official Kaggle training guide — three-person research team

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

## Team workflow and dependencies

Follow the setup, member instructions, handoff and acceptance steps below in order. The later data/objective/checkpoint sections remain the technical reference. All paths in executable cells are examples of attached Dataset layouts: replace them with the paths shown in your own Kaggle input panel. Run the common configuration cell above, then the relevant member cell **before any notebook setup cell**. Use a fresh kernel for each experiment; setup captures environment values once.

| Owner | Work and deliverables | Dependencies |
|---|---|---|
| Member 1 — Grounding Agent | Notebook 01, Qwen3-Embedding-0.6B; database column cards, PK–FK schema graphs, Node-Weighted Steiner Tree integration, embedding/index export and retrieval validation; support final integration | Repository, training/retrieval splits, SQLite schemas, model weights |
| Member 2 — Generation Agent | Notebook 02 SFT-only and SFT+JEPA in separate runs; Notebook 03 Dense GRPO initialized from selected SFT checkpoints; adapter and reward diagnostics | SFT data/base weights first; GRPO additionally needs SFT artifact, reward data and SQLite databases |
| Member 3 — Validation Agent | Notebook 04 ModernBERT System-1 training and calibration; own evaluation harness configuration, split audit and benchmark records | Prebuilt execution-labeled candidates allow independent start; new candidate sampling depends on Member 2's trained generator |
| All three | Notebook 05 final evaluation: Member 1 verifies retrieval/schema integration, Member 2 verifies generator/base identity, Member 3 runs and audits evaluation; all review results | Accepted Grounding + Generation + calibrated System-1 artifacts and frozen held-out data |

01 and 02 can run concurrently; 04 can also run concurrently when independently sampled candidate artifacts already exist. 03 follows 02. Notebook 04's `CANDIDATE_SOURCE` branch labels supplied candidates; it does **not** automatically sample a generator. No training stage requires another notebook's live Python state.

### Common preparation for each account

1. Upload/open the assigned existing `.ipynb` in Kaggle. Attach the same versioned repository snapshot and licensed research data. Select a GPU runtime for real training; start on `cuda:0` of one T4. Set the common cell above and your member cell below. Use a unique output root for every run.
2. Enable Internet for initial package/model downloads, or attach compatible wheels and authorized complete local model/tokenizer directories. The notebook's automatic installer uses pip and does not configure an offline wheel directory for you. Install offline wheels explicitly before setup if needed, then use `INSTALL_DEPS=0`. Keep CUDA torch >=2.5. The pinned direct dependencies in [requirements-kaggle.txt](requirements-kaggle.txt) are a candidate Python 3.10/3.11 lock, not a certified Kaggle environment. Restart the kernel after changing imported packages and rerun configuration.
3. For Member 2 and the evaluation operator, authorize the gated Llama base first. With a Kaggle Secret named `HF_TOKEN`, run the following before model loading when downloading from HF:

```python
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
login(token=UserSecretsClient().get_secret("HF_TOKEN"), add_to_git_credential=False)
```

4. Execute the notebook in order. Confirm printed mode, data, output, GPU and resume path before the training cell. Retain `environment.json`, configuration cells, data versions and actual model identity with the artifact.

### Member 1 setup — Notebook 01

Attach repository + `text2sql-data` with training/retrieval files and SQLite databases. Download Qwen weights with Internet or attach a local SentenceTransformer-compatible model and set `EMBEDDING_MODEL` to its directory. No Generation or System-1 checkpoint is needed.

```python
import os
os.environ.update({
    'KAGGLE_OUTPUT_DIR': '/kaggle/working/m1-grounding-seed42',
    'TRAIN_DATA': '/kaggle/input/text2sql-data/train.jsonl',
    'RETRIEVAL_EVAL_DATA': '/kaggle/input/text2sql-data/retrieval_dev.jsonl',
    'DB_DIR': '/kaggle/input/text2sql-data/databases',
    'EMBEDDING_MODEL': 'Qwen/Qwen3-Embedding-0.6B',
})
os.environ.pop('CARDS_PATH', None)  # Let 01 extract database-scoped cards/graphs.
os.environ.pop('RESUME_FROM_CHECKPOINT', None)  # Fresh training.
```

Run 01; inspect held-out retrieval and index metadata. Save the entire Grounding stage, including cards, `schema_graphs.json`, column index and `final/`. If supplying `CARDS_PATH`, attach it explicitly; the automatic card/graph extraction branch is then bypassed, so preserve schema-graph provenance separately. Support Member 3 in verifying schema coverage at integration.

### Member 2 setup — Notebook 02, then 03

For 02 attach repository + SFT data; download or attach the authorized Llama-3.2-3B-Instruct base including tokenizer. SFT does not require SQLite or another trained agent.

```python
import os
os.environ.update({
    'KAGGLE_OUTPUT_DIR': '/kaggle/working/m2-sft-only-seed42',
    'SFT_DATA': '/kaggle/input/text2sql-data/sft_train.jsonl',
    'GENERATOR_MODEL': 'meta-llama/Llama-3.2-3B-Instruct',
    'LAMBDA_JEPA': '0',
})
os.environ.pop('RESUME_FROM_CHECKPOINT', None)
```

Run 02 and retain its returned `summary['checkpoint_path']`. For SFT+JEPA, restart the kernel, rerun common setup and the same member cell, then apply this cell before notebook setup:

```python
os.environ.update({
    'KAGGLE_OUTPUT_DIR': '/kaggle/working/m2-sft-jepa-seed42',
    'LAMBDA_JEPA': '0.1',
})
```

Keep data, seed, base model and budgets matched. This is fresh training, not continuation from the baseline: the JEPA objective differs. Inspect `dropped_overlength` and actual optimizer updates; short lengths can exclude complete reasoning+SQL examples.

For 03 attach repository + GRPO data/SQLite databases + one chosen complete SFT adapter artifact + access to the same base weights. Start a fresh kernel and common setup, then:

```python
import os
os.environ.update({
    'KAGGLE_OUTPUT_DIR': '/kaggle/working/m2-grpo-from-jepa-seed42',
    'SFT_CHECKPOINT': '/kaggle/input/sft-jepa-artifacts/train/sft_jepa_3b/checkpoint-100',
    'GRPO_DATA': '/kaggle/input/text2sql-data/grpo_train.jsonl',
    'DB_DIR': '/kaggle/input/text2sql-data/databases',
    'GROUP_SIZE': '2',
    'MAX_NEW_TOKENS': '128',
})
os.environ.pop('RESUME_FROM_CHECKPOINT', None)
```

Replace `checkpoint-100` with the actual complete SFT checkpoint. `SFT_CHECKPOINT` initializes a **new GRPO objective**; it is not GRPO resume. Use `BASE_MODEL_PATH` for a relocated identical local base. Retain the returned GRPO checkpoint and its frozen `reference/` adapter. Inspect four reward components, zero-variance groups, skipped examples and actual policy updates. Repeat from the SFT-only artifact in a separate root if comparing both initialization paths.

Notebook 03 does not switch to 7B automatically. An optional 7B experiment requires its own Notebook 02 SFT adapter with the corresponding base (for example Qwen2.5-Coder-7B), then 03 with that adapter. Never mix Llama 3B adapters with a 7B base. Phase names still contain `3b`; record actual model identity. 7B T4 fit remains unverified.

### Member 3 setup — Notebook 04 and evaluation ownership

Attach repository + three question-disjoint labeled candidate files; download ModernBERT-large or attach its full model/tokenizer directory and use that path in `DECISION_MODEL`. No Generation checkpoint is required for prebuilt labels. SQLite databases are additionally required if constructing labels from supplied candidate SQL.

```python
import os
os.environ.update({
    'KAGGLE_OUTPUT_DIR': '/kaggle/working/m3-system1-seed42',
    'DECISION_DATA': '/kaggle/input/decision-data/decision_train.jsonl',
    'CALIBRATION_DATA': '/kaggle/input/decision-data/decision_calibration.jsonl',
    'DECISION_TEST_DATA': '/kaggle/input/decision-data/decision_test.jsonl',
    'DECISION_MODEL': 'answerdotai/ModernBERT-large',
})
os.environ.pop('CANDIDATE_SOURCE', None)
os.environ.pop('RESUME_FROM_CHECKPOINT', None)
```

For automatic execution labeling/grouped splits, instead set:

```python
os.environ.update({
    'CANDIDATE_SOURCE': '/kaggle/input/candidates/sampled_candidates.jsonl',
    'DB_DIR': '/kaggle/input/text2sql-data/databases',
})
```

This branch overrides the three labeled paths with files it writes inside the stage output; use a new root to avoid exclusive-write collisions. Preserve these generated splits for resume and the final overlap audit. On resume, attach those exact split files using the first configuration and unset `CANDIDATE_SOURCE`. Both correct and incorrect training candidates are required. Keep every candidate for one question in one split. Save `final/` with `calibration_meta.json`; inspect temperature, held-out NLL calibration procedure and scaled/unscaled test ECE. Member 3 owns leakage auditing and the evaluation checklists below.

### Actual notebook controls and conservative T4 starting settings

| Interface | Used by |
|---|---|
| `KAGGLE_REPO_DIR`, `KAGGLE_DATA_DIR`, `KAGGLE_OUTPUT_DIR`, `PIPELINE_MODE`, `INSTALL_DEPS`, `TRAIN_DEVICE`, `SEED` | All five setup paths; output is `<root>/<train or smoke>/<phase>/` |
| `MAX_LENGTH`, `MAX_STEPS`, `SAVE_STEPS`, `RESUME_FROM_CHECKPOINT` | Training calls in 01–04; 05 reads these but does not pass them to evaluation |
| `GRAD_ACCUM` | 02/03 only; 01/04 use runner default accumulation eight |
| Stage data/model variables | Exactly those shown in each member configuration and the data-contract table |
| `BASE_MODEL_PATH` | Shared PEFT loader override for the identical base behind an adapter |
| Evaluation controls | Exactly those in the integration cell below; use `MAX_PROMPT_LENGTH`, not `MAX_LENGTH` |

Use the common `MAX_LENGTH=512`, `GRAD_ACCUM=8`, `MAX_STEPS=100`, `SAVE_STEPS=25` as an initial short real run; for a first GPU acceptance test lower `MAX_STEPS` to `2` and `SAVE_STEPS` to `1` in a separate root. Generation uses NF4 double quantization/FP16, rank-16 LoRA, checkpointing and a one-example microbatch. GRPO starts with group size two and 128 new tokens. These are conservative **starting points, not measured T4-safe guarantees**. Grounding and ModernBERT use full-model optimization, not Generation's QLoRA. SFT has a default three-epoch ceiling and GRPO two; `MAX_STEPS` is an optimizer-update cap, not a promise that it will be reached. Grounding/System-1 cycle data to their step budget. Epochs, learning rates, LoRA rank, microbatch size and hard-negative counts are not configurable through invented environment variables; changing them requires an explicit runner-call change in a separate experiment.

On OOM, restart the kernel to release stale models, run only one stage/process on the selected GPU, then lower `MAX_LENGTH` (for example 512 → 256) in a fresh run. Check SFT dropped examples and System-1 question/candidate budget errors rather than accepting invalid truncation. For 03 lower `MAX_NEW_TOKENS` (128 → 64); keep `GROUP_SIZE>=2`. For 05 lower `MAX_PROMPT_LENGTH`/`MAX_NEW_TOKENS`, start `NUM_CANDIDATES=1`, and keep Grounding/scoring on CPU. Candidate sampling is sequential; lowering count mainly reduces work. Accumulation is not microbatch size: lowering it generally does not solve activation OOM. Shorter sequences cannot solve model/optimizer residency OOM; if even loading fails, inspect actual CUDA/package compatibility and memory before attempting a larger model. Never replace the base model under an existing adapter. Changed sequence/objective contracts require a fresh run, not forced resume.

### Output paths and cross-account handoff

| Notebook | Directory under `KAGGLE_OUTPUT_DIR` in real mode | Downstream pointer |
|---|---|---|
| 01 | `train/grounding/` | `GROUNDING_CHECKPOINT=.../train/grounding/final` |
| 02 | `train/sft_jepa_3b/` for both ablations | `SFT_CHECKPOINT` or `GENERATOR_CHECKPOINT=.../checkpoint-N` |
| 03 | `train/grpo_3b/` | `GENERATOR_CHECKPOINT=.../checkpoint-N` |
| 04 | `train/system1/` | `DECISION_CHECKPOINT=.../train/system1/final` |
| 05 | `train/evaluation/` | `metrics.json`, `benchmark_summary.md`, `logs/evaluation_results.jsonl` |

1. Before the session ends, save a Kaggle notebook version retaining outputs or download the **entire stage directory**, including logs/environment, complete checkpoints, tokenizers, final exports and GRPO `reference/`. `/kaggle/working` is ephemeral; no automatic cloud transfer exists.
2. Publish the saved outputs as a private, versioned Kaggle Dataset accessible to the other two team accounts through normal sharing permissions. Record repository/data versions, configuration, source checkpoint identity and checksums. Share authorized adapters, not gated base weights without the required rights.
3. The recipient attaches that Dataset, inspects its actual directory layout, and updates paths accordingly; upload layouts may add a parent folder. Attach original training inputs and identical base weights separately. Test loading the shared artifact in a fresh session before marking the handoff accepted.
4. Preserve `COMPLETE` markers and `training_state.pt` for recovery; use only trusted team optimizer-state artifacts. `final/` is an inference export for Grounding/System-1, not their optimizer-resume checkpoint. Never point downstream model loading at a stage root when a final/checkpoint directory is required.

### Fresh training and recovery examples

Fresh training means a new output root with `RESUME_FROM_CHECKPOINT` unset. Downloading pretrained base weights is still fresh task training. Every SFT ablation starts fresh; GRPO initializes from SFT but starts its own optimizer unless resuming GRPO.

```python
# Fresh experiment (apply before setup in the relevant notebook).
os.environ['KAGGLE_OUTPUT_DIR'] = '/kaggle/working/new-experiment-seed42'
os.environ.pop('RESUME_FROM_CHECKPOINT', None)
```

```python
# Interrupted run, same session and same stage output root:
os.environ['RESUME_FROM_CHECKPOINT'] = 'latest'
# Rerun setup before the training call; latest searches only this stage directory.
```

```python
# New session: first rerun the same common/member data and objective configuration.
os.environ.update({
    'KAGGLE_OUTPUT_DIR': '/kaggle/working/m2-sft-jepa-recovered',
    'RESUME_FROM_CHECKPOINT': '/kaggle/input/sft-jepa-artifacts/train/sft_jepa_3b/checkpoint-100',
    'MAX_STEPS': '200',  # Total desired updates, not 200 additional updates.
})
```

Use the actual saved step and keep data bytes, model identity, seed and objective/length contracts unchanged. A new root is necessary when extending completed Grounding/System-1 runs because existing `final/` destinations are protected. SFT/GRPO also respect their epoch ceilings: a checkpoint at the final default epoch will not become a longer run just by raising `MAX_STEPS`. Resume GRPO from a GRPO checkpoint while retaining its original `SFT_CHECKPOINT` and base access; an SFT checkpoint is not a GRPO recovery state. Incomplete hidden staging directories are ignored. `latest` does not search attached datasets. Do not remove completion checks or overwrite immutable checkpoints to bypass a failure.

### Smoke and real-training acceptance checklists

For each member, first restart the kernel and run this configuration **instead of production configuration**, then execute the assigned notebook in order. It removes leftover data/model/checkpoint overrides. All three also run 05 in smoke mode.

```python
import os
from pathlib import Path
for key in (
    'TRAIN_DATA', 'RETRIEVAL_EVAL_DATA', 'DB_DIR', 'CARDS_PATH', 'SFT_DATA',
    'GRPO_DATA', 'DECISION_DATA', 'CALIBRATION_DATA', 'DECISION_TEST_DATA',
    'CANDIDATE_SOURCE', 'EVAL_DATA', 'SFT_CHECKPOINT', 'GENERATOR_CHECKPOINT',
    'DECISION_CHECKPOINT', 'GROUNDING_CHECKPOINT', 'BASE_MODEL_PATH',
    'EMBEDDING_MODEL', 'GENERATOR_MODEL', 'DECISION_MODEL',
    'TRAINING_DATA_PATHS', 'RESUME_FROM_CHECKPOINT',
):
    os.environ.pop(key, None)
os.environ.update({
    'KAGGLE_REPO_DIR': '/kaggle/input/optimize-mars-repo/Optimize-Mars-Sl',
    'PIPELINE_MODE': 'smoke', 'INSTALL_DEPS': '0', 'TRAIN_DEVICE': 'cpu',
    'KAGGLE_OUTPUT_DIR': '/kaggle/working/team-smoke-01',
})
os.environ['KAGGLE_DATA_DIR'] = str(Path(os.environ['KAGGLE_REPO_DIR']) / 'data/mini')
```

- [ ] Every cell completes in order; printed mode is smoke and output is under `smoke/<phase>/`.
- [ ] Training summaries say `trained: false`; evaluation says `is_benchmark: false`. Retain wiring logs; report no accuracy/convergence claim.
- [ ] Repeat local test commands below when validating code, and inspect skipped dependencies. Smoke success alone does not validate real model loading.

Then restart and restore real common/member configuration with a new root:

- [ ] Confirm GPU inventory/CUDA torch, authorized model access, package versions, dataset provenance and split disjointness. Record exact configuration and input versions.
- [ ] Run two optimizer updates with real data/model; verify finite losses, actual updates, complete checkpoints and measured GPU memory. Diagnose failures without switching to smoke.
- [ ] For 01 check retrieval metrics/index/schema graphs; for 02 check completion masking and overlength counts in logs; for 03 check reward variance and actual policy updates; for 04 check both labels, calibration temperature and test ECE.
- [ ] Reload the exported inference artifact and test checkpoint resume in a fresh session with unchanged contracts. Retain original artifacts and compare expected step progression.
- [ ] Only after acceptance, run the predeclared full budget in a distinct root (or valid resume), log actual updates/time/peak VRAM and held-out results, and save/share outputs. A completed run is not proof of convergence or a benchmark target.

### Joint integration and benchmark — Notebook 05

Member 3 operates the harness; Member 1 verifies schema retrieval/Steiner integration and Member 2 verifies the selected adapter and identical base. Attach repository, all three trained artifacts, base weights/access, held-out questions/SQLite databases, and all training manifests. Run common setup in a fresh GPU kernel, then:

```python
import os
os.environ.update({
    'KAGGLE_OUTPUT_DIR': '/kaggle/working/team-eval-grpo-k1-seed42',
    'GENERATOR_CHECKPOINT': '/kaggle/input/grpo-artifacts/train/grpo_3b/checkpoint-100',
    'GROUNDING_CHECKPOINT': '/kaggle/input/grounding-artifacts/train/grounding/final',
    'DECISION_CHECKPOINT': '/kaggle/input/system1-artifacts/train/system1/final',
    'EVAL_DATA': '/kaggle/input/text2sql-data/test.jsonl',
    'DB_DIR': '/kaggle/input/text2sql-data/databases',
    'NUM_CANDIDATES': '1', 'MAX_NEW_TOKENS': '128',
    'MAX_PROMPT_LENGTH': '512', 'RETRIEVAL_TOP_K': '25',
    'SCORER_DEVICE': 'cpu', 'GROUNDING_DEVICE': 'cpu',
    'GRAMMAR_MODE': 'posthoc',
    'TRAINING_DATA_PATHS': os.pathsep.join([
        '/kaggle/input/text2sql-data/train.jsonl',
        '/kaggle/input/text2sql-data/sft_train.jsonl',
        '/kaggle/input/text2sql-data/grpo_train.jsonl',
        '/kaggle/input/decision-data/decision_train.jsonl',
    ]),
})
os.environ.pop('RESUME_FROM_CHECKPOINT', None)  # 05 has no training-state resume.
```

1. Freeze split provenance and configurations before looking at final test results. Keep retrieval development, candidate calibration/test and benchmark test questions disjoint from task training; also keep final benchmark questions out of model selection/calibration. Include every actual training source in `TRAINING_DATA_PATHS`, including both SFT variants and generated candidate-training splits where different. Audit files need `db_id` and `question`; preserve these in SFT records or provide faithful manifests with those keys. The harness checks exact `(db_id, question)` pairs; independently normalize/deduplicate questions and audit database/split provenance. It cannot detect every paraphrase or missing manifest.
2. Use held-out `SQL` exclusively in scoring after prediction/selection. Do not copy it into schema/evidence, candidate prompts, grounding inputs, model selection or System-1 classifier inputs. Execution-derived candidate labels may use reference SQL only within the designated training/calibration datasets. Never build these datasets from benchmark questions. Gold must not select, repair or rerank a benchmark prediction.
3. Run a small, predefined held-out integration subset to verify fresh artifact loading and report paths, then a separate frozen benchmark run/root. Confirm `training_overlap_checked=true`, real benchmark status, trained Grounding identity and actual grammar mode. A lexical Grounding ablation must be explicitly labeled. No missing manifest or smoke report qualifies for acceptance.
4. Evaluate SFT-only, SFT+JEPA and GRPO with identical held-out data, seed, retrieval, generation budgets and validation settings. After measured memory acceptance, sweep `NUM_CANDIDATES` through 1, 4, 8, 16 with separate roots; predeclare any changes to prompt/new-token budgets (for example 2048/256) and apply them consistently. Optional native XGrammar runs form separate matched comparisons.
5. Save `train/evaluation/metrics.json`, `benchmark_summary.md`, `logs/evaluation_results.jsonl`, environment and configuration with artifact/data versions. Reports are exclusive-write; use a new root on rerun. Review EX, VSR, coverage, overall and accepted-only ECE, latency distributions and measured VRAM together. CPU retrieval/scoring latency counts. Keep abstentions/timeouts explicit and retain failure cases without test-driven retraining.
6. All three review the saved report and leakage audit. Publish only measured results; no smoke metric, historical scaffold accuracy, target ECE or inferred T4 memory bound is a benchmark result.

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

Historical implementation inventory, exact CPU environments, retained validation logs and unresolved acceptance checks are maintained in [KAGGLE_TRAINING_STATUS.md](KAGGLE_TRAINING_STATUS.md#3-recorded-test-results). Those recorded checks include notebook syntax/nbformat validation, in-memory Python compilation and dependency consistency. They are prior CPU evidence, not a new run performed while editing this guide. Native tiny-model CPU training/export/resume is tested; full Qwen3-Embedding-0.6B, Llama 3B/optional 7B and ModernBERT-large Kaggle GPU training, CUDA/NF4, native XGrammar, convergence, T4 fit and benchmark quality remain unverified.
