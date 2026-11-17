# Notebook 01 sessions and validation (B5.2c)

All new behavior is opt-in. No full-training settings are enabled by this change.
The Frozen B3 dataset, model identity, labels, mining, loss, sampling algorithm,
checkpoint format, HF namespaces and retention defaults are unchanged.

## Controls

| Trainer/CLI argument | Notebook environment | Default / meaning |
| --- | --- | --- |
| `global_max_steps` / `--global_max_steps` | `GLOBAL_MAX_STEPS` | Unset: legacy mode. Otherwise fixed absolute optimizer-step target for a fresh experiment. |
| `session_stop_step` / `--session_stop_step` | `SESSION_STOP_STEP` | Unset: global target. Otherwise absolute completed optimizer step to stop this session. Requires a global target. |
| `eval_steps` / `--eval_steps` | `EVAL_STEPS` | `0`: periodic validation disabled. Positive: validate at multiples of this many completed optimizer updates, and at the global/legacy target. |
| `evaluation_kind` / `--evaluation_kind` | `EVALUATION_KIND` | Required for periodic evaluation and for evaluation in session mode: `internal_validation` or `external_held_out`. |
| `eval_path` / `--eval_path` | `RETRIEVAL_EVAL_DATA` | Explicit held-out input. Existing completion-only evaluation remains available when periodic validation is disabled. |

Notebook 01 labels its freshly prepared full-data internal split automatically.
When attaching the existing Frozen B3 validation split, explicitly set
`EVALUATION_KIND=internal_validation`. Official external held-out evaluation must
be labeled `external_held_out`; internal BIRD Train splits are not external results.
The existing preparation-only/full-training gate is preserved.

## Session contract and resume

Without `global_max_steps`, `max_steps` retains its original absolute optimizer
update semantics and reaching it exports `final/` and the offline column index.
Legacy runs can still increase `max_steps` when resuming into a suitable output
root. Periodic validation remains disabled by default.

With `global_max_steps`, that target replaces `max_steps` for the entire new
experiment. Require `1 <= session_stop_step <= global_max_steps`; the session
boundary may change between invocations, but the target may not. This mode
requires writable HF sync. Every checkpoint save, including evaluation saves
and session boundaries, must have a successful verified remote backup in the
current invocation before training continues or returns success. A successful
save/upload already verified in this invocation is reused; a new invocation
verifies again. This requirement overrides a larger sync interval and also
stops on backup failure under `warn` policy. A failed backup leaves local data
intact and never triggers a restart from zero.

At an intermediate boundary the runner saves a complete checkpoint (unless the
same completed step was just saved or loaded), verifies its backup, logs
`session_stopped`, writes `session_summary.json`, and returns `status=session_stopped`,
the absolute `step`, `global_max_steps`, and `checkpoint_path`. `model_path` and
`index` are null. It creates no final export or column index. The session summary
is replaceable operational metadata, not the optimizer checkpoint.

At the actual target, backup verification and required validation must succeed
before exporting `final/`, column metadata/index and `training_summary.json` with
`status=completed`. An exclusive `.final-export.claim` prevents a second export
into that root. Existing final/index artifacts are rejected. A failed partial
export/claim is preserved; recover from the target checkpoint into a fresh output
root rather than overwriting it. A checkpoint at a save/eval/session intersection
is created only once. A resume already at the target can export without another
optimizer update or duplicate checkpoint.

New checkpoint training contracts add `training_plan_version=1` and the fixed
`global_max_steps`. Evaluation-enabled contracts also pin validation content
SHA-256, classification, interval, top-k=10 and validation version. Sync
fingerprints include the same evaluation identity. Changes to target, evaluation
dataset, classification, interval, seed, accumulation, sampling, grounding labels
or existing objective/base identity reject resume. Session boundaries, output
location and save frequency are operational controls. Resume steps greater than
the proposed boundary are rejected. A session-mode output containing checkpoints
requires explicit resume; it cannot silently be reused as a fresh run.

Use the same experiment ID and exact prepared train/cards/validation files in
each session. For cross-session recovery select `remote` (or an explicit compatible
local checkpoint); do not set `none` to extend a run. Remote lookup failures or
incompatible artifacts do not fall back to step zero. Optimizer, scaler and
Python/NumPy/Torch/CUDA RNG use the existing restoration path. No checkpoint
migration or mutable sampler cursor is introduced. Old sequential and
`epoch_shuffle_v1` checkpoints remain usable under their original configuration;
opting into a fixed plan or periodic validation requires a fresh experiment.

## Epoch accounting

An epoch contains 8149 training examples. With accumulation A, optimizer step s
(zero based) consumes positions `s*A` through `s*A+A-1` of the continuous stream.
Resume at N completed updates starts at `N*A`. Accumulation continues across
epoch boundaries, as specified in [deterministic sampling](EMBEDDING_SAMPLING.md).

| Epoch budget at A=8 | Absolute update ceiling | Consumed examples | Into next epoch |
| --- | ---: | ---: | ---: |
| 1 | 1019 | 8152 | 3 |
| 2 | 2038 | 16304 | 6 |
| 3 | 3056 | 24448 | 1 |

These are `ceil(epochs * 8149 / 8)`, not a sum of rounded epoch lengths. For
example, a *proposed* three-epoch plan could keep `GLOBAL_MAX_STEPS=3056` across
sessions stopping at 1000, 2000 and 3056. These illustrate absolute boundaries;
they are not approved final experiment settings and nothing runs automatically.

## Validation and logging

Validation uses a separately built held-out example list. Normalized
`(db_id, question)` overlap with training is rejected before loading the model.
Optimizer updates and hard-negative mining only read training examples. The
validation label list never enters training mining or the sampler.

Evaluation runs in eval mode with gradients disabled. A guard restores each
module's previous mode and all Python/NumPy/Torch/CUDA RNG states on success or
failure. It receives no optimizer and does not advance optimizer/scaler state or
the position-addressed sampler. Retrieval ranks columns within the question's
database. Recall@10 is micro recall across positive columns; Hit@10 is the
fraction of validation questions retrieving at least one positive.

Successful metrics are appended to the existing `metrics.jsonl`, for example:

```json
{"event":"retrieval_validation","phase":"embedding","step":100,"evaluation_kind":"internal_validation","eval_sha256":"<sha256>","k":10,"status":"ok","recall_at_10":0.72,"hit_at_10":0.83,"questions":906,"time_unix":1234567890}
```

The numbers above are illustrative, not measured B3 results. A failure uses
`status=failed` and the exception class in `error`, with no invented metric values.
Metrics must be finite and in [0,1]. `validation_progress` in checkpoint `extra`
stores `last_evaluation_step` and `last_validation`. Every periodic evaluation
forces a checkpoint so normal successful resume does not evaluate the same
completed step again. At non-aligned intermediate session stops, validation
waits for the next scheduled step; the actual target always evaluates. Final
export reuses successful target metrics rather than evaluating twice.

On validation failure, training stops after saving the completed update and,
in session mode, verifying its remote backup. Its failure marker is checkpointed.
Fix the cause and explicitly resume that checkpoint: an intermediate failed
validation is not automatically retried at the same step; evaluation continues
at the next scheduled step. At a failed final validation, resume at the target
retries validation without retraining so export can recover. The original failed
checkpoint stays immutable; successful retry metrics are in the final summary/log.
Do not alter pinned validation data to make an incompatible resume pass.

An uncheckpointed process crash can lose the latest validation watermark just
as it can lose uncheckpointed updates. Replaying those updates may replay that
evaluation; this is not repeated evaluation of a successfully resumed checkpoint.
No claim of distributed exactly-once logging is made.

## Storage and runtime planning

The initial proposed values are `sampling_strategy=epoch_shuffle_v1`, accumulation
8, saves every 50 updates, evaluation every 100 updates, local keep=1, sync every
save, and failure policy `stop`. They are planning values, not experiment approval.

Use [B5.2b retention and disk budgeting](CHECKPOINT_RETENTION.md). The supplied
Kaggle storage snapshot reports 19.50 GiB free in working storage and 966.89 GiB
in HF cache on **separate filesystems**. Preflight still measures current space;
these are not hard-coded capacities. On separate filesystems, required free
space is `C + E + R` at output and `C + R` at cache; existing checkpoints already
reduce free space. C is the conservative estimated new checkpoint size, E the
reserved final export, R the safety reserve. One real checkpoint measured about
6.67 GiB. Final export also rechecks `E + R` before creating its exclusive claim,
including target-only resumes that do not save another checkpoint.
Keep the export budget even at temporary stops. HF cache and hidden
interrupted staging are never automatically deleted.

Each evaluation encodes all 3539 column cards (batch 4) and all 906 held-out
questions (one at a time), with CPU NumPy retrieval ranking. It adds model forward
passes, GPU activation/embedding memory and CPU vector storage while optimizer
state remains resident. It does not reuse potentially stale training embeddings.
Every periodic eval also incurs checkpoint serialization and synchronous remote
verification; align eval/save intervals to avoid additional saves (100/50 does).
RNG isolation does not make evaluation free. Measure duration and peak VRAM on
the actual Kaggle GPU before approving a final interval; this task runs only CPU
fixtures. Hashing, read-back and cache growth add I/O/storage cost. Budgets are
estimates, not reservations; use one writer per output root. GPU numerical
reproducibility still depends on the existing runtime and hardware.

## CPU verification

From the repository root, each suite was run in a separate process:

```bash
for suite in embedding_sessions embedding_sampling sql_grounding_audit checkpoint_sync runtime checkpoint_retention kaggle_environment notebooks grounding_system1_pipeline
do
  /tmp/b52b-validation/bin/python -m unittest discover -s tests -p "test_${suite}.py"
done
```

| Suite | Passed | Failed / skipped |
| --- | ---: | ---: |
| embedding_sessions | 26 | 0 / 0 |
| embedding_sampling | 6 | 0 / 0 |
| sql_grounding_audit | 34 | 0 / 0 |
| checkpoint_sync | 27 | 0 / 0 |
| runtime | 9 | 0 / 0 |
| checkpoint_retention | 27 | 0 / 0 |
| kaggle_environment | 22 | 0 / 0 |
| notebooks | 4 | 0 / 0 |
| grounding_system1_pipeline | 16 | 0 / 0 |
| **Total** | **171** | **0 / 0** |

The isolated test environment uses CPython 3.14, torch 2.15.0+cpu, NumPy 2.5.5
and pinned SQLGlot 26.16.4. No project/CUDA dependency profile was changed.
New integration tests use a tiny local module with dropout and an in-memory Hub.
They verify cross-session equality of parameters, optimizer tensors, RNG and the
sample stream at accumulation 8; absolute validation scheduling and checkpointed
watermarks; strict and legacy labels; failure preservation, retention, final
export recovery and immutable export/checkpoint handling. No real HF checkpoint,
pretrained model download, Frozen B3 modification or full/GPU training is involved.
