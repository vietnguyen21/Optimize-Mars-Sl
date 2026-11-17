# Safe local checkpoint retention (B5.2b)

Retention is **disabled by default**. `CHECKPOINT_LOCAL_KEEP=0` keeps every local
checkpoint, as before. A positive integer N keeps the latest N complete,
compatible checkpoints belonging to the active experiment and stage within the
configured output directory. N must be at least 1 when enabled. This setting is
operational: changing it does not change the training compatibility contract.

Recommended Notebook 01 full-training settings for a fresh experiment:

```python
os.environ['CHECKPOINT_SYNC_ENABLED'] = '1'
os.environ['CHECKPOINT_LOCAL_KEEP'] = '1'
os.environ['CHECKPOINT_SYNC_EVERY_N_SAVES'] = '1'
os.environ['CHECKPOINT_SYNC_FAILURE_POLICY'] = 'stop'
os.environ['CHECKPOINT_DISK_RESERVE_BYTES'] = str(1024**3)  # 1 GiB; tune to workload
# Existing private repo, unique experiment ID and output root are still required.
# Keep the same sampling and strict/legacy label settings when resuming.
```

Set these variables before initializing the checkpoint backend/training call.
Retention requires enabled HF sync. Read-only download backends never prune,
even if the variable is set. Sync intervals larger than one delay pruning until
a verified upload; unsynced saves can therefore accumulate. Recommend N=1 and
sync every save for Grounding, where one real checkpoint is about 6.67 GiB.

## Lifecycle and protection

1. Preflight disk space before creating the temporary checkpoint directory.
2. Serialize model, optimizer, scaler, scheduler when present, and RNG state to
   hidden staging. Build the manifest, write COMPLETE last, verify the manifest
   inventory and every SHA-256, then atomically rename to `checkpoint-N`.
3. Stage/upload immutable remote bytes, verify them, publish to the existing HF
   namespace, and read back the published manifest and every file at the pinned
   commit. No remote checkpoint is replaced or deleted.
4. Only after that verification, consider pruning local directories. The uploaded
   checkpoint must be the newest eligible local checkpoint. Keep it and the
   latest N; verify the entire candidate plan before deleting any older entry.

Pruning is limited to direct `checkpoint-<integer>` children of the original
output root with matching experiment ID, stage and compatible training contract.
Legacy checkpoints without manifests, incompatible/foreign checkpoints,
incomplete directories, and all hidden/interrupted staging directories remain.
Unsafe manifest paths, symlinks, mounted checkpoint subtrees, corrupt inventories,
changed roots and outside-root upload paths block pruning. The deletion uses
symlink-safe, directory-descriptor-relative `rmtree`. The latest checkpoint is
never a deletion candidate. Older eligible local saves need not each have an
individual successful upload: the newest checkpoint has just been verified both
locally and remotely and supersedes them for recovery.

Repeated sync is idempotent. With retention enabled, it rechecks remote bytes
even after a previous success; a failed later verification cannot authorize
deletion. Successful backup/pruning and `LocalRetentionRejected` events appear
in `checkpoint_sync.jsonl`. A retention rejection preserves the newest local
checkpoint and its verified remote backup; `stop` raises, while `warn` reports
the refusal and continues. Neither mode relaxes validation.

## Disk budget

At save preflight, let C be the estimated new checkpoint payload, E the estimated
final model export, and R the safety reserve. Existing checkpoints, cache blobs,
datasets and other outputs already consume space and reduce measured free bytes.

| Layout | Additional free bytes required |
| --- | --- |
| Output and HF cache share a filesystem | `2*C + E + R` on that filesystem |
| Separate filesystems | `C + E + R` on output, `C + R` on cache |
| Sync disabled | `C + E + R` on output |

C includes model/optimizer/extra tensor sizes, 20% serialization allowance and
1 MiB metadata allowance. E uses the same allowance for exported model tensors.
PEFT estimates count adapter/modules-to-save weights. Set
`CHECKPOINT_FINAL_EXPORT_RESERVE_BYTES` (default 0) to a larger export floor when
custom export artifacts need more space: the effective E is the maximum of the
estimate and this value. Negative budgets are rejected.
`CHECKPOINT_DISK_RESERVE_BYTES` retains its existing default of 256 MiB; 1 GiB is
recommended here. Resume downloads reserve a full checkpoint for E because model
tensors have not yet been loaded, which is conservative.

For C=6.67 GiB, a shared filesystem needs **13.34 GiB + E + R of free space**
before the next save, in addition to the retained checkpoint already on disk.
Retention N=1 cannot remove this overlap requirement. A distinct cache directory
on the same filesystem does not count as separate storage (`st_dev` determines
the budget). Move `HF_HUB_CACHE` to a genuinely separate larger filesystem before
backend initialization when available; keep the working output in writable storage.

These are estimates, not a disk reservation. Concurrent processes, filesystem
quotas, unusual serialization/export overhead, and other generated artifacts
can exhaust space after preflight. Use one writer per output root/experiment;
external concurrent mutation is unsupported. HF cache pruning is deliberately
outside this policy: cached historical blobs may accumulate, and hidden partial
directories are preserved. Preflight fails rather than deleting recovery data
to manufacture headroom. Capacity must accommodate the real workload.

## Recovering after failed upload or verification

The newly completed local checkpoint and all older local checkpoints remain
when upload or remote verification fails. With `stop`, restore authentication,
network and disk capacity, then resume explicitly from that verified local
checkpoint using the same experiment, data, base model, sampling, labels and
optimizer configuration. Retry the backup before continuing; for a standalone
backup retry, use the matching backend's `sync(checkpoint_path)` without training.
Immutable staging/publishing makes a retry safe even if a commit response was lost.
After successful verification, eligible older checkpoints may be pruned.

With `warn`, training can continue, but unsynced local checkpoints accumulate and
the next save may fail disk preflight. Read `checkpoint_sync.jsonl` and resolve
the backup failure; do not delete the only recoverable checkpoint for space.
If the session is lost before successful backup, recover only from the latest
previously verified remote checkpoint. No incompatible fallback or automatic
restart from step zero is introduced. Optimizer/scaler/RNG restoration,
`epoch_shuffle_v1` and strict/legacy Grounding contracts remain unchanged.

## CPU validation

Run each suite in its own process from the repository root:

```bash
for suite in embedding_sampling sql_grounding_audit checkpoint_sync runtime checkpoint_retention kaggle_environment
do
  /tmp/b52b-validation/bin/python -m unittest discover -s tests -p "test_${suite}.py"
done
```

Recorded B5.2b results: embedding sampling 6/6, Grounding Audit 34/34, checkpoint
sync 27/27, runtime 9/9, retention 27/27, Kaggle environment 22/22: **125 passed,
0 failed, 0 skipped**. Notebook 01 code-cell/Python syntax and `git diff --check`
also passed. The isolated `/tmp/b52b-validation` environment used CPython 3.14,
CPU-only torch 2.15.0+cpu, NumPy 2.5.5 and pinned SQLGlot 26.16.4. These are local
test tools, not changes to the project's production/CUDA dependency profiles.
No pretrained model downloads, actual HF checkpoint access, GPU/full training,
or user checkpoint deletion was performed. Only temporary fixture checkpoints
were pruned, against an in-memory fake Hub.
