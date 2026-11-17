# Notebook 01 deterministic sampling (B5.2a)

`run_embedding_training` and its CLI accept `sampling_strategy` / `--sampling_strategy`:

- `sequential` (default): preserves the original modulo example order.
- `epoch_shuffle_v1`: visits every usable training example exactly once in each
  complete epoch, in a new deterministic permutation.

Notebook 01 exposes `SAMPLING_STRATEGY` (default `sequential`) and `GRAD_ACCUM`
(default `8`). For a fresh full-training experiment, set
`SAMPLING_STRATEGY=epoch_shuffle_v1`. These options do not enable training or
bypass the notebook's existing full-data preparation gate.

An epoch is `len(examples)` consumed positions in the constructed training list
(8149 for the Frozen BIRD training examples), independent of optimizer updates.
Validation examples and column cards are not part of this sampling list. Keep
input content, example construction, and input order identical on resume.

Each microbatch contains one example. With accumulation A, optimizer step s
(zero based) takes absolute positions `s*A` through `s*A+A-1`. A may be 1, 2, 4,
or 8 (other positive values also work). Updates crossing an epoch boundary take
the remaining examples of the old epoch, then the start of the next permutation;
there is no dropping, padding, or resetting accumulation. Duplicates across an
epoch boundary are possible, but never within one epoch.

Version 1 ranks example indices by the raw SHA-256 digest of the ASCII string
`epoch_shuffle_v1:{seed}:{epoch}:{index}`, breaking hash ties by index. Epoch and
index are zero based. This specifies the permutation independently of Python
hash randomization, RNG state, or Python's shuffle implementation. Only the
current epoch permutation is cached; the cache is disposable.

A checkpoint at completed optimizer step N resumes at absolute position `N*A`.
The usual optimizer, scaler, and Python/NumPy/Torch/CUDA RNG restoration remains
in force. Sampling uses no global RNG and needs no additional saved cursor.
AMP-skipped updates retain the same step and retry the same positions. Resume
never silently resets optimizer step or RNG. This guarantees example order;
bitwise GPU numerical equivalence still depends on the existing runtime.

New training contracts and sync compatibility fingerprints include
`sampling_strategy` and `sampling_version=1`. Changing strategy, version, seed,
or accumulation rejects resume. Old contracts without either sampling field
mean sequential v1 and remain loadable with their original configuration,
including old sync manifests (versions 1 and 2). They cannot initialize a
shuffled resume. Choose a fresh experiment ID/output root to change strategy.
No checkpoint migration, remote writes, dataset changes, retention, or validation
scheduling is involved.
