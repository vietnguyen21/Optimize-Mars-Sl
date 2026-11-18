# M0 Kaggle environment qualification

`KAGGLE_PROFILE=LEGACY` is the default and retains the original profiles, bootstrap and experiment paths. `KAGGLE_PROFILE=MODERN` explicitly selects the new core or grounding candidate. Notebook 01 stops after environment preflight in MODERN mode, before checkpoint initialization or dataset preparation. Modern stages 02–05 are unavailable. M0 changes environment wiring only.

## Version matrix

| Component | Core candidate | Grounding candidate |
| --- | --- | --- |
| Image runtime | CPython 3.13.16 x86-64 on T4 | same |
| Protected image torch stack | 2.11.0+cu130 / torchvision 0.26.0+cu130 / torchaudio 2.11.0+cu130 / Triton 3.6.0 | same |
| Transformers / Hub / tokenizers | 5.20.2 / 1.33.0 / 0.23.3 | same |
| Accelerate / safetensors | 1.15.0 / 0.8.0 | same |
| NetworkX / SQLGlot / Pydantic | 3.7 / 26.16.4 / 2.14.1 | same |
| NumPy / packaging | Resolve from the actual image under the root constraints | same |
| Sentence Transformers | omitted | 5.3.0 |

The candidate manifest records SHA-256 for the checked-in resolver inputs. It has no qualification record or transitive wheel hashes yet. The observed Kaggle runtime has the target Python and torch versions. Its published image tag or digest is still unverified. Capture the running build before reviewing either lock. [Kaggle image releases](https://github.com/Kaggle/docker-python/releases)

## Capture, review, and use a lock

Run on a cold Kaggle T4 session from the migration checkout. The image interpreter creates a clean venv with no system site-packages. A selective import directory links the protected CUDA distributions and their dependency closure back to the image using installed wheel records. Application wheels install into the venv. The base image keeps its SQLGlot and Ibis installations. An attached Git-free source snapshot requires `KAGGLE_REPO_COMMIT` set to its verified commit. When the runtime exposes a trusted OCI digest, pass it as `KAGGLE_IMAGE_DIGEST=sha256:…`. The tool records Kaggle build metadata when available and explicitly records a missing digest.

```bash
python -I -B scripts/check_kaggle_environment.py --profile MODERN \
  --create-modern-venv /kaggle/working/m0-grounding-venv
/kaggle/working/m0-grounding-venv/bin/python -I -B scripts/check_kaggle_environment.py \
  --profile MODERN --stage grounding --resolve-modern-lock \
  --output /kaggle/working/m0-grounding-lock
```

Repeat with `--stage core` and a separate venv and output directory. This only resolves. The command runs a binary-only resolver dry-run under protected image constraints, verifies the planned active graph, and resolves exact wheel filenames and SHA-256 for the nonprotected closure. Review `modern-resolution.json` and `modern.lock.json`. Confirm the runtime fingerprint, each wheel and constraints, introduced conflicts and active graph. If the resolver lacks a wheel hash or proposes a protected change, resolve the discrepancy before approval. Store reviewed locks by stage alongside the image fingerprint. The lock represents preinstalled image packages by their version and metadata, with a separate native-library inventory fingerprint. An unchanged installed application package is checked by version and dependencies. Its original installed bytes are image-provided.

In a fresh kernel on the reviewed image, set `KAGGLE_PROFILE=MODERN`, `KAGGLE_MODERN_LOCK` to the reviewed stage-specific lock and `KAGGLE_MODERN_LOCK_SHA256` to its approved hash. For Notebook 01 set `KAGGLE_MODERN_PYTHON` to the absolute venv Python path and `INSTALL_DEPS=1` to enable installation. Notebook setup invokes separate isolated install and preflight processes. For the CLI use `--install-modern-lock` in the venv. The installer checks the projected closure and installed package ownership of loaded modules, verifies an exact hash-constrained dry-run and installs only necessary application wheels with dependency expansion disabled. After successful bootstrap, verify from an isolated process with the same environment.

```bash
/kaggle/working/m0-grounding-venv/bin/python -I -B scripts/check_kaggle_environment.py \
  --profile MODERN --stage grounding --device cuda:0 --output /kaggle/working/m0-isolated
```

When both GPUs are visible, repeat with `cuda:1` and another output directory. Expect locked imports, synchronized CUDA backward with finite gradients and writable disk checks. No pretrained model or dataset is required. Notebook 01 modern mode uses a separate `modern-environment/grounding` output root. Stop and recapture for review if the runtime fingerprint differs. This includes changes in visible GPU topology or driver.

## Report schema

All modern JSON reports use `schema_version: 1`, `profile`, `stage`, `runtime` and `repository`. The runtime contains Python/ABI, machine/libc, image digest or identity gap, Kaggle build commit/date, CUDA runtime and driver versions, GPU capabilities, torch build and architectures, protected distribution versions, and a fingerprint of the protected distributions' native library size/mtime inventory. Managed installation targets application wheels only. System CUDA packages and shared libraries remain image-owned.

- `modern-resolution.json` records the image-wide pip diagnostics as `image_global_pip`, the isolated interpreter's global pip diagnostics before resolution, and projected dependency conflicts.
- `modern.lock.json` records sorted roots, the exact runtime, package versions and Requires-Dist, and application wheel filenames and SHA-256.
- `modern-bootstrap.json` / `modern-preflight.json` record the effective lock and its SHA-256, projected conflicts when installing, observed conflicts and global pip stdout, stderr and return codes before and after. URL tokens are redacted. Active or introduced conflicts stop the operation. Recognized unrelated preinstalled conflicts remain visible. Preflight subsequently checks loaded imports, kernel and disk.
- Conflict entries have `[requiring-package, requirement, installed-version]` shape. `fatal_active` includes active-root and transitive failures. `introduced` includes newly broken edges even when outside the active graph. `unrelated_preinstalled` includes surviving baseline conflicts whose requiring package is outside the active graph.

## M0.2 root cause and decision

Native SQLGlot 25.20.2 lacks the strict qualification argument used by `sql_grounding.py`. In a local comparison using the actual 25.20.2 wheel, the SQL Audit and Grounding/System-1 suites produced 25 errors across 50 tests. The errors include qualification API failures and the expected rejection by the exact-version audit guard. Retain 26.16.4 and the historical guard. No compatibility adapter, label semantics change, or Frozen B3 write was needed. The observed Pydantic downgrade caused four unrelated conflicts on the Kaggle image, and installing SQLGlot 26.16.4 globally conflicted with Ibis's <25.21 constraint. The native Pydantic and NetworkX versions passed the tests below. [SQLGlot 26.16.4 qualifier](https://github.com/tobymao/sqlglot/blob/v26.16.4/sqlglot/optimizer/qualify.py#L16)

Both tested Sentence Transformers versions satisfied the local metadata closure. The 5.3.0 tiny native grounding train/export/reload/resume fixture passed. On 5.7.0, `tokenize` warned that it was deprecated and returned a string among the features, causing the current loop's tensor transfer to fail. Keep 5.3.0 in the isolated grounding profile. A separate exploratory ModernBERT fixture failed on both stacks because Transformers 5.20.2 rejects the existing `reference_compile` argument. Modern System-1 stays out of scope.

### Local test results

Python 3.14.10, torch 2.15.0+cpu, Pydantic 2.14.1, NetworkX 3.7, SQLGlot 26.16.4, Transformers 5.20.2, Sentence Transformers 5.3.0. Tests ran through unittest in the isolated CPU venv with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. All requested tests ran.

| unittest target under `tests.` | Passed | Failures / errors / skips |
| --- | ---: | --- |
| `test_sql_grounding_audit` | 34 | 0 / 0 / 0 |
| `test_grounding_system1_pipeline` | 16 | 0 / 0 / 0 |
| `test_grounding` | 25 | 0 / 0 / 0 |
| `test_grammar_and_validation` | 30 | 0 / 0 / 0 |
| `test_modern_environment` | 12 | 0 / 0 / 0 |
| `test_kaggle_environment` | 22 | 0 / 0 / 0 |
| `test_notebooks` | 4 | 0 / 0 / 0 |
| `test_hf_grounding_system1.NativeGroundingSystem1.test_native_sentence_transformer_train_export_resume` | 1 | 0 / 0 / 0 |

The SQLite test computes a mean over 24 queries with a 10 ms threshold. In 30 batches pinned to CPU 0 with fresh read-only SQLite connections per batch, the local mean had a 0.0211 ms minimum, 0.0220 ms median and 0.0337 ms maximum using SQLite 3.54.1. No batch crossed the threshold. The reported 17.40 ms Kaggle result did not reproduce locally. Repeat under comparable SQLite/build, affinity, caching and host-load conditions on Kaggle. The test and threshold are unchanged.

### Resolver and lock status

Real binary-only pip dry-runs in the local CPU bridge resolved 47 packages with 30 hashed application wheels for core, and 54 packages with 37 hashed application wheels for grounding. Each had zero fatal active and zero introduced conflicts, with no planned protected replacement. The grounding wheels installed via the hash-checked path, isolated `pip check` passed, and a new interpreter imported the locked stack using torch from the source environment. The source environment's separate `pip check` diagnostic recorded existing conflicts. Both resolutions selected SQLGlot wheel `sqlglot-26.16.4-py3-none-any.whl` with SHA-256 `dbff0ace2e21c9169182d6852b8f6eb56d86fc6477354abcb5bda7ab93292975`.

The CPU rehearsal bypassed only the Kaggle runtime eligibility predicate. It still checked bridge fingerprint consistency and wheel hashes. Those locks are local evidence and cannot qualify Kaggle. Re-run both resolutions and locked installs on the actual T4 image, preserve image-wide diagnostics, compare the protected runtime fingerprint and verify synchronized kernels in fresh venv processes. Remaining risks include NVIDIA library layout, any protected dependency version incompatible with the application closure, and the unresolved Kaggle SQLite timing outlier. No pretrained models, private checkpoints, or Frozen B3 data were used.
