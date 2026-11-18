# M0 Kaggle environment qualification

`KAGGLE_PROFILE=LEGACY` is the default and retains the original profiles, bootstrap and experiment paths. `KAGGLE_PROFILE=MODERN` explicitly selects the new core or grounding candidate. Notebook 01 stops after environment preflight in MODERN mode, before checkpoint initialization or dataset preparation. Modern stages 02–05 are unavailable. M0 changes environment wiring only.

## Version matrix

| Component | Core candidate | Grounding candidate |
| --- | --- | --- |
| Image runtime | CPython 3.13.16 x86-64 on T4 | same |
| Protected image torch stack | 2.11.0+cu130 / torchvision 0.26.0+cu130 / torchaudio 2.11.0+cu130 / Triton 3.6.0 | same |
| Transformers / Hub / tokenizers | 5.20.2 / 1.33.0 / 0.23.3 | same |
| Accelerate / safetensors | 1.15.0 / 0.8.0 | same |
| NetworkX / SQLGlot / Pydantic | 3.4.2 / 26.16.4 / 2.11.3 | same |
| NumPy / packaging | Resolve from the actual image under the root constraints | same |
| Sentence Transformers | omitted | 5.3.0 |

The candidate manifest records SHA-256 for the checked-in resolver inputs. It has no qualification record or transitive wheel hashes yet. The current host has Python 3.14, lacks the Kaggle CUDA stack and has no running Kaggle session. Published v173/v174 release information does not identify the deployed runtime. Capture the actual running build before reviewing either lock. [Kaggle image releases](https://github.com/Kaggle/docker-python/releases)

## Capture, review, and use a lock

Run on a cold Kaggle T4 session from the migration checkout. An attached Git-free source snapshot requires `KAGGLE_REPO_COMMIT` set to its verified commit. When the runtime exposes a trusted OCI digest, pass it as `KAGGLE_IMAGE_DIGEST=sha256:…`. The tool records Kaggle build metadata when available and explicitly records a missing digest.

```bash
python scripts/check_kaggle_environment.py --profile MODERN --stage grounding \
  --resolve-modern-lock --output /kaggle/working/m0-grounding-lock
```

Repeat with `--stage core` and a separate output directory. This only resolves. The command runs a binary-only resolver dry-run under protected image constraints, verifies the planned active graph, and resolves exact wheel filenames and SHA-256 for the nonprotected closure. Review `modern-resolution.json` and `modern.lock.json`. Confirm the runtime fingerprint, each wheel and constraints, introduced conflicts and active graph. If the resolver lacks a wheel hash or proposes a protected change, resolve the discrepancy before approval. Store reviewed locks by stage alongside the image fingerprint. The lock represents preinstalled image packages by their version and metadata, with a separate native-library inventory fingerprint. An unchanged installed application package is checked by version and dependencies. Its original installed bytes are image-provided.

In a fresh kernel on the reviewed image, set `KAGGLE_PROFILE=MODERN`, `KAGGLE_MODERN_LOCK` to the reviewed stage-specific lock and `KAGGLE_MODERN_LOCK_SHA256` to its approved hash. To allow installation, set `INSTALL_DEPS=1` before Notebook 01 setup. The installer checks the projected closure and installed package ownership of loaded modules, verifies an exact hash-constrained dry-run and installs only necessary application wheels with dependency expansion disabled. After successful bootstrap, verify from an isolated process with the same environment.

```bash
python -I scripts/check_kaggle_environment.py --profile MODERN --stage grounding \
  --device cuda:0 --output /kaggle/working/m0-isolated
```

When both GPUs are visible, repeat with `cuda:1` and another output directory. Expect locked imports, synchronized CUDA backward with finite gradients and writable disk checks. No pretrained model or dataset is required. Notebook 01 modern mode uses a separate `modern-environment/grounding` output root. Stop and recapture for review if the runtime fingerprint differs. This includes changes in visible GPU topology or driver.

## Report schema

All modern JSON reports use `schema_version: 1`, `profile`, `stage`, `runtime` and `repository`. The runtime contains Python/ABI, machine/libc, image digest or identity gap, Kaggle build commit/date, CUDA runtime and driver versions, GPU capabilities, torch build and architectures, protected distribution versions, and a fingerprint of the protected distributions' native library size/mtime inventory. Managed installation targets application wheels only. System CUDA packages and shared libraries remain image-owned.

- `modern-resolution.json` records full global pip diagnostics before resolution and projected dependency conflicts.
- `modern.lock.json` records sorted roots, the exact runtime, package versions and Requires-Dist, and application wheel filenames and SHA-256.
- `modern-bootstrap.json` / `modern-preflight.json` record the effective lock and its SHA-256, projected conflicts when installing, observed conflicts and global pip stdout, stderr and return codes before and after. URL tokens are redacted. Active or introduced conflicts stop the operation. Recognized unrelated preinstalled conflicts remain visible. Preflight subsequently checks loaded imports, kernel and disk.
- Conflict entries have `[requiring-package, requirement, installed-version]` shape. `fatal_active` includes active-root and transitive failures. `introduced` includes newly broken edges even when outside the active graph. `unrelated_preinstalled` includes surviving baseline conflicts whose requiring package is outside the active graph.

## Local evidence and remaining gates

The dependency-light suite ran 363 tests, with 278 passing and 85 dependency-required skips. The 9 M0 tests passed, covering profile isolation, closure markers/extras, conflict classification, CUDA distribution protection, lock hash and image drift, warm-module rejection, smoke behavior and wheel hash requirements. The modern CLI offline smoke check passed. Live resolution, isolated real HF imports and actual T4 kernel checks remain pending. The audit and this phase do not certify modern training readiness.
