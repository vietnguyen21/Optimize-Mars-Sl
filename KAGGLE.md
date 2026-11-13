# Official Kaggle training guide — three-person research team

## Status and evidence

These pipelines replace the earlier dry-run scaffolds. Historical mini-data accuracy and calibration statements in PROJECT.md/HANDOVER.md are **not pretrained model benchmark evidence**. Smoke outputs now carry `mode: smoke`, `trained: false` (training runners), and `is_benchmark: false` (evaluation). Real mode never silently substitutes synthetic training or gold-based generation. No Kaggle execution, full-model convergence, T4 memory bound, or benchmark target has been verified locally. See the validation section below for actual checks.

## Bootstrap sau environment audit

Dùng kernel mới. Package `src.training` nay lazy-load API nên import `src.training.kaggle` không kéo Torch/Transformers vào trước bước cài đặt. Các notebook 01–05 chọn profile theo stage:

| Stage | Profile |
|---|---|
| Grounding | `requirements/kaggle-grounding.txt` |
| SFT/GRPO | `requirements/kaggle-generation.txt` |
| System-1 | `requirements/kaggle-core.txt` |
| Evaluation | `requirements/kaggle-evaluation.txt` |
| Local validation tooling | `requirements/kaggle-dev.txt` |

`requirements.txt`, `requirements-kaggle.txt` và entry point cũ trong docs cùng trỏ về profile chung; không cài TRL/Datasets/Unsloth/xFormers/Flash Attention khi code không sử dụng. XGrammar vẫn optional và phải chọn bản đã kiểm chứng riêng.

Với `INSTALL_DEPS=1`, bootstrap yêu cầu CPython 3.10–3.12 cho các profile chung; riêng Notebook 01 (`grounding`) cho phép CPython thường 3.13. Image phải có CUDA torch >=2.5,<3; với Python 3.13 cần torch >=2.6. Không hỗ trợ free-threaded Python. Nó constrain phiên bản torch/torchvision/torchaudio/triton preinstalled, chạy pip dry-run với binary wheels, từ chối kế hoạch thay CUDA stack hoặc thay package đã import, rồi mới install và `pip check`. Giữ `dependency_snapshot.json` với baseline, profile và version thực tế. Không cài lại CUDA/PyTorch để né resolver error. Các profile pin application API nhưng **chưa là transitive lock hoặc certified Kaggle image**; giữ cùng image và snapshot khi chuyển account.

Nếu warm kernel có module khác bản trên disk hoặc cần thay package đã load: dừng, restart kernel rồi chạy lại từ cấu hình. Không restart hoặc `os._exit` giữa một job Save & Run All rồi mong job tự tiếp tục. Nếu package được image preload không tương thích, chọn image/profile phù hợp trước khi run. `INSTALL_DEPS=0` không tắt preflight: notebook vẫn kiểm tra version, import, write/disk và CUDA kernel; Generation/Evaluation kiểm tra NF4 primitive trước tải model. `PIPELINE_MODE=smoke` bỏ qua install/Hub/GPU và không được coi là kiểm chứng dependency native.

Các guard bổ sung: ModernBERT luôn `reference_compile=False` trong training/scoring để không tự dùng Triton trên P100. Grounding/System-1 chỉ tăng step khi optimizer update thực; `AMP_MAX_CONSECUTIVE_SKIPS` mặc định 8, quá ngưỡng sẽ dừng và giữ checkpoint trước đó. CUDA checkpoint mới ghi active device; khi đổi GPU topology, chỉ map RNG training GPU sang thiết bị được chọn và cảnh báo không bảo đảm bitwise reproducibility giữa hardware khác nhau.

Kiểm tra nhẹ, không download model hoặc ghi Hub:

```bash
python scripts/check_kaggle_environment.py --smoke
# Chỉ chạy trên GPU Kaggle đã bootstrap profile tương ứng:
python scripts/check_kaggle_environment.py --stage grounding --device cuda:0
python scripts/check_kaggle_environment.py --stage sft_jepa_3b --device cuda:0 --quantization
python -m unittest discover -s tests -v
python scripts/run_smoke_tests.py
```

Pretrained/full-size training và Hub thật vẫn cần acceptance hai phiên ở cuối hướng dẫn. Các test local chỉ dùng tiny random models và Hub giả lập. Không kết luận 7B fit trên T4/P100 từ import hoặc NF4 primitive test.

### Notebook 01: Python mới trên Kaggle (2026-11-13)

Runtime người dùng cung cấp: **CPython 3.13.16, Linux x86_64, glibc 2.39**. Gate bootstrap và preflight dùng chung chính sách Python. Chỉ mở rộng `grounding` sang CPython thường 3.13, không suy rộng sang PEFT/BNB/evaluation chưa xác minh. Giữ nguyên mọi dependency pin; wheel metadata hỗ trợ Python 3.13. Pydantic 2.11.3/core 2.33.1 có wheel cp313, tokenizers 0.21.4 dùng cp39-abi3. Python 3.14 vẫn bị từ chối: core 2.33.1 không có cp314 wheel. Không nâng pin không cần thiết cho runtime 3.13.

Sources: [pydantic-core 2.33.1 wheels](https://pypi.org/project/pydantic-core/2.33.1/), [tokenizers 0.21.4](https://pypi.org/project/tokenizers/0.21.4/), [PyTorch release matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md).

Trước setup, ghi lại đúng runtime (không dump environment hoặc token):

```python
import sys, torch
print(sys.version)
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), torch.cuda.get_arch_list())
```

Dùng clean kernel, `INSTALL_DEPS=1` và chạy lại setup Notebook 01. Không import ST/NumPy/Transformers trước bootstrap. Vẫn có constrained binary-only pip dry-run, bảo vệ CUDA stack, kiểm loaded packages, pip check, snapshot và preflight. Snapshot nay ghi cả Python và effective requirements đúng profile. Nếu thiếu wheel, resolver conflict, kernel warm hoặc torch image quá cũ: dừng, không dùng `--no-deps`, không cài lại torch/CUDA trong notebook để né lỗi.

Sau setup, kiểm tra bằng subprocess mới:

```python
import subprocess
subprocess.run([sys.executable, str(REPO_DIR/'scripts/check_kaggle_environment.py'),
                '--stage', 'grounding', '--device', 'cuda:0',
                '--output', '/kaggle/working/environment-check-01'], check=True, cwd=REPO_DIR)
```

`REPO_DIR` là thư mục source đã attach. CUDA build/driver, kiến trúc GPU và kernel forward/backward vẫn phải pass trên T4/P100 thật; Python/wheel tương thích không bảo đảm CUDA tương thích. Đặc biệt image torch mới có thể không chứa kernels cho P100. Không nâng Python trong phiên resume cũ: dùng lại môi trường và snapshot gốc.

## Workflow acceptance GPU/HF được chuẩn bị

Dùng [KAGGLE_GPU_ACCEPTANCE.md](KAGGLE_GPU_ACCEPTANCE.md) cho script và lệnh T4/P100, optional private Hub round-trip, hai session A/B, expected state/cursor/hash và pass/fail gates. Tool `scripts/run_kaggle_acceptance.py` mặc định không install hoặc upload; `--hub-upload` mới cho ghi fixture <5 MiB trong `acceptance-<UUID>` riêng. Không dùng checkpoint fixture cho research training.

**Kaggle GPU/NF4/Hub thật: NOT EXECUTED trong workspace chuẩn bị tool.** CPU fixture và Hub mocks chỉ là local validation. Chưa xác minh bootstrap trên fresh image Kaggle thực; installer hiện có được tái sử dụng, không thay PyTorch/CUDA stack. TRL optional cho custom loops; `--require-trl` cho gate riêng. Tiny architecture probes không tải pretrained model và không xác minh full-size memory/quality. Chi tiết các flag xem `python scripts/run_kaggle_acceptance.py --help`.

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
    'INSTALL_DEPS': '1',      # cold-kernel component bootstrap, with CUDA constraints
    'TRAIN_DEVICE': 'cuda:0',
    'HF_HOME': '/kaggle/working/hf-cache',
    'SEED': '42',
    'MAX_LENGTH': '512',
    'MAX_STEPS': '100',
    'SAVE_STEPS': '25',
    'GRAD_ACCUM': '8',
})
```

Use Internet-enabled Kaggle for first-time package/model downloads, or attach approved wheel/model datasets for offline use. `requirements-kaggle.txt` is a candidate Python 3.10–3.12 application profile, not a validated Kaggle image. Keep Kaggle's CUDA-enabled torch >=2.5. Optional XGrammar must be installed separately and its native/tokenizer compatibility verified. All installed package versions, Python, GPU name and GPU memory are recorded in `environment.json`.

`meta-llama/Llama-3.2-3B-Instruct` requires accepted model access terms and authenticated HF access. Obtain a token through Kaggle Secrets and grant the configured Kaggle Secret to the notebook; loaders pass the token directly without persisting login credentials, or set `GENERATOR_MODEL` to an attached authorized local base model. Base weights remain required to load a PEFT adapter. Set `BASE_MODEL_PATH` to the corresponding attached base-model directory if the path stored in a shared adapter is unavailable in the receiving account/session. The override must refer to the same model and vocabulary used in SFT. The project does not export gated base weights.

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
2. Enable Internet for initial package/model downloads, or attach compatible wheels and authorized complete local model/tokenizer directories. The notebook's automatic installer uses pip and does not configure an offline wheel directory for you. Install offline wheels explicitly before setup if needed, then use `INSTALL_DEPS=0`. Keep CUDA torch >=2.5. The pinned direct dependencies in [requirements-kaggle.txt](requirements-kaggle.txt) are a candidate Python 3.10–3.12 lock, not a certified Kaggle environment. Restart the kernel after changing imported packages and rerun configuration.
3. For Member 2 and the evaluation operator, authorize the gated Llama base first. With a Kaggle Secret named `HF_TOKEN`, run the following before model loading when downloading from HF:

```python
import os
os.environ['MODEL_HF_TOKEN_SECRET'] = 'HF_TOKEN'  # tên Secret, không phải giá trị token
# Model loaders tự đọc Secret và truyền token trực tiếp; không cần login()
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

1. Before the session ends, save a Kaggle notebook version retaining outputs or download the **entire stage directory**, including logs/environment, complete checkpoints, tokenizers, final exports and GRPO `reference/`. `/kaggle/working` is ephemeral; enable the HF checkpoint backend below for automatic backups. Keep inference exports and reports separately.
2. Publish the saved outputs as a private, versioned Kaggle Dataset accessible to the other two team accounts through normal sharing permissions. Record repository/data versions, configuration, source checkpoint identity and checksums. Share authorized adapters, not gated base weights without the required rights.
3. The recipient attaches that Dataset, inspects its actual directory layout, and updates paths accordingly; upload layouts may add a parent folder. Attach original training inputs and identical base weights separately. Test loading the shared artifact in a fresh session before marking the handoff accepted.
4. Preserve `COMPLETE` markers and `training_state.pt` for recovery; use only trusted team optimizer-state artifacts. `final/` is an inference export for Grounding/System-1, not their optimizer-resume checkpoint. Never point downstream model loading at a stage root when a final/checkpoint directory is required.

### Fresh training and recovery examples

Fresh training means a new output root/experiment ID with `CHECKPOINT_RESUME_SOURCE=none` and `RESUME_FROM_CHECKPOINT` unset. Downloading pretrained base weights is still fresh task training. Every SFT ablation starts fresh; GRPO initializes from SFT but starts its own optimizer unless resuming GRPO.

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

Use the actual saved step and keep data bytes, model identity, seed and objective/length contracts unchanged. A new root is necessary when extending completed Grounding/System-1 runs because existing `final/` destinations are protected. SFT/GRPO also respect their epoch ceilings: a checkpoint at the final default epoch will not become a longer run just by raising `MAX_STEPS`. Resume GRPO from a complete GRPO checkpoint containing its immutable `reference/` adapter, with base access; an SFT checkpoint is not a GRPO recovery state. Incomplete hidden staging directories are ignored. `latest` does not search attached datasets. Do not remove completion checks or overwrite immutable checkpoints to bypass a failure.

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

## Đồng bộ checkpoint tự động lên Hugging Face — hướng dẫn cho 3 thành viên

Cơ chế này đã có kiểm thử local với Hub giả lập và tiny model CPU; **chưa chạy upload/download Hub thật hoặc Kaggle GPU**. Có thể bắt đầu kiểm thử Kaggle có giám sát theo checklist dưới đây; chưa được coi là production-ready.

### 1. Tạo kho riêng tư và Secret

1. Mỗi thành viên tạo một **Model repository, Private** trên Hugging Face, ví dụ `ten-m1/mars-checkpoints`, `ten-m2/mars-checkpoints`, `ten-m3/mars-checkpoints`. Không dùng repository GitHub chứa mã nguồn. Backend không tự tạo repo; tạo repo có README/commit đầu tiên trước khi chạy.
2. Tại HF Settings → Access Tokens, tạo token có quyền **write** vào đúng repo. Nếu dùng fine-grained token, cấp cả quyền đọc nội dung và quyền ghi repo đích. Không dùng token của người khác để ghi cùng thí nghiệm.
3. Trong Kaggle → Add-ons → Secrets, thêm `HF_TOKEN`, dán token vào giá trị Secret và bật quyền truy cập cho notebook. Không dán token vào cell, JSON cấu hình hoặc dataset. Ngoài Kaggle, có thể cung cấp biến môi trường `HF_TOKEN` qua secret manager; biến môi trường theo tên `CHECKPOINT_HF_TOKEN_SECRET` được ưu tiên, rồi mới đọc Kaggle Secrets.
4. Bật Internet cho notebook. Preflight xác minh `private is True`, xác định tài khoản HF và giữ namespace cho tài khoản đó **trước khi tải model/training**. Repo public hoặc namespace thuộc tài khoản khác đều bị từ chối. Khi service không truy cập được và policy `warn`, backend không upload; `auto` chỉ tiếp tục nếu có checkpoint local hợp lệ, không tự chạy fresh khi không biết trạng thái remote.

Tham khảo chính thức: [HF tạo repo private và quyền token](https://huggingface.co/docs/huggingface_hub/quick-start), [commit nguyên tử và `parent_commit`](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.create_commit). Không cần git clone/login vào GitHub hoặc lưu token trong git credentials. Token checkpoint chỉ truyền trực tiếp vào HF API. Model loaders dùng `MODEL_HF_TOKEN_SECRET` nếu đặt, nếu không dùng tên Secret checkpoint (mặc định `HF_TOKEN`). Có thể dùng hai Secret khác nhau cho quyền write backup và quyền read model gated. Member 2 vẫn phải chấp nhận quyền truy cập model; token không được ghi vào config, checkpoint hoặc git credentials.

### 2. Cấu hình chung trước cell setup

```python
import os
os.environ.update({
    'CHECKPOINT_SYNC_ENABLED': '1',
    'CHECKPOINT_SYNC_BACKEND': 'hf_hub',
    'CHECKPOINT_HF_REPO_ID': 'ten-m1/mars-checkpoints',  # thay bằng repo của mình
    'CHECKPOINT_HF_TOKEN_SECRET': 'HF_TOKEN',         # tên Secret, không phải token
    'CHECKPOINT_EXPERIMENT_ID': 'm1-grounding-seed42-v1',
    'CHECKPOINT_SYNC_EVERY_N_SAVES': '1',
    'CHECKPOINT_RESUME_SOURCE': 'auto',
    'CHECKPOINT_SYNC_FAILURE_POLICY': 'warn',
    'SAVE_STEPS': '25',
})
os.environ.pop('RESUME_FROM_CHECKPOINT', None)  # không để override cũ che chế độ auto
```

`CHECKPOINT_SYNC_ENABLED` mặc định `0` để giữ luồng local/offline cũ. Khi bật sync, phải đặt repo và ID; các giá trị mặc định còn lại là `hf_hub`, `HF_TOKEN`, `1`, `none`, `warn`. Muốn tự phục hồi phải đặt `CHECKPOINT_RESUME_SOURCE=auto` rõ ràng. `RESUME_FROM_CHECKPOINT` nếu có sẽ ưu tiên hơn biến này. `CHECKPOINT_SYNC_MAX_ATTEMPTS` mặc định 3, cho phép 1–6 lần; backoff 1, 2, 4, 8, 16 giây, số lần hữu hạn. Các thư viện HTTP của HF còn có timeout/retry riêng.

| Thành viên | Notebook / stage namespace | Repo riêng và ID đề xuất |
|---|---|---|
| 1 | 01 / `grounding` | `ten-m1/mars-checkpoints`, `m1-grounding-seed42-v1` |
| 2 | 02 / `sft_jepa_3b` | `ten-m2/mars-checkpoints`, `m2-sft-only-seed42-v1` hoặc `m2-sft-jepa-seed42-v1` |
| 2 | 03 / `grpo_3b` | repo của Member 2, `m2-grpo-from-jepa-seed42-v1` |
| 3 | 04 / `system1` | `ten-m3/mars-checkpoints`, `m3-modernbert-seed42-v1` |

Remote path là `checkpoints/<experiment-id>/<stage>/checkpoint-N/`. `owner.json` giữ quyền ghi cho **tài khoản HF** đã tạo namespace. `parent_commit` ngăn hai commit cạnh tranh ghi đè nhau; checkpoint hoàn tất không bị thay thế. Không chạy đồng thời hai notebook dùng cùng repo/ID/stage, kể cả khi dùng cùng tài khoản HF: cơ chế không khóa độc quyền toàn bộ phiên training. Mỗi seed, ablation, dataset hoặc cấu hình mục tiêu mới phải có ID và output root mới. Member 2 chạy 02 trước 03; GRPO lần đầu cần `SFT_CHECKPOINT`. Khi resume một checkpoint GRPO đầy đủ, runner dùng `reference/` bên trong checkpoint và không cần đường dẫn SFT của phiên cũ.

### 3. Lưu và backup khi training

- `SAVE_STEPS` là số optimizer update giữa hai lần lưu **local**. `CHECKPOINT_SYNC_EVERY_N_SAVES` là số lần lưu local giữa hai lần upload; hai tần suất độc lập. Ví dụ 25 và 2: lưu local mỗi 25 update, upload mỗi 50 update; checkpoint training cuối được thử upload trước retrieval/calibration/export, rồi runner kiểm tra lại khi hoàn tất.
- Model/adapter, tokenizer khi có, optimizer, scheduler khi dùng, AMP scaler, RNG Python/NumPy/Torch/CUDA, JEPA predictor, step/cursor và resume contract giữ định dạng `training_state.pt` hiện có. Grounding/System-1 suy ra cursor từ step và accumulation; SFT/GRPO lưu epoch/batch cursor. Các runner hiện tại không tạo LR scheduler; runtime vẫn hỗ trợ lưu/khôi phục scheduler nếu được truyền vào.
- Local staging chứa manifest SHA-256/kích thước từng file, `COMPLETE` được ghi sau cùng, rồi rename công bố checkpoint. Chỉ checkpoint có `COMPLETE` và `training_state.pt` mới đủ điều kiện upload. `checkpoint_manifest.json` ghi step, stage, experiment, compatibility fingerprint, dataset/configuration fingerprint, inventory và `status=complete`.
- Upload đồng bộ theo **hai pha**: commit vào `_uploads/<manifest-hash>/` với `INTENT.json`; đọc lại từng file ở revision đó để xác minh; sau đó mới commit thư mục `checkpoint-N` và completion manifest. Discovery bỏ qua `_uploads`, nên lỗi read-back trước publish không tạo checkpoint hoàn tất. Các blob lớn dùng chung nội dung được Hub deduplicate. Retry sau mất phản hồi nhận diện staging/publication đã tồn tại và không ghi đè checkpoint hoàn tất. Staging được giữ để điều tra, không tự xóa.
- Log hiển thị thành công/thất bại, step, payload bytes, thời gian và `skipped`. `checkpoint_sync.jsonl` ở stage output giữ log; bytes là kích thước payload logic của commit, không phải số byte HTTP thực tế sau dedup/retry. Việc đọc lại để xác minh có thể cần tải thêm dữ liệu vào HF cache.
- `warn`: checkpoint local được giữ, in cảnh báo **REMOTE BACKUP UNSUCCESSFUL** và tiếp tục. Khi resume `auto`, remote lỗi/không tương thích không chặn local đã xác minh; khi không có bản hợp lệ, hệ thống báo lỗi thay vì tự khởi động từ step 0. Training thành công không đồng nghĩa backup thành công; kiểm tra log cuối. `stop`: ném lỗi sau khi đã giữ checkpoint local, không tiếp tục training. Resume thất bại không được tự đổi thành chạy mới từ step 0.
- Không xóa checkpoint local/remote, không upload dataset hoặc toàn bộ output root. Chỉ upload thư mục checkpoint đã hoàn tất. `final/` của 01/04, index, schema graphs, calibration metadata, dữ liệu split và các báo cáo vẫn cần giữ/bàn giao riêng qua Kaggle outputs hoặc Dataset; chúng không nằm trong backup optimizer checkpoint.

### 4. Bắt đầu mới và resume trong phiên khác

| Giá trị `CHECKPOINT_RESUME_SOURCE` | Hành vi |
|---|---|
| `none` | Chạy mới; chọn ID và output mới để không va checkpoint cũ |
| `local` | Chọn checkpoint local hợp lệ có step cao nhất; thiếu thì báo lỗi |
| `latest` | Alias tương thích cũ của `local` |
| `remote` | Bắt buộc tải checkpoint remote tương thích, đầy đủ, checksum đúng; thiếu thì báo lỗi |
| `auto` | So local/remote, ưu tiên step hoàn tất cao nhất; bằng step ưu tiên local; namespace trống mới được chạy từ đầu |
| đường dẫn checkpoint | Dùng chính xác thư mục đó, xác minh completeness/integrity/compatibility; không fallback |

Bắt đầu thí nghiệm mới: đặt `CHECKPOINT_RESUME_SOURCE=none`, ID mới và `KAGGLE_OUTPUT_DIR` mới; xóa override `RESUME_FROM_CHECKPOINT` bằng `os.environ.pop`. Không xóa checkpoint cũ.

Resume cùng thí nghiệm trong **phiên Kaggle hoàn toàn mới**: attach cùng snapshot code, dữ liệu/split và base model được cấp quyền; giữ nguyên ID/repo/model/objective/seed/length/accumulation/learning rate. Chỉ tăng `MAX_STEPS` thành tổng update mong muốn và đổi output root:

```python
os.environ.update({
    'KAGGLE_OUTPUT_DIR': '/kaggle/working/recovered-session-b',
    'CHECKPOINT_RESUME_SOURCE': 'remote',  # dùng remote cho kiểm thử xác nhận; auto cho vận hành
    'MAX_STEPS': '200',
})
os.environ.pop('RESUME_FROM_CHECKPOINT', None)
```

Preflight in latest completed local/remote steps; runner in `resume_checkpoint` và `verified_completed_step`. Thấy step N đã lưu, update tiếp theo phải là N+1; không chấp nhận mất optimizer/cursor hoặc tự về 0. Nếu dataset/config mismatch: dùng lại đúng cấu hình hoặc tạo ID mới, không bỏ kiểm tra. Checkpoint mới có `base_identity.json`: model Hub được pin commit SHA, local base được hash weights/config/tokenizer. `BASE_MODEL_PATH` cho phép chuyển vị trí **cùng base model** khi PEFT loader tải adapter và được kiểm tra identity. Hub→local override yêu cầu đúng các file nguồn theo hash metadata; không chấp nhận model chỉ có cùng tensor shape. Nếu dùng base local không còn đường dẫn gốc, attach lại đúng model và cấu hình override này. Không đổi base dưới adapter đã train.

SFT/GRPO có giới hạn epoch ngoài `MAX_STEPS`: tăng step cap không kéo dài training đã hết epoch. Muốn kéo dài epoch cần chỉnh lời gọi runner rõ ràng. Grounding/System-1 bảo vệ `final/` đã có, nên dùng output mới khi tiếp tục sau một run hoàn tất. Checkpoint legacy chỉ có marker/state vẫn đọc được qua contract cũ; chưa có manifest/hash ban đầu để xác minh dữ liệu trước đó, nên chỉ dùng artifact đáng tin. Lưu thêm một update sẽ tạo checkpoint chuẩn mới để backup. Legacy adapter không có identity và cần `BASE_MODEL_PATH` phải được operator xác minh trước, rồi opt-in `ALLOW_LEGACY_BASE_OVERRIDE=1`; hệ thống không thể suy ra identity lịch sử. Legacy CUDA checkpoint thiếu active device chỉ được tự restore trên cùng số GPU; không đoán mapping khi topology khác.

### 5. Chia sẻ khác tài khoản và Notebook 05 chỉ đọc

Chia sẻ repo private qua quyền thành viên tổ chức HF phù hợp; repo cá nhân không có quyền cộng tác cần chuyển artifact qua repo tổ chức/private Kaggle Dataset được cấp quyền. Token người nhận phải có quyền đọc repo nguồn. Không trao token cá nhân cho người khác. Người nhận tải bằng helper **chỉ đọc**, không claim namespace, không unpickle state và không ghi vào repo nguồn:

```python
# Cấu hình CHECKPOINT_SYNC_ENABLED=1, repo/experiment nguồn và Secret có quyền đọc trước.
from src.training.checkpoint_sync import download_checkpoint_artifact
checkpoint = download_checkpoint_artifact('/kaggle/working/shared-artifacts', 'sft_jepa_3b')
# Dùng làm SFT_CHECKPOINT cho lần GRPO đầu, hoặc GENERATOR_CHECKPOINT cho evaluation.
```

Nếu tiếp tục training từ checkpoint được bàn giao: sau download, đổi `CHECKPOINT_HF_REPO_ID` sang repo người nhận, `CHECKPOINT_EXPERIMENT_ID` sang ID mới, và đặt `RESUME_FROM_CHECKPOINT=checkpoint`. Giữ dữ liệu/cấu hình; update mới được backup vào namespace mới. Nếu chỉ load và không tạo update mới thì checkpoint vẫn thuộc namespace nguồn, không tự đổi nhãn/bản sao. Không dùng `remote` để ghi tiếp trực tiếp namespace của người gửi.

Notebook 05 không gọi preflight có quyền ghi hoặc sync training. Khi cần, operator chủ động gọi helper trên trong cell cấu hình rồi đặt các artifact path. SFT/GRPO adapter checkpoint có thể dùng trực tiếp cho Generation. Grounding/System-1 inference yêu cầu `final/`, index/calibration artifacts riêng; không thay `final/` bằng optimizer checkpoint rồi coi như đã hiệu chỉnh nhiệt độ.

### 6. Lỗi thường gặp, dung lượng và bảo mật

| Triệu chứng | Xử lý |
|---|---|
| Token unavailable / Secret bị từ chối | Kiểm tra đúng tên `HF_TOKEN`, quyền Secret cho notebook; không in token để debug |
| HTTP 401/403 hoặc service failed | Kiểm tra token còn hiệu lực, scope đọc/ghi đúng repo, quyền tổ chức và quyền gated base riêng |
| HTTP 404 / không xác minh private | Kiểm tra `owner/repo`, loại **Model**, repo đã có commit đầu; repo phải private |
| Namespace belongs to another account | Dùng read-only download để bàn giao, repo/ID đích mới cho người nhận |
| Timeout, DNS, Internet tắt | Bật Internet, kiểm tra quota/dịch vụ; retry hữu hạn. `warn` giữ local nhưng không có backup mới |
| Checksum/manifest mismatch | Không resume artifact lỗi. Chọn checkpoint hoàn tất cũ hơn nếu có, giữ bản lỗi để điều tra |
| Dataset/configuration mismatch | Attach đúng phiên bản dữ liệu/split, giữ objective/model/seed/length/hyperparameters |
| Insufficient disk space / ENOSPC | Giảm tần suất lưu phù hợp, tăng dung lượng trước run hoặc chuyển run sang output còn trống; không xóa bản hợp lệ duy nhất |
| Checkpoint step đã tồn tại | Resume đúng step hoặc output/experiment mới; hệ thống không ghi đè checkpoint đã hoàn tất |

Model full + AdamW có thể lớn hơn adapter rất nhiều. Tính cả checkpoint local, staging và HF cache; save ước tính tensor payload với headroom; download kiểm tra cả output và HF cache (2× payload nếu cùng filesystem); upload kiểm tra chỗ cho read-back cache. `CHECKPOINT_DISK_RESERVE_BYTES` mặc định 256 MiB. Đây là lower-bound guard, không phải dự báo quota/serialization chính xác cho toàn run. Full-model FP32 + AdamW checkpoint xấp xỉ 12 byte/parameter, vì vậy cần dự toán số checkpoint giữ lại và cache trước run. Log sync có thể không ghi được khi ổ đầy; lỗi OS được giữ nguyên để chẩn đoán. Artifact optimizer là pickle: chỉ load từ nhóm/repo đáng tin; checksum xác minh tính toàn vẹn, không chứng minh nguồn đáng tin hay giấy phép phân phối. Dữ liệu/ngữ cảnh nghiên cứu có thể nằm trong model, nên repo luôn private và quyền truy cập tối thiểu. Không bật HTTP debug logging chứa credential bên ngoài ứng dụng.

### 7. Checklist hai phiên Kaggle thật

1. **Chuẩn bị:** chọn notebook được giao, repo private có commit đầu, Secret write đúng repo; attach dữ liệu/base; bật Internet/GPU. Dùng ID thử nghiệm mới, `SAVE_STEPS=1`, sync mỗi save, `MAX_STEPS=2`, policy `stop`, resume `none`. Giữ đủ dữ liệu để có ít nhất 3–4 update trong epoch budget.
2. **Phiên A:** Save Version → **Save & Run All**. Xác nhận privacy preflight và step tiến 1, 2; xem `checkpoint_sync.jsonl` có `success=true`, step 2 và duration/bytes. Mở HF repo bằng UI, kiểm tra đúng namespace, marker, manifest, state, weights/tokenizer; 03 có `reference/`. Ghi commit SHA và cấu hình, không ghi token.
3. **Giữ bằng chứng A:** lưu logs/environment/metrics và artifacts `final/`/split riêng nếu có. Không xóa checkpoint. Kết thúc phiên A sau khi upload thành công; không dựa vào trình duyệt mở hoặc background upload sau khi process bị kill.
4. **Phiên B độc lập:** mở kernel/session mới; không attach output checkpoint A, dùng output root trống khác. Attach cùng code/dữ liệu/base; cùng repo/ID/Secret và cấu hình, đổi `MAX_STEPS=3` hoặc 4, resume **remote**, bỏ `RESUME_FROM_CHECKPOINT`. Với 03 bỏ `SFT_CHECKPOINT` cũ để chứng minh `reference/` tự đủ (base model vẫn bắt buộc).
5. **Kiểm chứng restore:** latest local phải trống, remote phải 2; `verified_completed_step=2`; update mới từ 3. Kiểm tra state optimizer/scaler/RNG, cursor, JEPA predictor (02 có lambda>0), reference hash (03) và checkpoint 3/4. Không có checkpoint step 0/1 bị ghi lại. Dùng `MAX_STEPS` đủ nhỏ để kiểm tra trước khi chạy dài.
6. **Kiểm chứng Hub B:** kiểm tra sync cuối thành công và checkpoint mới; checkpoint 2/commit A còn nguyên. Thử rerun với tổng step đã đạt ở output mới để xác nhận không ghi đè. Với 01/04 giữ và so sánh `final/`/calibration output riêng.
7. **Mô phỏng lỗi có kiểm soát:** trong một run thử riêng, ngắt mạng hoặc dùng token không có quyền ghi sau checkpoint đã lưu; kiểm tra policy `warn`/`stop` và local vẫn còn. Khôi phục quyền/mạng, retry trên checkpoint hoàn tất; không sửa checksum/marker để vượt lỗi.
8. **Chấp nhận ban đầu:** cả ba thành viên ghi lại version code/packages, GPU/VRAM, commit SHA, duration/dung lượng, start/resume/final steps và kết quả kiểm tra trạng thái. Chỉ sau upload+resume thật mới đánh dấu backend đã được kiểm chứng tích hợp; full-size convergence/benchmark vẫn là cổng đánh giá riêng.

Đồng bộ định kỳ giảm rủi ro mất tiến độ, **không bảo đảm phục hồi optimizer step chưa lưu hoặc checkpoint chưa upload xong**. Kaggle `/kaggle/working` là tạm thời, GPU có giới hạn runtime và mạng có thể mất. Save & Run All chạy phía Kaggle không đòi trình duyệt mở, nhưng process bị kill không thể tiếp tục upload. Luôn chọn khoảng lưu/backup dựa trên thời gian thực đo được và mức tiến độ chấp nhận mất.


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
