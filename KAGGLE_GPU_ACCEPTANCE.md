# Kaggle GPU acceptance và private HF checkpoint round-trip

**Trạng thái khi chuẩn bị workflow: T4, P100, CUDA/NF4, private Hub và hai session Kaggle thật = NOT EXECUTED.** Workspace hiện tại không có Kaggle connector/session được cấp quyền. Test CPU và Hub giả lập chỉ xác minh tool, không phải acceptance GPU.

Tool mới: `scripts/run_kaggle_acceptance.py`. Nó tái sử dụng `validate_environment`, constrained bootstrap, `save_checkpoint`/`load_checkpoint`, `CheckpointSync`, checksum và model identity của project. Không có full-training mode hoặc download pretrained model. Các model probe chỉ có một layer nhỏ, vocab 32, sequence 2–8 tokens.

## 1. Các mức bằng chứng

| Mức | Thực hiện | Không chứng minh |
|---|---|---|
| `--cpu-test` | Fixture local: optimizer, scheduler, scaler khi có, Python/NumPy/Torch RNG, predictor, frozen reference, data cursor, manifest và continuation | GPU, NF4, dependency profile Kaggle, Hub thật |
| GPU `--phase local` | Profile/version/import/pip check, CUDA kernel, fixture GPU; NF4 với profile Generation/Evaluation | Private Hub nếu chưa bật flag; full-size training |
| `--component-probe` | Tiny Qwen3 + ST/HN-SupCon; tiny Llama + project NF4/PEFT/SFT/JEPA/GRPO APIs; tiny ModernBERT/RLCD hoặc generation+classifier init | Pretrained weights, real dataset, GRPO reward quality, calibration/E2E benchmark |
| `--hub-upload` | Opt-in ghi một fixture checkpoint nhỏ lên repo private, download và verify qua backend thật | Recovery khi process bị kill trước save/upload |
| Session A → B | Tải checkpoint vào output trống rồi so state và một update tiếp theo với golden continuation | Tool không tự chứng minh runtime Kaggle đã được đóng/mở; operator phải xác nhận |
| Real component integration | Sau đó chạy notebook nghiên cứu với đúng artifact/data và 1–2 optimizer updates | Full-budget fit/convergence/benchmark; đặc biệt không chứng minh 7B fit 16 GB |

`acceptance_report.json` luôn có `full_training_allowed: false`: fixture pass là một cổng kỹ thuật, không tự cấp phép chạy full research. Chỉ coi là bằng chứng Kaggle nếu `execution_site=kaggle`, hardware đúng và operator lưu được bằng chứng runtime. CPU mode ghi `PASS_LOCAL_CPU_ONLY`; các mục chưa chạy giữ `NOT_EXECUTED`. GPU thành công ghi `PASS_INFRASTRUCTURE`, không phải production readiness.

## 2. Bootstrap và kiểm tra an toàn

Dùng cùng snapshot source ở A/B, bao gồm các thay đổi chưa commit nếu có. Repo được attach dưới `/kaggle/input`; output/cache phải ở `/kaggle/working`. Không attach checkpoint của A vào B.

Cấu hình cell trước khi chạy subprocess:

```python
import os, sys, subprocess
from pathlib import Path
REPO_DIR = Path('/kaggle/input/optimize-mars-repo/Optimize-Mars-Sl')  # sửa đúng layout
assert (REPO_DIR/'src').is_dir()
os.environ.update({
    'HF_HOME': '/kaggle/working/hf-acceptance-cache-a',
    'CHECKPOINT_HF_REPO_ID': 'YOUR_ACCOUNT/private-checkpoints',
    'CHECKPOINT_HF_TOKEN_SECRET': 'HF_TOKEN',  # tên Secret, KHÔNG phải token
})
```

Tạo trước một **HF Model repository private có README/commit đầu**. Trong Kaggle Secrets, tạo `HF_TOKEN`, bật quyền cho notebook. Token phải có read/write đúng repo cho A. Không dán token vào code/command, không dump environment. Model token có thể tách bằng `MODEL_HF_TOKEN_SECRET`; `--check-model-secret` chỉ kiểm tra Secret khả dụng, không chứng minh quyền gated model.

Chọn **một** GPU target: `t4` hoặc `p100`; không đổi target giữa A/B của cùng run. `--device cuda:0` chọn một GPU, không gộp VRAM hai T4.

Bootstrap được chạy trong process mới trước import ML:

- Python 3.10–3.12 và CUDA torch >=2.5,<3 của image là điều kiện hiện có, không được tool tự thay thế.
- Chọn component profile từ `requirements/kaggle-*.txt`.
- Pip dry-run + constraints bảo vệ torch/torchvision/torchaudio/triton; từ chối thay package đã load; install và `pip check`.
- Giữ `dependency_snapshot.json`, `environment.json`, package versions và driver/GPU report.
- Profile là candidate API constraints, chưa phải certified Kaggle image/full transitive lock. Nếu resolver đòi thay CUDA stack hoặc kernel đã preload package không phù hợp: dừng và chọn image/profile phù hợp, không dùng `--no-deps` để bỏ lỗi.

Không có flag bootstrap thì tool không install gì. Kiểm tra wiring offline trước:

```bash
python scripts/run_kaggle_acceptance.py --help
python scripts/run_kaggle_acceptance.py --cpu-test --output /tmp/mars-acceptance-cpu
```

CPU mode cấm bootstrap/Hub/session/Secret flags. Dù ambient environment có `CHECKPOINT_SYNC_ENABLED=1` hoặc experiment training thật, tool vẫn override trong scope riêng rồi restore lại; không ghi namespace training đó.

## 3. GPU acceptance local, chưa ghi Hub

T4, ví dụ SFT/JEPA:

```python
subprocess.run([
    sys.executable, str(REPO_DIR/'scripts/run_kaggle_acceptance.py'),
    '--gpu', 't4', '--device', 'cuda:0', '--component', 'sft_jepa_3b',
    '--bootstrap', '--component-probe', '--phase', 'local',
    '--output', '/kaggle/working/acceptance-local',
], check=True, cwd=REPO_DIR)
```

P100: đổi **duy nhất** `--gpu t4` thành `--gpu p100` trong ví dụ này. Tool kiểm cả GPU name và capability `(7,5)`/`(6,0)`, không chấp nhận một GPU khác chỉ vì có cùng VRAM.

Lặp lại riêng với `--component grounding`, `system1`, `grpo_3b`, `evaluation`. Mỗi lần tạo UUID/output riêng. `--steps` mặc định 2, chỉ cho phép 1–3; golden branch và replay thêm một update trên model tiny. Đây không phải optimizer updates trên 3B/7B thật.

Imports Transformers/PEFT/Accelerate/BNB được ghi rõ trong report. TRL không được dùng trong custom loops: thiếu TRL sẽ là `NOT_INSTALLED_OPTIONAL`, không tự cài TRL và không đánh dấu import thành công. Nếu cần gate TRL riêng, thêm `--require-trl`; lúc đó thiếu/broken import sẽ fail. Profile Grounding/System-1 không bắt buộc PEFT/BNB; NF4 ghi `NOT_EXECUTED` khi không thuộc profile đó. Chạy Generation hoặc Evaluation để kiểm chứng NF4.

## 4. Session A — upload fixture qua backend hiện có

Chỉ chạy cell sau khi chủ động chấp nhận ghi test artifact vào repo private đã cấu hình:

```python
subprocess.run([
    sys.executable, str(REPO_DIR/'scripts/run_kaggle_acceptance.py'),
    '--phase', 'session-a', '--gpu', 't4', '--device', 'cuda:0',
    '--component', 'sft_jepa_3b', '--component-probe', '--bootstrap',
    '--steps', '2', '--hub-upload',
    '--output', '/kaggle/working/acceptance-session-a',
], check=True, cwd=REPO_DIR)
```

Không có `--hub-upload` thì Session A bị từ chối trước network access. Namespace do tool tự tạo:

```text
checkpoints/acceptance-<UUID4-hex>/acceptance_fixture_v1/checkpoint-2/
```

Tool không nhận experiment training ID và không cho đặt run ID cho A/local. Payload được giới hạn **5 MiB**, chỉ chứa synthetic model/state, không chứa datasets, token hoặc model nghiên cứu. Backend vẫn dùng private/owner checks, staging verification, immutable publication và bounded retries. Staging/checkpoint không tự xóa. Không đổi config training của notebook 01–05.

Kết quả cần giữ:

- `result=PASS_INFRASTRUCTURE`, `execution_site=kaggle`.
- Environment/CUDA/NF4/import/pip checks và component probe tương ứng PASS.
- `fixture_lifecycle`: restored step 2, optimizer steps `[2]`, scheduler epoch 2, cursor 4; continuation step 3, optimizer `[3]`, cursor 6.
- Hashes model/predictor/reference/optimizer/scheduler/scaler/RNG match expected; `next_rng_probe=PASS`, `exact_continuation=PASS`.
- `private_repo_read=PASS`, `private_hub_round_trip=PASS`, payload bytes, immutable `round_trip_revision`.
- `cross_session_resume=NOT_EXECUTED` ở A. Same-session download không được coi là bằng chứng hai session.

Ghi lại `run_id` gồm 32 ký tự hex, repo, component, GPU target, steps, commit SHA và package snapshot. Mở HF UI xác nhận đúng namespace, `COMPLETE`, `checkpoint_manifest.json`, `base_identity.json`, `training_state.pt`; checkpoint fixture dùng native state, không phải adapter 3B. Không dùng nó làm đầu vào SFT/GRPO nghiên cứu.

## 5. Session B — clean runtime, chỉ download mặc định

1. Giữ report/snapshot bằng Save Version hoặc download; sau đó kết thúc runtime A.
2. Mở runtime Kaggle mới, cùng loại GPU. Attach cùng source snapshot; không attach output/checkpoint A. Có thể giữ riêng report nhỏ để đối chiếu.
3. Cấu hình lại Secret/repo, đặt cache mới và run ID từ A:

```python
import os, sys, subprocess
from pathlib import Path
REPO_DIR = Path('/kaggle/input/optimize-mars-repo/Optimize-Mars-Sl')
os.environ.update({
    'HF_HOME': '/kaggle/working/hf-acceptance-cache-b',
    'CHECKPOINT_HF_REPO_ID': 'YOUR_ACCOUNT/private-checkpoints',
    'CHECKPOINT_HF_TOKEN_SECRET': 'HF_TOKEN',
})
RUN_ID = 'PASTE_32_HEX_RUN_ID_FROM_A'
subprocess.run([
    sys.executable, str(REPO_DIR/'scripts/run_kaggle_acceptance.py'),
    '--phase', 'session-b', '--run-id', RUN_ID,
    '--gpu', 't4', '--device', 'cuda:0',
    '--component', 'sft_jepa_3b', '--component-probe', '--bootstrap',
    '--steps', '2', '--output', '/kaggle/working/acceptance-session-b',
], check=True, cwd=REPO_DIR)
```

B tìm đúng source step 2 trong namespace của A, kiểm manifest/size trước download, fetch vào thư mục trống, rồi restore bằng runtime thật. `base_identity.json` phải khớp model fixture được khởi tạo lại; contract gồm dataset/source fingerprint, Python/Torch/CUDA build, GPU target và component. Không có dữ liệu/model state của process A được dùng từ RAM.

Golden continuation được lưu trong checkpoint A dưới dạng hash, không phải weights phỏng đoán. B so từng nhóm state sau restore và sau một optimizer update, kiểm RNG draws Python/NumPy/Torch CPU/active GPU. Frozen reference là reference của fixture; probe GRPO riêng kiểm immutable PEFT reference adapter. B lưu checkpoint local step 3. `cross_session_resume=PASS` chứng minh Hub restore vào output mới và replay đúng; trường `fresh_kaggle_runtime=OPERATOR_ATTESTATION_REQUIRED` nhắc operator xác nhận thực sự đã tạo runtime mới. Tool không tự chứng minh hai máy/session khác nhau.

B mặc định dùng backend read-only để fetch và backend disabled để lưu local: **không ghi repo**. Có thể thêm `--hub-upload` rõ ràng để thử backup step 3 và round-trip lần nữa; token/HF identity phải được phép ghi namespace A. Mỗi A/B write test chỉ chạy một lần; backend không overwrite step hoàn tất. Muốn thử lại, tạo A mới hoặc dùng B read-only với output root mới.

Với Kaggle account khác: token người nhận phải được chia sẻ quyền read repo hợp lệ. Ownership có thể từ chối write bằng HF identity khác; đừng chia sẻ token cá nhân để vượt guard. Kiểm write riêng bằng một A mới thuộc repo của người nhận.

## 6. T4/P100 checklist và resource assessment

| Điều kiện | T4 | P100 |
|---|---|---|
| Hardware profile | T4, CC 7.5 | P100, CC 6.0 |
| CUDA binary | Phải chạy kernel thật; GPU enumeration chưa đủ | Phải có kernel phù hợp Pascal, không mặc định mọi wheel mới hỗ trợ |
| Precision baseline | FP16 cho QLoRA, không giả định native BF16 | FP16, không giả định native BF16 |
| NF4 | Test BNB thực tế | Test riêng; khả năng NF4 không đồng nghĩa `LLM.int8()` được hỗ trợ |
| ModernBERT | `reference_compile=False` | Bắt buộc baseline này để không tự chạy compiler không phù hợp |
| Flash Attention 2 chính | Không cài/ép dùng trong workflow | Không cài/ép dùng |
| Resume | Cùng profile/software/GPU target để so golden bitwise | Run ID riêng; không dùng golden T4 để tuyên bố bitwise P100 |

[NVIDIA capabilities](https://developer.nvidia.com/cuda/gpus/legacy), [BNB 0.45.5 requirements](https://raw.githubusercontent.com/bitsandbytes-foundation/bitsandbytes/0.45.5/docs/source/installation.mdx), [FlashAttention support](https://github.com/Dao-AILab/flash-attention#nvidia-cuda-support).

| Research component | Strategy / memory pressure | Rủi ro và cổng profiling |
|---|---|---|
| Schema preparation | SQLite/graphs chủ yếu CPU | Dung lượng dữ liệu, schema provenance, disk; không cần GPU |
| Qwen3 embedding 0.6B | Full model + AdamW + eager attention/checkpointing; nếu FP32, tensors nền xấp xỉ 16 bytes/parameter trước activation | Hard-negative batch, attention buffers và full checkpoints lớn; profile cả T4/P100 |
| SFT+JEPA 3B | NF4 double quantization, FP16 compute, LoRA + predictor | Activations/logits/all hidden states cho JEPA; base/tokenizer identity; phải chạy real 1–2 updates |
| GRPO 3B | Shared quantized base, policy/reference adapters, rollout tuần tự | KV cache, full-vocab logits, generation+backward khác peak; zero-variance rewards có thể không update; cần đo runtime/VRAM |
| GRPO 7B | Cùng chiến lược, cần SFT adapter đúng base 7B | **Chưa chứng minh fit 16 GB**. Không ngoại suy từ tiny fixture hoặc 3B; test loading rồi rollout/backward riêng, budget ngắn |
| ModernBERT-large System-1 | Full-model FP32 + AMP, dual LR; không NF4 | Compiler disabled không chứng minh memory fit; calibration/data split artifacts cần giữ riêng |
| Evaluation | Generator NF4; Grounding/scorer mặc định CPU; K candidates tuần tự | Prompt/KV/logit scores và CPU RAM/latency; cần artifact calibrated thật; tiny generation/classifier không phải E2E acceptance |

FP32 AdamW checkpoint khoảng 12 bytes/parameter: 0.6B danh nghĩa khoảng 7.2 GB, 395M khoảng 4.7 GB, trước overhead. Đây là estimate có điều kiện, không phải số đo. 4-bit payload lý tưởng của 3B/7B khoảng 1.5/3.5 GB **chưa gồm** modules không quantized, optimizer/adapter, activations, KV cache và allocator. Peak của tool chỉ dành cho tiny fixtures/probes, tuyệt đối không ngoại suy thành full-model VRAM.

## 7. Output, diagnostics và acceptance gate

Tool giữ `environment.json`, snapshot nếu bootstrap, tiny assets, checkpoints và `acceptance_report.json` dưới output acceptance riêng. Exit 0 chỉ khi các bước được yêu cầu thực sự pass; exit 1 khi check fail; exit 2 cho options/output không hợp lệ. Optional/không yêu cầu không được báo PASS. Không tự xóa artifact để retry.

| Failure | Hành động |
|---|---|
| `environment`/imports/version mismatch | So package snapshot, chạy `check_kaggle_environment.py` với đúng stage; fresh kernel/image; không thay CUDA stack mù |
| Bootstrap protected stack/warm package | Chọn profile/image phù hợp hoặc cold process; không dùng `--no-deps` để bỏ conflict |
| GPU target mismatch / CUDA kernel error | Kiểm tên GPU, CC, torch build/arch/driver; T4 và P100 là hai acceptance riêng |
| NF4/import BNB lỗi | Kiểm binary BNB/CUDA trên đúng GPU; không quy mất Internet thành package incompatibility |
| Private repo/Secret 401/403/404 | Kiểm Secret được grant, đúng private Model repo có commit, token scope/owner; không in token |
| Upload/read-back/network/disk | Xem sanitized HTTP category trong `checkpoint_sync.jsonl`; retry hữu hạn; staging không là completed checkpoint |
| Session A source step không thấy | Kiểm repo/run ID/steps/component/GPU/package/source snapshot; không fallback fresh |
| Restored/continued hash mismatch | Dừng. Kiểm đúng checkpoint/contract, package/CUDA/hardware và RNG/cursor; không bỏ assertion để lấy PASS |
| OOM | Dùng fresh runtime. Tiny probe OOM là fail hạ tầng; full-model fit vẫn phải profile riêng |
| Output tồn tại | Output root mới; không xóa checkpoint nghiên cứu hoặc overwrite evidence |

**PASS để chuyển sang thử integration ngắn:** đúng GPU + imports/profile + CUDA/NF4 tương ứng + tiny component + fixture lifecycle + private Hub A round-trip + B clean-runtime proof; lưu đủ logs/snapshot/revisions và xác nhận runtime separation thủ công.

**PASS trước full training:** thêm real-component integration 1–2 updates với dữ liệu/authorized base/pretrained weights thực; đo loading/forward/backward/rollout peak VRAM, disk growth, upload duration/throughput; resume thật optimizer/RNG/JEPA/reference adapter/cursor; bảo quản `final/`, calibration/index/graph/splits; xác nhận budget checkpoint không làm đầy disk/quota. Với GRPO phải có actual policy updates, không chỉ `NO_POLICY_UPDATES`. Cấu hình 7B là cổng riêng, không tự được duyệt từ run 3B. Notebook 05 vẫn read-only với training checkpoints.

Bất kỳ check bắt buộc FAIL hoặc NOT EXECUTED đều chặn full training. Session kill có thể làm mất optimizer steps chưa save/upload; một lần acceptance pass không chứng minh độ ổn định dài hạn hoặc model quality.
