# Adversarial Challenge Report — Milestone M3 (Phase 3: Google Colab Training Notebooks)
**Agent**: `challenger_m3` (Empirical Challenger: critic, specialist)  
**Date**: 2026-11-02  
**Working Directory**: `/mnt/c/Workspace/optimize/.agents/teamwork/challenger_m3/`  
**Verdict**: **REJECT (Request Changes)**

---

## 1. Observation (Quan Sát Thực Nghiệm)

1. **AST & Syntax Robustness**:
   - Quét toàn bộ 5 notebook (`notebooks/01_data_preparation_and_embedding.ipynb` đến `05_end_to_end_evaluation.ipynb`) qua `ast.parse`: 100% các ô mã nguồn (49/49 code cells) đều là cú pháp Python hợp lệ, không có lỗi cú pháp hay escape không hợp lệ.

2. **Khảo sát Tính Khả Dụng Của Mã Nhập Khẩu (Import & Symbol Resolution)**:
   Khi thực thi phân tích đối chiếu tĩnh và động giữa các lệnh `import` trong notebook với các module thực tế trong `src/` và `scripts/`, phát hiện **4 lỗi nghiêm trọng** do nhập khẩu các hàm không tồn tại (Phantom Functions):
   - **Lỗi 1 (`notebooks/01_data_preparation_and_embedding.ipynb`, Ô số 2)**:
     ```python
     from scripts.make_mini_dataset import generate_all
     generate_all(DATA_DIR)
     ```
     *Thực tế*: Module `scripts/make_mini_dataset.py` chỉ định nghĩa `create_financial_db`, `create_schools_db`, `generate_mini_bird_jsonl`, `generate_mini_synsql_jsonl`, và `main()`. Hoàn toàn không có hàm `generate_all`. Nếu người dùng chạy trên Colab mới chưa có sẵn `data/mini/`, ô lệnh lập tức văng lỗi:
     `ImportError: cannot import name 'generate_all' from 'scripts.make_mini_dataset'`
   - **Lỗi 2 (`notebooks/02_sft_and_jepa_training_3b.ipynb`, Ô số 8)**:
     ```python
     from src.training.train_sft_generator import run_sft_training
     train_summary = run_sft_training(...)
     ```
     *Thực tế*: Module `src/training/train_sft_generator.py` chỉ xuất `run_dry_run` và `main()`. Không tồn tại hàm `run_sft_training`. Chạy ô này lập tức ném lỗi:
     `ImportError: cannot import name 'run_sft_training' from 'src.training.train_sft_generator'`
   - **Lỗi 3 (`notebooks/03_grpo_training_3b_and_7b.ipynb`, Ô số 8)**:
     ```python
     from src.training.train_grpo_generator import run_grpo_training
     grpo_summary = run_grpo_training(...)
     ```
     *Thực tế*: Module `src/training/train_grpo_generator.py` chỉ xuất `run_dry_run` và `main()`. Không tồn tại hàm `run_grpo_training`. Chạy ô này lập tức ném lỗi:
     `ImportError: cannot import name 'run_grpo_training' from 'src.training.train_grpo_generator'`
   - **Lỗi 4 (`notebooks/04_system1_laya421m_training.ipynb`, Ô số 7)**:
     ```python
     from src.training.train_decision_model import run_decision_model_training
     train_results = run_decision_model_training(...)
     ```
     *Thực tế*: Module `src/training/train_decision_model.py` chỉ xuất `run_dry_run` và `main()`. Không tồn tại hàm `run_decision_model_training`. Chạy ô này lập tức ném lỗi:
     `ImportError: cannot import name 'run_decision_model_training' from 'src.training.train_decision_model'`

3. **Mô Phỏng Môi Trường Colab & Google Drive Mount**:
   - Các notebook 02, 03, 04, 05 đều bao bọc khối mount Drive trong:
     ```python
     try:
         from google.colab import drive
         drive.mount('/content/drive')
         ...
     except (ImportError, Exception):
         ... fallback sang ./colab_drive_mock/
     ```
     Khối lệnh xử lý suy thoái mượt mà (graceful degradation) khi chạy ngoài môi trường Colab mà không làm crash tiến trình.
   - *Lưu ý nhỏ ở Notebook 05 Ô 1*: `import torch` được đặt ở đầu cell trước khối `try: drive.mount`, do đó nếu môi trường kiểm thử cục bộ chưa cài `torch`, cell này sẽ dừng trước khi chạm tới fallback Drive.

4. **Kiểm Tra Phục Hồi Checkpoint (Edge Cases)**:
   - Thư mục trống hoặc không tồn tại: `find_latest_checkpoint` trả về `None` an toàn.
   - Thư mục chỉ chứa `.ipynb_checkpoints` hoặc tên dị thường (`checkpoint-abc`): Biểu thức chính quy `r"^checkpoint[-_](\d+)$"` lọc chính xác và trả về `None`.
   - Thứ tự bước kiểm tra (Step Ordering): Checkpoints `checkpoint-2`, `checkpoint-10`, `checkpoint-100` được sắp xếp theo số học (`step = int(...)`), trả về `checkpoint-100` (không bị lỗi sắp xếp từ điển nơi `"checkpoint-2" > "checkpoint-100"`).
   - Khi không có checkpoint đánh số, tự động fallback về thư mục `best_model` nếu tồn tại.
   - Khi có cả Drive (`checkpoint-50`) và Local (`checkpoint-100`), hàm `get_resume_checkpoint("latest", ...)` ưu tiên Drive nhằm bảo đảm phục hồi khi máy ảo Colab bị sập và xóa sạch ổ cứng SSD tạm thời.

5. **Bộ Chuyển Mạch Mô Hình 3B vs 7B & Phần Cứng Trong Notebook 03**:
   - Chuyển mạch tầng phần cứng (Hardware Tiers):
     * VRAM >= 40.0 GB: `Pro+ A100`, micro-batch 4, $K=16$ rollouts, BF16.
     * VRAM >= 20.0 GB: `Pro L4`, micro-batch 2, $K=8$ rollouts, BF16.
     * VRAM < 20.0 GB hoặc CPU: `Free T4`, micro-batch 1, $K=8$ rollouts, QLoRA 4-bit, FP16.
   - Chuyển mạch mô hình 3B vs 7B:
     * Cập nhật chính xác `model_id` (`meta-llama/Llama-3.2-3B-Instruct` vs `Qwen/Qwen2.5-Coder-7B-Instruct`).
     * Cập nhật chính xác độ dài chuỗi tối đa (`max_seq_len`: 1024 trên T4 cho 3B, 2048 cho 7B).
     * **Thiếu sót**: Trong từ điển `MODEL_CONFIGS` của Notebook 03 không có khóa `target_modules` cho LoRA (khác với Notebook 02 có chỉ định rõ 7 module). Module `train_grpo_generator.py` cũng không cung cấp cấu hình LoRA mặc định.

6. **Triển Khai Bộ Kiểm Thử Đối Kháng**:
   - Đã tạo tệp kiểm thử độc lập: `tests/test_m3_stress.py` (8 bài kiểm thử stress test chuyên sâu).
   - Lệnh thực thi: `python3 -m unittest tests/test_m3_stress.py -v` $\rightarrow$ `Ran 8 tests in 0.154s` $\rightarrow$ `OK` (100% Passed).
   - Lệnh toàn dự án: `python3 -m unittest discover tests/ -v` $\rightarrow$ `Ran 110 tests in 0.863s` $\rightarrow$ `OK`.

---

## 2. Logic Chain (Chuỗi Lập Luận)

1. **Từ Observation 1**: worker_m3 tuyên bố trong handoff rằng toàn bộ code cells đã vượt qua `ast.parse` và 10 unit test trong `tests/test_notebooks.py`. Tuy nhiên, `ast.parse` chỉ kiểm tra tính hợp lệ về mặt ngữ pháp tĩnh (tokens & grammar của Python) mà hoàn toàn không thực hiện liên kết ký hiệu (name/symbol resolution) hoặc nạp module.
2. **Từ Observation 2**:
   - Khi người dùng hoặc nhà nghiên cứu mở và chạy notebook 01, 02, 03, hoặc 04 trên môi trường Google Colab thực tế, việc thực thi các ô huấn luyện chính (Cell 8 ở NB 02, Cell 8 ở NB 03, Cell 7 ở NB 04, Cell 2 ở NB 01) sẽ ngay lập tức phát sinh `ImportError` vì các hàm `generate_all`, `run_sft_training`, `run_grpo_training`, `run_decision_model_training` không hề tồn tại trong mã nguồn `src/` và `scripts/`.
   - Các hàm huấn luyện thực tế trong `src/training/` được thiết kế dưới tên `run_dry_run(...)` hoặc thực thi qua lệnh CLI `main()`.
   - Sự không nhất quán giữa giao diện lập trình được gọi trong notebook và giao diện thực tế của mã nguồn là lỗi nghiêm trọng làm gãy quy trình thực thi thực tế (contract drift / broken execution contract).
3. **Từ Observation 5**:
   - Mặc dù cấu hình phần cứng và ID mô hình được cập nhật đúng, việc thiếu định nghĩa `target_modules` trong cấu hình LoRA của Notebook 03 khiến mô hình 7B (Qwen2.5-Coder) và 3B không thể khởi tạo bộ điều hợp LoRA tùy chỉnh chuẩn xác nếu không dùng giá trị mặc định nội bộ.
4. **Kết Luận Logic**: Mã nguồn các notebook không thể được phê duyệt (`REJECT`) khi vẫn còn các lỗi import làm đứt gãy quá trình chạy thực tế.

---

## 3. Caveats (Lưu Ý & Giả Định)

- Thử nghiệm trên môi trường WSL cục bộ không có GPU CUDA vật lý; các kiểm thử phần cứng Colab (A100/L4/T4) được xác minh thông qua mô phỏng nhánh logic dựa trên thuộc tính `total_vram_gb`.
- Việc huấn luyện thực tế nhiều epoch với trọng số mô hình lớn yêu cầu kết nối mạng tải checkpoint HuggingFace có bản quyền và tài khoản Google Colab Pro.

---

## 4. Conclusion (Kết Luận & Yêu Cầu Thay Đổi)

### Phán Quyết: **REJECT (Yêu Cầu Thay Đổi)**

Yêu cầu `worker_m3` thực hiện các điều chỉnh sau để hoàn tất Milestone M3:
1. **Sửa các hàm import trong Notebook**:
   - Tại `notebooks/01_data_preparation_and_embedding.ipynb` (Cell 2): Thay `from scripts.make_mini_dataset import generate_all` bằng `from scripts.make_mini_dataset import main as generate_mini_dataset` (hoặc bổ sung hàm wrapper `generate_all` vào `scripts/make_mini_dataset.py`).
   - Tại `notebooks/02_sft_and_jepa_training_3b.ipynb` (Cell 8): Bổ sung hàm alias/wrapper `run_sft_training(...)` vào `src/training/train_sft_generator.py` (chuyển tiếp tới `run_dry_run` hoặc `main`), hoặc sửa notebook gọi đúng hàm hiện có.
   - Tại `notebooks/03_grpo_training_3b_and_7b.ipynb` (Cell 8): Bổ sung hàm alias/wrapper `run_grpo_training(...)` vào `src/training/train_grpo_generator.py`, hoặc sửa notebook gọi đúng hàm hiện có.
   - Tại `notebooks/04_system1_laya421m_training.ipynb` (Cell 7): Bổ sung hàm alias/wrapper `run_decision_model_training(...)` vào `src/training/train_decision_model.py`, hoặc sửa notebook gọi đúng hàm hiện có.
2. **Bổ sung `target_modules` trong Notebook 03**:
   - Khai báo rõ `target_modules` trong `MODEL_CONFIGS` của cả 3B và 7B (`["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`).

---

## 5. Verification Method (Phương Pháp Xác Minh Độc Lập)

Bất kỳ kiểm định viên nào cũng có thể kiểm chứng độc lập các phát hiện trên bằng các lệnh sau:

### Lệnh 1: Chạy bộ kiểm thử đối kháng M3 độc lập
```bash
python3 -m unittest tests/test_m3_stress.py -v
```
*Kết quả*: 8/8 tests pass, kiểm chứng toàn diện AST, Drive fallback, checkpoint sorting và phát hiện các import ảo.

### Lệnh 2: Xác minh trực tiếp 4 hàm import bị thiếu
```bash
python3 -c "
import importlib
checks = [
    ('scripts.make_mini_dataset', 'generate_all'),
    ('src.training.train_sft_generator', 'run_sft_training'),
    ('src.training.train_grpo_generator', 'run_grpo_training'),
    ('src.training.train_decision_model', 'run_decision_model_training')
]
for mod, func in checks:
    m = importlib.import_module(mod)
    assert hasattr(m, func), f'CRITICAL: {func} missing from {mod}!'
"
```
*Kết quả hiện tại*: Phát sinh lỗi `AssertionError: CRITICAL: generate_all missing from scripts.make_mini_dataset!` minh chứng rõ ràng lỗi import chưa giải quyết.

### Lệnh 3: Chạy toàn bộ 110 bài kiểm thử của dự án
```bash
python3 -m unittest discover tests/ -v
```
*Kết quả*: 110 tests passed.
