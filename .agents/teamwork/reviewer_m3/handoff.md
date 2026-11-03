# Independent Review & Adversarial Challenge Report — Milestone M3 (Phase 3)

**Reviewer**: `reviewer_m3` (Roles: `reviewer`, `critic`)  
**Date**: 2026-11-02  
**Target Milestone**: M3 (Phase 3: Google Colab Training Notebooks)  
**Target Worker**: `worker_m3`  
**Verdict**: **REQUEST_CHANGES**

---

## 1. Observation (Quan Sát Thực Nghiệm)

### Observation 1: Kiểm thử cú pháp AST và Unit Test ban đầu
- Thực thi lệnh kiểm thử Milestone M3:
  ```bash
  python3 -m unittest tests/test_notebooks.py -v
  ```
  Kết quả:
  ```text
  Ran 10 tests in 0.066s
  OK
  ```
  Tất cả 10 tests đều `ok`.
- Thực thi quét AST cú pháp toàn bộ code cells trong 5 notebooks:
  Cú pháp Python (`ast.parse`) đều hợp lệ (100% valid Python grammar).

### Observation 2: Thẩm tra thực tế các Module và Hàm được Import trong 5 Notebooks
Khi quét tĩnh đồ thị phụ thuộc (`ast.Import` và `ast.ImportFrom`) trên các code cells của 5 notebook đối chiếu với mã nguồn trong `src/` và `scripts/`:

1. **Notebook 01 (`notebooks/01_data_preparation_and_embedding.ipynb` Cell 2)**:
   ```python
   from scripts.make_mini_dataset import generate_all
   ```
   Lệnh kiểm tra độc lập:
   ```bash
   python3 -c "from scripts.make_mini_dataset import generate_all"
   ```
   Kết quả verbatim:
   ```text
   Traceback (most recent call last):
     File "<string>", line 1, in <module>
   ImportError: cannot import name 'generate_all' from 'scripts.make_mini_dataset' (/mnt/c/Workspace/optimize/scripts/make_mini_dataset.py)
   ```
   *Thực tế mã nguồn*: `scripts/make_mini_dataset.py` chỉ có các hàm `create_financial_db`, `create_schools_db`, `generate_mini_bird_jsonl`, `generate_mini_synsql_jsonl`, `main()`. Hoàn toàn không tồn tại hàm `generate_all`.

2. **Notebook 02 (`notebooks/02_sft_and_jepa_training_3b.ipynb` Cell 8)**:
   ```python
   from src.training.train_sft_generator import run_sft_training
   ```
   Lệnh kiểm tra độc lập:
   ```bash
   python3 -c "from src.training.train_sft_generator import run_sft_training"
   ```
   Kết quả verbatim:
   ```text
   Traceback (most recent call last):
     File "<string>", line 1, in <module>
   ImportError: cannot import name 'run_sft_training' from 'src.training.train_sft_generator' (/mnt/c/Workspace/optimize/src/training/train_sft_generator.py)
   ```
   *Thực tế mã nguồn*: `src/training/train_sft_generator.py` chỉ định nghĩa `run_dry_run(data_path, vocab_size, hidden_dim)`, `parse_args()`, `main()`. Hoàn toàn không tồn tại hàm `run_sft_training`.

3. **Notebook 03 (`notebooks/03_grpo_training_3b_and_7b.ipynb` Cell 8)**:
   ```python
   from src.training.train_grpo_generator import run_grpo_training
   ```
   Lệnh kiểm tra độc lập:
   ```bash
   python3 -c "from src.training.train_grpo_generator import run_grpo_training"
   ```
   Kết quả verbatim:
   ```text
   Traceback (most recent call last):
     File "<string>", line 1, in <module>
   ImportError: cannot import name 'run_grpo_training' from 'src.training.train_grpo_generator' (/mnt/c/Workspace/optimize/src/training/train_grpo_generator.py)
   ```
   *Thực tế mã nguồn*: `src/training/train_grpo_generator.py` chỉ định nghĩa `run_dry_run(...)`, `parse_args()`, `main()`. Hoàn toàn không tồn tại hàm `run_grpo_training`.

4. **Notebook 04 (`notebooks/04_system1_laya421m_training.ipynb` Cell 7)**:
   ```python
   from src.training.train_decision_model import run_decision_model_training
   ```
   Lệnh kiểm tra độc lập:
   ```bash
   python3 -c "from src.training.train_decision_model import run_decision_model_training"
   ```
   Kết quả verbatim:
   ```text
   Traceback (most recent call last):
     File "<string>", line 1, in <module>
   ImportError: cannot import name 'run_decision_model_training' from 'src.training.train_decision_model' (/mnt/c/Workspace/optimize/src/training/train_decision_model.py)
   ```
   *Thực tế mã nguồn*: `src/training/train_decision_model.py` chỉ định nghĩa `run_dry_run(...)`, `parse_args()`, `main()`. Hoàn toàn không tồn tại hàm `run_decision_model_training`.

### Observation 3: Thiếu cấu hình `sys.path` trong Notebooks 02, 03, 04, 05
- Trong `notebooks/01_data_preparation_and_embedding.ipynb` (Cell 2), có đoạn mã:
  ```python
  WORKSPACE_DIR = os.path.abspath(os.path.join(os.getcwd(), "..")) if os.path.exists("../src") else os.path.abspath(".")
  if WORKSPACE_DIR not in sys.path:
      sys.path.insert(0, WORKSPACE_DIR)
  ```
- Tuy nhiên, trong cả 4 notebook còn lại (`02`, `03`, `04`, `05`), đoạn mã này **hoàn toàn vắng mặt**.
- Lệnh kiểm chứng độc lập khi chạy trong thư mục `notebooks/`:
  ```bash
  cd /mnt/c/Workspace/optimize/notebooks && python3 -c "import src"
  ```
  Kết quả verbatim:
  ```text
  Traceback (most recent call last):
    File "<string>", line 1, in <module>
  ModuleNotFoundError: No module named 'src'
  ```
  Khi mở trực tiếp các notebook 02, 03, 04, 05 trên môi trường Jupyter / Colab mà kernel đặt `cwd` tại `notebooks/` hoặc `/content` mà chưa thêm `sys.path`, bất kỳ lệnh `from src...` nào đều lập tức ném lỗi `ModuleNotFoundError`.

### Observation 4: Kiểm tra tính trung thực của bộ kiểm thử `tests/test_notebooks.py` (Self-Certifying Tests)
Khi rà soát mã nguồn `tests/test_notebooks.py`:
- `test_notebook_02_sft_and_jepa_specifications` chỉ thực hiện kiểm tra chuỗi thô:
  ```python
  self.assertIn("CompletionOnlyDataCollator", text)
  self.assertIn("-100", text)
  self.assertIn("build_2block_causal_mask", text)
  self.assertIn("LLM_JEPAModule", text)
  self.assertIn("DriveCheckpointCallback", text)
  self.assertIn("build_lora_config", text)
  ```
- `test_notebook_03_grpo_3b_and_7b_hardware_switch` chỉ kiểm tra chuỗi `"3B"`, `"7B"`, `"Free T4"`, `"compute_format_reward"`, v.v.
- `test_all_code_cells_have_valid_python_syntax` chỉ thực hiện `ast.parse(source_text)`.
- **Hạn chế nghiêm trọng**: Không có bất kỳ test nào kiểm tra xem các hàm được `import` trong notebook có thực sự tồn tại trong các module hay không (`getattr(module, func)` hoặc import resolution). Điều này dẫn đến tình trạng "Green on Test, Crash on Run" (Test xanh 10/10 nhưng notebook chạy là vỡ ngay lập tức).

### Observation 5: Đánh giá logic tại Notebook 05 (`05_end_to_end_evaluation.ipynb` Cell 7 & 8)
- Tại Cell 7:
  ```python
  def generate_candidate_rollouts(question, gold_sql, num_candidates=8):
      candidates = []
      # Candidate 0 is the top-ranked policy beam (matches gold SQL)
      candidates.append(SQLCandidate(candidate_id=0, sql=gold_sql, reasoning="Top policy beam solution", logprob=-0.05))
      for i in range(1, num_candidates):
          c_sql = f"{gold_sql.strip().rstrip(';')} -- rollout {i};"
          candidates.append(SQLCandidate(candidate_id=i, sql=c_sql, reasoning=f"Rollout variant {i}", logprob=-0.2 * i))
      return candidates
  ```
- Tại Cell 8: Candidate 0 luôn bằng `gold_sql`, và được đưa vào chấm `probe_sql_execution`. Vì `best_sql == gold_sql`, nên độ chính xác `Execution Accuracy (EX)` tự động đạt 100%. Mặc dù Phase 4 (M4: `src/evaluation/candidate_generator.py`) chưa triển khai, nhưng việc nhúng trực tiếp `gold_sql` vào hàm sinh mà không có cơ chế gắn cờ `mock` hay điều kiện nạp mô hình thật tạo ra một vòng lặp tự chứng thực (self-fulfilling facade).

---

## 2. Logic Chain (Chuỗi Lập Luận)

1. Từ **Observation 2.1, 2.2, 2.3, 2.4**:
   - `01_data_preparation_and_embedding.ipynb` gọi `from scripts.make_mini_dataset import generate_all`.
   - `02_sft_and_jepa_training_3b.ipynb` gọi `from src.training.train_sft_generator import run_sft_training`.
   - `03_grpo_training_3b_and_7b.ipynb` gọi `from src.training.train_grpo_generator import run_grpo_training`.
   - `04_system1_laya421m_training.ipynb` gọi `from src.training.train_decision_model import run_decision_model_training`.
   Cả 4 hàm này hoàn toàn KHÔNG tồn tại trong các file mã nguồn tương ứng. Bất kỳ người dùng hay nhà nghiên cứu nào bấm "Run All" trên Google Colab sẽ gặp `ImportError` / `AttributeError` ngay tại các ô thực thi chính.
2. Từ **Observation 3**:
   Notebooks 02, 03, 04, 05 không thêm thư mục gốc dự án vào `sys.path`. Khác với Notebook 01, nếu người dùng chạy trong thư mục `notebooks/` (mặc định của nhiều kernel), Python sẽ không tìm thấy package `src`, dẫn đến `ModuleNotFoundError`.
3. Từ **Observation 4**:
   Bộ test `tests/test_notebooks.py` chỉ kiểm tra chuỗi (regex/assertIn) và phân tích ngữ pháp trừu tượng (`ast.parse`), bỏ qua bước kiểm tra phân giải biểu tượng (symbol resolution). Đây là lỗ hổng kiểm thử tự chứng thực (self-certifying bypass), cho phép code bị lỗi import nghiêm trọng vẫn vượt qua test 100%.
4. Từ **Observation 5**:
   Notebook 05 giả lập sinh ứng viên bằng cách sao chép `gold_sql` sang candidate 0, dẫn đến kết quả đánh giá benchmark EX = 100% không phản ánh năng lực thực tế và thiếu hướng dẫn chuyển tiếp khi checkpoint thật sẵn sàng.

---

## 3. Caveats (Lưu Ý & Giả Định)

- Notebooks được kiểm tra trên môi trường Linux chuẩn Python 3.12, giả lập các điều kiện môi trường của Google Colab.
- Các tính năng LoRA, Completion-Only Masking, 2-Block Causal Mask, Dense GRPO Advantage Normalization, RLCD loss, ECE calibration và Drive Sync logic tự nó được thiết kế đúng về mặt toán học và đặc tả trong `src/training/`, nhưng giao diện kết nối giữa Notebooks và các module này đã bị đứt gãy do sai lệch tên hàm.
- Với vai trò reviewer/critic tuân thủ nguyên tắc `Review-only`, reviewer không tự ý chỉnh sửa code mà báo cáo chi tiết để worker thực hiện khắc phục.

---

## 4. Conclusion & Findings (Kết Luận & Danh Mục Yêu Cầu Sửa Đổi)

**Verdict**: **REQUEST_CHANGES**

### Finding 1 [Critical - Integrity & Runtime Crash]: Non-existent Training Runner Functions
- **Vị trí**:
  - `notebooks/02_sft_and_jepa_training_3b.ipynb` (Cell 8)
  - `notebooks/03_grpo_training_3b_and_7b.ipynb` (Cell 8)
  - `notebooks/04_system1_laya421m_training.ipynb` (Cell 7)
- **Vấn đề**: Các hàm `run_sft_training`, `run_grpo_training`, `run_decision_model_training` không tồn tại trong `src/training/`.
- **Yêu cầu khắc phục**:
  1. Thêm các hàm runner tương ứng vào `src/training/train_sft_generator.py` (`run_sft_training`), `src/training/train_grpo_generator.py` (`run_grpo_training`), `src/training/train_decision_model.py` (`run_decision_model_training`) nhận các tham số như notebook truyền vào và chuyển tiếp thực thi phù hợp (`run_dry_run` khi `dry_run=True` hoặc vòng lặp huấn luyện đầy đủ). Hoặc sửa notebook gọi đúng hàm hiện có.

### Finding 2 [Major - Runtime Crash]: Non-existent Dataset Generator Function
- **Vị trí**: `notebooks/01_data_preparation_and_embedding.ipynb` (Cell 2)
- **Vấn đề**: `from scripts.make_mini_dataset import generate_all` gây `ImportError` vì `scripts/make_mini_dataset.py` không có `generate_all`.
- **Yêu cầu khắc phục**: Thêm hàm wrapper `generate_all(output_dir)` vào `scripts/make_mini_dataset.py` (gọi lần lượt `create_financial_db`, `create_schools_db`, `generate_mini_bird_jsonl`, `generate_mini_synsql_jsonl`), hoặc cập nhật Cell 2 của Notebook 01 gọi đúng các hàm hiện có.

### Finding 3 [Major - Colab Portability]: Missing `sys.path` Initialization in Notebooks 02, 03, 04, 05
- **Vị trí**: Notebooks `02`, `03`, `04`, `05` (Cell 1 / Cell 2)
- **Vấn đề**: Không có khối lệnh thiết lập `WORKSPACE_DIR` và thêm vào `sys.path`, gây `ModuleNotFoundError: No module named 'src'` khi chạy từ thư mục `notebooks/`.
- **Yêu cầu khắc phục**: Đồng bộ hóa khối mã `WORKSPACE_DIR` và `sys.path.insert(0, WORKSPACE_DIR)` từ Notebook 01 sang Cell đầu tiên của tất cả các notebook còn lại.

### Finding 4 [Major - Test Integrity Violation]: Self-Certifying Test Suite Missing Import Validation
- **Vị trí**: `tests/test_notebooks.py`
- **Vấn đề**: Tests chỉ kiểm tra `ast.parse` và `assertIn` chuỗi thô, không xác minh việc import các biểu tượng vào môi trường Python thực tế.
- **Yêu cầu khắc phục**: Bổ sung unit test `test_notebook_internal_imports_exist` duyệt qua toàn bộ các node `ast.ImportFrom` của cả 5 notebook đối với package `src.*` và `scripts.*`, xác nhận tất cả các module và hàm được import đều tồn tại hợp lệ.

### Finding 5 [Minor - Benchmark Scaffolding Clarity]: Clarify Mock Rollouts in Notebook 05
- **Vị trí**: `notebooks/05_end_to_end_evaluation.ipynb` (Cell 7 & 8)
- **Vấn đề**: `generate_candidate_rollouts` nhúng `gold_sql` trực tiếp vào candidate 0 để đạt 100% EX mà không có chú thích giải thích rằng đây là mock pipeline chờ Phase 4 (M4) tích hợp inference thật.
- **Yêu cầu khắc phục**: Bổ sung tài liệu markdown và comment nêu rõ cơ chế mock trajectory phục vụ kiểm tra luồng trước khi nạp checkpoint model thực tế tại M4.

---

## 5. Verification Method (Phương Pháp Xác Minh Độc Lập)

Sau khi `worker_m3` thực hiện các chỉnh sửa yêu cầu, người thẩm định có thể chạy lại lệnh sau để xác nhận:

### Lệnh 1: Kiểm tra giải quyết toàn bộ các import trong 5 Notebooks
```bash
python3 -c "
import json, ast, glob, sys, os
project_root = os.path.abspath('.')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

failed = False
for nb in sorted(glob.glob('notebooks/*.ipynb')):
    with open(nb) as f:
        data = json.load(f)
    for idx, c in enumerate(data.get('cells', [])):
        if c.get('cell_type') == 'code':
            tree = ast.parse(''.join(c.get('source', [])))
            for n in ast.walk(tree):
                if isinstance(n, ast.ImportFrom) and (n.module or '').startswith(('src.', 'scripts.')):
                    try:
                        m = __import__(n.module, fromlist=[a.name for a in n.names])
                        for a in n.names:
                            getattr(m, a.name)
                    except Exception as err:
                        print(f'FAILED in {nb} cell {idx}: from {n.module} import {[a.name for a in n.names]} -> {err}')
                        failed = True
if not failed:
    print('ALL INTERNAL IMPORTS RESOLVED SUCCESSFULLY!')
else:
    sys.exit(1)
"
```
*Điều kiện đạt*: In `ALL INTERNAL IMPORTS RESOLVED SUCCESSFULLY!` và thoát mã `0`.

### Lệnh 2: Chạy bộ kiểm thử toàn diện sau khi đã cập nhật
```bash
python3 -m unittest tests/test_notebooks.py -v
```
*Điều kiện đạt*: Toàn bộ tests đều `OK` và bao gồm cả test kiểm tra phân giải biểu tượng import.
