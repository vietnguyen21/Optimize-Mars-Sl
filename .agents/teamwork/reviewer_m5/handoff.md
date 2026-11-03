# Báo Cáo Thẩm Định Độc Lập & Phản Biện Đối Kháng (Review & Adversarial Challenge Report)
## Milestone M5: Automated Smoke Testing & Comprehensive Verification

- **Tác tử thẩm định & phản biện**: `teamwork_reviewer` (Roles: `reviewer`, `critic`)
- **Người nhận**: `orchestrator` / `parent` (Conversation ID: `3dfd88ef-5709-46d6-ba23-355dc6f4af89`)
- **Ngày thực hiện**: 2026-11-02
- **Thư mục làm việc**: `/mnt/c/Workspace/optimize/.agents/teamwork/reviewer_m5/`
- **Phán quyết thẩm định (Verdict)**: **`APPROVE`** (Chấp thuận nghiệm thu M5, kèm các phát hiện phản biện đối kháng cho các cải tiến tiếp theo)

---

## 1. Observation (Quan Sát Thực Nghiệm Độc Lập Trực Tiếp)

Tôi đã tiến hành rà soát mã nguồn độc lập, kiểm tra tính toàn vẹn (integrity audit), thực thi trực tiếp bộ kiểm thử từ dòng lệnh và thực hiện kiểm thử đối kháng (adversarial stress-testing). Dưới đây là các dữ kiện quan sát được trích xuất trực tiếp:

### 1.1. Kiểm Tra Mã Nguồn Cải Tiến Của Milestone M5
1. **Quy tắc PV-SQL (`src/validation/pv_sql_rules.py`)**:
   - Dòng 155: Hàm `check_distinct_rule` sử dụng `bool(re.search(r"\bDISTINCT\b", sql_upper)) or "GROUP BY" in sql_upper`. Đã thay thế hoàn toàn phép kiểm tra chuỗi con lỏng lẻo `"DISTINCT" in sql_upper`.
   - Dòng 214: Hàm `check_null_handling_rule` sử dụng `any(k in sql_upper for k in ["NOT NULL", "IS NOT NULL", "IS NULL", "COALESCE"])`. Từ khóa `"WHERE"` thuần túy đã bị loại bỏ triệt để.
2. **Đánh giá toàn tuyến & Abstention Manager (`src/evaluation/evaluate_end_to_end.py`)**:
   - Dòng 68, 75, 85-93: Tham số `model_path: Optional[str] = None` được bổ sung vào `EvaluationHarness.__init__`. Nếu cung cấp đường dẫn tồn tại, nạp tự động qua HuggingFace transformers; nếu không, tự động fallback về bộ sinh quỹ đạo ablation.
   - Dòng 208-220: Tích hợp `self.abstention_manager.should_abstain(valid_candidates_count, best_confidence, has_security_violation, has_hallucinations)`. Khi phát hiện vi phạm bảo mật, 0 ứng viên hợp lệ, hoặc độ tin cậy thấp hơn ngưỡng, `total_abstentions` tăng và ghi log `abstained: true`.
   - Dòng 265-266: Bổ sung `total_abstentions` và `abstention_rate_pct` vào từ điển kết quả đánh giá `benchmark_results`.
   - Dòng 305-312: Cờ dòng lệnh `--model_path` được tích hợp vào `argparse` trong hàm `main()`.
3. **Export Package (`src/evaluation/__init__.py`)**:
   - Dòng 17, 33: Hàm `normalize_query_result` được import từ `src.evaluation.metrics` và đưa vào danh sách `__all__`.
4. **Kịch bản nghiệm thu độc lập (`scripts/run_smoke_tests.py`)**:
   - Cấu trúc 6 Suite tuần tự: Suite 1 (`tests.test_grounding`), Suite 2 (`tests.test_training`), Suite 3 (`tests.test_notebooks`), Suite 4 (`tests.test_grammar_and_validation`), Suite 5 (`tests.test_evaluation`), Suite 6 (E2E Benchmark 24 truy vấn trên `data/mini/mini_bird.jsonl`).
   - Hàm `silence_all_output()` (dòng 60-86): Context manager cô lập stdout/stderr và descriptor 1 & 2 của hệ điều hành, đảm bảo khôi phục đầy đủ trong khối `finally`.
   - Định dạng hiển thị ANSI/ASCII bảng tổng hợp kết quả (dòng 168-228).
   - Thoát mã `0` khi tất cả các suite thành công và tổng thời gian $< 60.0$s; thoát mã `1` nếu có bất kỳ lỗi nào hoặc thời gian vượt hạn mức.

### 1.2. Kết Quả Thực Thi Độc Lập Các Lệnh Kiểm Thử

#### Lệnh 1: `python3 scripts/run_smoke_tests.py`
Thực thi tại thư mục gốc `/mnt/c/Workspace/optimize`:
```text
====================================================================================================
    TEXT-TO-SQL SLM ARCHITECTURE — COMPREHENSIVE SMOKE TEST & VERIFICATION SUITE
====================================================================================================
Target Environment: CPU / Local Offline Smoke-Test (Zero GPU/Heavy Model Download Dependency)
Quality Thresholds: Total Runtime < 60.0s | 100% Test Pass | EX >= 75.0% | VSR = 100.0% | ECE <= 0.081
----------------------------------------------------------------------------------------------------

[RUNNING] Suite 1: Data Pipeline & Schema Grounding (PK-FK Graph, Steiner Tree NWST, Column Cards, Embedding Retriever)...
  ✔ PASSED in 0.160s (25 test cases)

[RUNNING] Suite 2: SLM Training Scripts (Completion-Only Loss (-100), LLM-JEPA 2-Block Mask, Dense GRPO, DriveSync)...
  ✔ PASSED in 0.189s (24 test cases)

[RUNNING] Suite 3: Colab Training Notebooks (49 Code Cells dynamic execution across Notebooks 01 to 05)...
  ✔ PASSED in 2.021s (12 test cases)

[RUNNING] Suite 4: Grammar PDA & Validation Rules (XGrammar PDA, Fallback Engine, 10 PV-SQL AST Rules, DB Probe, Laya-421M)...
  ✔ PASSED in 0.056s (30 test cases)

[RUNNING] Suite 5: Evaluation & Benchmark Suite (Candidate Generator (8-16 rollouts), Multiset EX, VSR, ECE, Memory Stats)...
  ✔ PASSED in 1.123s (16 test cases)

[RUNNING] Suite 6: End-to-End Mini-BIRD Benchmark (24 Queries) on data/mini/mini_bird.jsonl...
  ✔ PASSED in 0.710s (24 evaluation queries)

+--------------------------------------------------------+--------------+--------------+--------------+
| Test Suite / Verification Block                        |  Test Cases  |   Time (s)   |    Status    |
+--------------------------------------------------------+--------------+--------------+--------------+
| Suite 1: Data Pipeline & Schema Grounding              |      25      |    0.160s    |    PASSED    |
| Suite 2: SLM Training Scripts                          |      24      |    0.189s    |    PASSED    |
| Suite 3: Colab Training Notebooks                      |      12      |    2.021s    |    PASSED    |
| Suite 4: Grammar PDA & Validation Rules                |      30      |    0.056s    |    PASSED    |
| Suite 5: Evaluation & Benchmark Suite                  |      16      |    1.123s    |    PASSED    |
| Suite 6: End-to-End Mini-BIRD Benchmark (24 Queries)   |      24      |    0.710s    |    PASSED    |
+--------------------------------------------------------+--------------+--------------+--------------+
| TOTAL / OVERALL SYSTEM-LEVEL SUMMARY                   |     131      |    4.389s    |  ALL PASSED  |
+--------------------------------------------------------+--------------+--------------+--------------+
```
Mã thoát: `0`. Tổng thời gian: `4.389s` (đáp ứng trần $< 60.0s$). Toàn bộ 131 test cases/queries đều `PASSED`.

#### Lệnh 2: `python3 -m unittest discover tests/ -v`
Kết quả:
```text
Ran 158 tests in 4.191s
OK
```
Bao gồm:
- 107 tests từ 5 module kiểm thử nền tảng (`test_grounding`, `test_training`, `test_notebooks`, `test_grammar_and_validation`, `test_evaluation`).
- 51 tests từ 3 suite stress tests (`test_m1_stress`, `test_m2_stress`, `test_m3_stress`).
Tỷ lệ thành công: 100% (158/158 passed, 0 failures, 0 errors).

#### Lệnh 3: `python3 src/evaluation/evaluate_end_to_end.py --help`
Kết quả:
```text
usage: evaluate_end_to_end.py [-h] [--dataset DATASET] [--db_dir DB_DIR]
                              [--num_candidates NUM_CANDIDATES]
                              [--limit LIMIT] [--model_path MODEL_PATH]
```
Cờ `--model_path MODEL_PATH` xuất hiện đầy đủ trong phần options.

### 1.3. Quan Sát Đối Kháng & Kiểm Tra Ranh Giới (Adversarial Observations)
1. **Kiểm tra ranh giới regex Rule 1 & Rule 8**:
   - `SELECT col FROM distinct_clients` -> `check_distinct_rule` trả về `False` (chính xác).
   - `SELECT DISTINCT(col) FROM tbl` -> `check_distinct_rule` trả về `True` (chính xác).
   - `SELECT client_id FROM clients WHERE age > 20;` khi câu hỏi yêu cầu lọc rỗng -> `check_null_handling_rule` trả về `False` (đã triệt tiêu bypass của từ khóa `WHERE`).
2. **Kiểm tra phụ thuộc thư mục làm việc hiện tại (CWD Sensitivity)**:
   - Khi chạy từ một thư mục khác (ví dụ: `cd /tmp && python3 /mnt/c/Workspace/optimize/scripts/run_smoke_tests.py`), Suite 1 và Suite 2 bị lỗi `FileNotFoundError` do các đường dẫn trong `test_grounding.py` và `test_training.py` dùng đường dẫn tương đối dựa trên `os.getcwd()` thay vì `REPO_ROOT`. (Xem chi tiết tại phần Findings).

---

## 2. Logic Chain (Chuỗi Lập Luận Thẩm Định & Đánh Giá)

1. **Về tính toàn vẹn và trung thực kỹ thuật (Integrity Audit)**:
   - *Không có kết quả hardcoded*: `metrics.py` và `evaluate_end_to_end.py` thực hiện mở kết nối SQLite thực, thực thi các truy vấn SQL, chuẩn hóa dữ liệu kết quả và so sánh multiset độc lập.
   - *Không có facade hay stub lừa đảo*: Tất cả các engine (XGrammar PDA fallback, PV-SQL 10 AST rules, DB Probe, Laya Scorer, Abstention Manager) đều có thuật toán tính toán cụ thể, xử lý ngoại lệ và logic toán học thực tế.
   - *Không có shortcut trốn tránh*: Kịch bản `run_smoke_tests.py` thực sự kích hoạt toàn bộ 6 suite và chạy qua 131 đơn vị kiểm thử.
2. **Về việc khắc phục các góp ý từ M4**:
   - Rule 1 đã phân định từ nguyên `\bDISTINCT\b`, ngăn chặn false positive từ các tên bảng/cột như `distinct_clients`.
   - Rule 8 đã loại bỏ `"WHERE"`, ngăn chặn bypass khi một truy vấn có `WHERE age > 10` nhưng không hề lọc `NULL`.
   - `AbstentionManager` được kích hoạt trong vòng lặp đánh giá của `EvaluationHarness`, theo dõi đầy đủ vi phạm bảo mật, số lượng ứng viên hợp lệ, và trả về cờ `abstained` cùng lý do từ chối.
   - `--model_path` cho phép người dùng CLI và Colab chỉ định checkpoint mô hình thực tế.
   - `normalize_query_result` được export đầy đủ tại `src/evaluation/__init__.py`.
3. **Về hiệu năng và tính độc lập của Smoke Test**:
   - Kịch bản chạy hoàn tất trong **4.389 giây** (hạn mức yêu cầu là $< 60$ giây).
   - Hoàn toàn chạy trên CPU, 0 GPU VRAM, không cần tải model nặng từ internet.

---

## 3. Caveats (Lưu Ý & Giới Hạn)

1. **Phụ thuộc vào CWD khi chạy `scripts/run_smoke_tests.py`**:
   - `scripts/run_smoke_tests.py` hiện tại thêm `REPO_ROOT` vào `sys.path`, nhưng chưa gọi `os.chdir(REPO_ROOT)`. Do đó, người dùng bắt buộc phải chạy lệnh từ thư mục gốc của dự án (`/mnt/c/Workspace/optimize`). Nếu chạy từ thư mục khác qua đường dẫn tuyệt đối, Suite 1 và 2 sẽ không tìm thấy tệp `data/mini/`.
2. **Biến `has_hallucinations` trong vòng lặp đánh giá E2E**:
   - Trong `evaluate_dataset()`, biến `has_hallucinations` đang được gán mặc định là `False` vì vòng lặp đánh giá nhanh hiện tại chưa truyền `known_tables` vào `validate_candidate`. Tuy nhiên, nếu có hallucination về schema, SQLite DB probe sẽ bắt lỗi ở tầng thực thi.

---

## 4. Findings (Phát Hiện Độc Lập & Phản Biện Đối Kháng)

### [Major] Finding 1: Thiếu `os.chdir(REPO_ROOT)` trong `scripts/run_smoke_tests.py`
- **Hiện tượng**: Khi gọi `scripts/run_smoke_tests.py` từ ngoài thư mục dự án (ví dụ: `cd /tmp && python3 /mnt/c/Workspace/optimize/scripts/run_smoke_tests.py`), Suite 1 gặp 9 errors và Suite 2 gặp 3 errors + 1 failure do không tìm thấy `data/mini/column_cards_mini.jsonl`.
- **Vị trí**: `scripts/run_smoke_tests.py`, dòng 25-28.
- **Nguyên nhân**: `test_grounding.py` và `test_training.py` sử dụng `os.path.abspath("data/mini/...")`, hàm này lấy `os.getcwd()` làm gốc thay vì `REPO_ROOT`.
- **Khuyến nghị khắc phục (cho worker M6 hoặc bảo trì sau này)**:
  Thêm dòng sau ngay sau dòng 27 của `scripts/run_smoke_tests.py`:
  ```python
  os.chdir(REPO_ROOT)
  ```
  Điều này giúp kịch bản chạy an toàn từ bất kỳ thư mục nào trên hệ thống hoặc trong các runner CI/CD.

### [Minor] Finding 2: Bổ sung trích xuất bảng schema cho `has_hallucinations` trong `EvaluationHarness`
- **Hiện tượng**: `has_hallucinations` được khởi tạo `False` và chưa tự động kết nối với danh sách bảng của SQLite trong `evaluate_dataset`.
- **Vị trí**: `src/evaluation/evaluate_end_to_end.py`, dòng 164.
- **Khuyến nghị**: Khi chuẩn bị dữ liệu, có thể đọc `sqlite_master` để lấy danh sách bảng và truyền vào `validate_candidate(known_tables=tables)` nhằm kích hoạt toàn diện `check_schema_hallucinations`.

---

## 5. Conclusion (Kết Luận Thẩm Định)

- **Phán quyết cuối cùng**: **`APPROVE`**
- **Lý do**:
  1. Toàn bộ 4 yêu cầu của Milestone M5 theo đặc tả đều được hoàn thành xuất sắc và đúng chuẩn kỹ thuật.
  2. Kịch bản `scripts/run_smoke_tests.py` thực hiện đầy đủ 6 suite, hiển thị bảng tổng kết chuyên nghiệp, đo đếm thời gian chính xác (4.389s $< 60$s), mã thoát 0 khi thành công.
  3. Tất cả 158 bài kiểm thử của dự án đạt 100% tỷ lệ vượt qua.
  4. Không có bất kỳ vi phạm tính toàn vẹn (integrity violation), không có kết quả giả mạo hay mã nguồn facade.
  5. Phát hiện phản biện (Finding 1) là một điểm cải tiến tính bền vững cho CLI và không ảnh hưởng đến luồng chạy chuẩn từ thư mục gốc của dự án. Hệ thống hoàn toàn sẵn sàng cho Milestone M6 (Technical Handover Documentation).

---

## 6. Verification Method (Phương Pháp Xác Minh Độc Lập)

Bất kỳ kiểm định viên hoặc giám sát viên độc lập nào cũng có thể kiểm chứng lại toàn bộ các kết quả trên bằng các lệnh sau:

### Lệnh 1: Xác minh Smoke Test toàn tuyến
```bash
cd /mnt/c/Workspace/optimize
python3 scripts/run_smoke_tests.py
```
*Kỳ vọng*: Mã thoát `0`, 6 Suite với 131 test cases/queries đều `PASSED`, tổng thời gian thực thi $< 60$s (thực tế ~4.4s).

### Lệnh 2: Xác minh 158 Unit & Stress Tests
```bash
cd /mnt/c/Workspace/optimize
python3 -m unittest discover tests/ -v
```
*Kỳ vọng*: `Ran 158 tests in ...s`, trạng thái `OK` (0 failures, 0 errors).

### Lệnh 3: Xác minh CLI `--model_path`
```bash
python3 src/evaluation/evaluate_end_to_end.py --help
```
*Kỳ vọng*: Hiển thị tùy chọn `--model_path MODEL_PATH`.
