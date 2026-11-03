# Báo Cáo Thẩm Định Pháp Y Độc Lập (Forensic Audit Report) — Milestone M2 (Phase 2)

**Thẩm định viên (Auditor)**: `auditor_m2` (`[FORENSIC-AUDITOR]` / `teamwork_auditor`)  
**Ngày thực hiện**: 2026-11-02  
**Thư mục làm việc**: `/mnt/c/Workspace/optimize/.agents/teamwork/auditor_m2/`  
**Đối tượng thẩm định**: Mã nguồn và kiểm thử Milestone M2 (`src/training/` và `tests/test_training.py`)  
**Chế độ liêm chính (Integrity Mode)**: `development` (theo `ORIGINAL_REQUEST.md`)  
**Phán quyết liêm chính (Integrity Verdict)**: **CLEAN (HỢP LỆ — KHÔNG VI PHẠM LIÊM CHÍNH)**

---

## 1. Observation (Quan Sát Thực Nghiệm Trực Tiếp)

Thẩm định viên đã tiến hành kiểm tra mã nguồn tĩnh, quét các tệp tiền chế, và chạy các probe động kiểm chứng tính toán thực tế. Dưới đây là bằng chứng thực nghiệm:

### 1.1. Quét tệp kết quả tiền chế (Pre-populated Artifact Scan)
- Lệnh:
  ```bash
  find /mnt/c/Workspace/optimize -maxdepth 3 -name '*.log' -o -name '*result*' -o -name '*output*'
  ```
- Kết quả: Không tìm thấy bất kỳ tệp log, kết quả hoặc chứng thực được ghi sẵn nào từ trước. Mã thoát: `0`.

### 1.2. Thẩm định mô-đun LLM-JEPA (`src/training/jepa_module.py`)
- **Mặt nạ Causal 2 khối (`build_2block_causal_mask`)**:
  - Tọa độ `i < T` và `j >= T` (Khối 1 nhìn sang Khối 2): Tất cả các giá trị đều là `-inf` (PyTorch) hoặc `-1e9` (Python fallback), thỏa mãn nghiêm ngặt điều kiện $< -10^8$.
  - Tọa độ `i < T` và `j <= i` (Khối 1 tự chú ý nhân quả): Bằng `0.0`.
  - Tọa độ `i >= T` và `j < T` (Khối 2 quan sát toàn bộ Khối 1): Bằng `0.0`.
  - Tọa độ `i >= T` và `T <= j <= i` (Khối 2 tự chú ý nhân quả): Bằng `0.0`.
  - Đã probe trên các kích thước: $(T=2, C=3), (T=7, C=5), (T=1, C=1)$. 100% các ô đều thỏa mãn điều kiện nhân quả không rò rỉ.
- **Khoảng cách Cosine (`compute_cosine_distance`)**:
  - Kiểm tra độ nhạy động qua góc quay vector từ $0^\circ$ đến $180^\circ$:
    * $0^\circ \to d = 0.0$
    * $30^\circ \to d = 0.1340$ ($1.0 - \cos(30^\circ)$)
    * $60^\circ \to d = 0.5000$ ($1.0 - \cos(60^\circ)$)
    * $90^\circ \to d = 1.0000$ (trực giao)
    * $120^\circ \to d = 1.5000$
    * $150^\circ \to d = 1.8660$
    * $180^\circ \to d = 2.0000$ (đối nghịch)
  - Khớp chính xác công thức toán học $d(u, v) = 1.0 - \text{sim}(u, v)$ tới sai số $< 10^{-6}$, không phải giá trị giả mạo hay hằng số cố định.

### 1.3. Thẩm định Completion-Only SFT (`src/training/train_sft_generator.py`)
- **Gán nhãn prompt $-100$ (`mask_prompt_tokens`)**:
  - Với chuỗi token `[10, 20, 30, 40, THINK_ID=500, 60, 70, 80]`:
  - `start_idx` tìm thấy chính xác tại index `4`.
  - Nhãn prompt `labels[:4]` là `[-100, -100, -100, -100]`.
  - Nhãn completion `labels[4:]` là `[500, 60, 70, 80]`.
- **Tính bất biến của hàm mất mát đối với prompt (`compute_completion_loss`)**:
  - Loss cơ sở trên chuỗi: `loss_base = 4.5705`.
  - Khi cộng nhiễu cực lớn ($+100.0$) vào toàn bộ logits của các vị trí prompt: `loss_perturbed_prompt = 4.5705` (Độ lệch tuyệt đối = `0.000000000`).
  - Khi tăng logit của token mục tiêu tại vị trí completion: Loss giảm mạnh từ `4.5705` xuống `2.8104`.
  - Chứng minh hàm mất mát phụ thuộc hoàn toàn và duy nhất vào các token completion sau thẻ `<think>`.
- **Cấu hình LoRA (`build_lora_config`)**:
  - Bao phủ đủ 7 module chiếu tuyến tính: `['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']`.

### 1.4. Thẩm định Dense GRPO (`src/training/train_grpo_generator.py`)
- **Đầu thưởng định dạng ($R_{\text{format}}$)**:
  - Kiểm tra các mẫu sinh bị lỗi: thiếu thẻ đóng, đảo ngược thứ tự thẻ (`<answer>` đứng trước `<think>`), thẻ rỗng chỉ chứa khoảng trắng đều trả về chính xác `0.0`. Thẻ hợp lệ trả về `1.0`.
- **Đầu thưởng thực thi SQLite ($R_{\text{exec}}$)**:
  - Đã probe trực tiếp trên tệp cơ sở dữ liệu thật `data/mini/financial_mini.sqlite`.
  - Khớp 100% với gold query: điểm `2.0`.
  - Thực thi thành công nhưng trả kết quả khác gold: điểm `1.0`.
  - Lỗi cú pháp SQL: điểm `0.0`.
  - Truy vấn phá hoại (`DROP TABLE clients;`): Bị chặn hoàn toàn, điểm `0.0`. Bảng dữ liệu được xác nhận nguyên vẹn (8 dòng bản ghi).
- **Đầu thưởng AST Clause Matching ($R_{\text{atomic}}$)**:
  - Phân tích cú pháp 5 mệnh đề: SELECT, FROM, WHERE, GROUP BY, ORDER BY.
  - Khi cả 5 mệnh đề phân tách hoàn toàn không có token chung: điểm trả về đúng `0.0`.
  - Khi trùng khớp hoàn toàn: điểm trả về đúng `1.0`.
- **Chuẩn hóa lợi thế nhóm ($A_j$)**:
  - Với phần thưởng nhóm đa dạng `[4.5, 2.0, 1.0, 0.0]`:
  - Lợi thế sau chuẩn hóa: `[1.5283, 0.0764, -0.5043, -1.1004]` (tổng trung bình $\approx 0.0$, triệt tiêu hiện tượng Advantage Collapse).

### 1.5. Thẩm định Drive Checkpoint Callback (`src/training/callbacks.py`)
- **Tìm kiếm Checkpoint mới nhất (`find_latest_checkpoint`)**:
  - Tạo các thư mục ngẫu nhiên `checkpoint-15`, `checkpoint-300`, `checkpoint-42`, `checkpoint-1050`, `checkpoint-80`, kèm các thư mục rác không khớp regex như `checkpoint-invalid`, `checkpoint-999x`.
  - Hàm quét khớp chính xác regex `r"^checkpoint[-_](\d+)$"`, bỏ qua các thư mục rác, và chọn đúng `checkpoint-1050`.
- **Sao lưu và Tỉa bớt trên Google Drive (`DriveCheckpointCallback`)**:
  - Thiết lập `max_drive_checkpoints=3`.
  - Khi tuần tự lưu checkpoint 100, 200, 300, 400, 500:
  - Thư mục Drive chỉ lưu giữ đúng 3 checkpoint mới nhất: `checkpoint-300`, `checkpoint-400`, `checkpoint-500`. Các checkpoint cũ 100 và 200 bị xóa bỏ sạch sẽ. Nội dung tệp weights bên trong được bảo toàn nguyên vẹn.
- **Khôi phục phiên (`get_resume_checkpoint`)**:
  - Khi cờ là `"latest"`, hàm ưu tiên tìm kiếm và trả về checkpoint trên Google Drive trước ổ local SSD để chống mất phiên khi runtime bị reset.

### 1.6. Thẩm định Laya-421M RLCD & Hiệu Chuẩn ECE (`src/training/train_decision_model.py`)
- **Tách 2 Learning Rates (`configure_dual_lr_optimizer`)**:
  - Tham số Backbone: $lr = 1 \times 10^{-5}, weight\_decay = 0.01$.
  - Tham số Head: $lr = 1 \times 10^{-4}, weight\_decay = 0.01$.
- **Hàm mất mát RLCD (`compute_rlcd_loss`)**:
  - $L = L_{\text{BCE}} + \alpha_{\text{cal}} (p - y)^2$.
  - Dự đoán đúng ($p=0.95, y=1$): loss = `0.0525`.
  - Dự đoán sai không chắc chắn ($p=0.51, y=0$): loss = `0.8434`.
  - Dự đoán sai nhưng tự tin thái quá ($p=0.99, y=0$): loss = `5.0952` (bị phạt nặng gấp $>6$ lần do hình phạt hiệu chuẩn bậc hai).
- **Tính toán ECE và Hiệu chuẩn nhiệt độ (`compute_ece`, `fit_temperature`)**:
  - Thử nghiệm trên phân phối hiệu chuẩn hoàn hảo: $ECE = 0.0000$.
  - Thử nghiệm trên phân phối mất hiệu chuẩn hoàn toàn: $ECE = 0.9500$.
  - Thử nghiệm trên phân phối dữ liệu mô hình quyết định: ECE ban đầu là `0.1505`. Sau khi tối ưu nhiệt độ $T=0.2$, ECE giảm xuống `0.0012 \le 0.081` (đạt chuẩn đặc tả).
  - Thử nghiệm trên dữ liệu nhiễu ngẫu nhiên: ECE trả về `0.0938`, chứng minh thuật toán không hardcode kết quả `0.0012`.

### 1.7. Kiểm thử đơn vị và hồi quy toàn bộ hệ thống
- Chạy `python3 -m unittest tests/test_training.py -v`:
  * `Ran 24 tests in 0.202s` -> `OK` (100% Passed).
- Chạy `python3 -m unittest discover tests/ -v`:
  * `Ran 74 tests in 0.519s` -> `OK` (100% Passed, không có regression).
- Chạy dry-run từng script:
  * `python3 src/training/train_sft_generator.py --dry_run` -> `{"status": "PASSED", "completion_masking_verified": true}`
  * `python3 src/training/train_grpo_generator.py --dry_run` -> `{"status": "PASSED", "advantage_collapse_prevented": true}`
  * `python3 src/training/train_embedding.py --dry_run` -> `{"status": "PASSED", "margin": 0.1}`
  * `python3 src/training/train_decision_model.py --dry_run` -> `{"status": "PASSED", "calibrated_ece": 0.0012}`

---

## 2. Logic Chain (Chuỗi Lập Luận Pháp Y)

1. **Phân tích bề mặt tấn công**:
   - Thẩm định viên đặt ra 5 giả thuyết hoài nghi:
     - *H1*: JEPA có thể rò rỉ thông tin từ Block 2 sang Block 1 hoặc mask chỉ là ma trận 0.
     - *H2*: SFT có thể vẫn để lộ gradient từ prompt tokens.
     - *H3*: GRPO có thể mock hàm thực thi SQLite bằng kết quả giả lập hoặc không ngăn chặn câu lệnh ghi.
     - *H4*: Callback có thể không xóa checkpoint cũ trên Drive gây tràn bộ nhớ.
     - *H5*: ECE có thể trả về giá trị hardcode `0.0012` để vượt qua kiểm thử.
2. **Đối chiếu thực nghiệm với các giả thuyết**:
   - *Bác bỏ H1*: Duyệt từng ô của ma trận mask trên nhiều kích thước khác nhau xác nhận tất cả ô cấm đều $< -10^8$. Thử nghiệm xoay vector chứng minh hàm cosine distance tính toán chuẩn xác từ $0.0$ đến $2.0$.
   - *Bác bỏ H2*: Kiểm tra độ nhạy bằng cách cộng giá trị khổng lồ vào logits của prompt tokens cho thấy loss giữ nguyên tuyệt đối (sai số $0.0$), trong khi can thiệp vào token completion lập tức làm thay đổi loss.
   - *Bác bỏ H3*: Lệnh probe thực thi trực tiếp trên tệp SQLite thật `financial_mini.sqlite`, câu lệnh `DROP TABLE` bị chặn và số lượng bản ghi bảng `clients` được xác nhận nguyên vẹn.
   - *Bác bỏ H4*: Thử nghiệm tạo 5 checkpoints cho thấy thư mục Drive chỉ còn đúng 3 thư mục mới nhất, 2 thư mục cũ đã bị xóa bằng `shutil.rmtree`.
   - *Bác bỏ H5*: Khi truyền dữ liệu nhiễu cao, `fit_temperature` trả về `0.0938`; khi truyền dữ liệu hoàn hảo, trả về `0.0`; khi truyền dữ liệu mô hình, trả về `0.0012`. Thuật toán hoàn toàn tính toán động theo công thức binned ECE.
3. **Tuân thủ quy ước thư mục**:
   - Toàn bộ mã nguồn nằm trong `src/training/`, kiểm thử nằm trong `tests/`, dữ liệu mẫu nằm trong `data/mini/`. Thư mục `.agents/teamwork/` chỉ chứa tài liệu markdown metadata của các agent.

---

## 3. Caveats (Lưu Ý & Phạm Vi Giới Hạn)

- Trong môi trường local CPU hiện tại, các mô hình ngôn ngữ lớn (3B/7B) không thể tải toàn bộ trọng số để chạy huấn luyện thực tế kéo dài. Việc thẩm định được thực hiện thông qua logic toán học của các forward pass, loss calculation, probe dữ liệu SQLite thật trên `data/mini/`, và chế độ `--dry_run`. Việc huấn luyện thực tế trên GPU phần cứng thuộc phạm vi triển khai của Milestone M3 (Google Colab Notebooks).

---

## 4. Conclusion (Kết Luận Thẩm Định)

- **Phán quyết (Verdict)**: **CLEAN (HỢP LỆ — KHÔNG VI PHẠM LIÊM CHÍNH)**.
- Toàn bộ 7 tệp tin trong `src/training/` và bài kiểm thử `tests/test_training.py` do `worker_m2` triển khai đều là mã nguồn đích thực, hoạt động ổn định, đúng đắn về mặt thuật toán và toán học, không có mã giả mạo (facade), không có kết quả hardcode, và vượt qua 100% các bài kiểm thử độc lập.
- Milestone M2 hoàn toàn đủ điều kiện nghiệm thu kỹ thuật.

---

## 5. Verification Method (Phương Pháp Xác Minh Độc Lập)

Bất kỳ thẩm định viên hoặc người dùng nào cũng có thể tái hiện việc xác minh độc lập thông qua các lệnh bash sau:

```bash
# 1. Chạy toàn bộ 24 bài kiểm thử của Milestone M2
python3 -m unittest tests/test_training.py -v

# 2. Chạy toàn bộ 74 bài kiểm thử của dự án
python3 -m unittest discover tests/ -v

# 3. Kiểm tra chế độ dry run của tất cả 4 script huấn luyện
python3 src/training/train_sft_generator.py --dry_run
python3 src/training/train_grpo_generator.py --dry_run
python3 src/training/train_embedding.py --dry_run
python3 src/training/train_decision_model.py --dry_run
```
