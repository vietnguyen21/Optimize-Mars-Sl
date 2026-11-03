# THIẾT KẾ KIẾN TRÚC HỆ THỐNG (ARCHITECTURE DESIGN DOCUMENT)

## 1. Triết Lý Thiết Kế Hệ Thống (Architectural Philosophy)

Hệ thống được thiết kế dựa trên nguyên lý **Heterogeneous Multi-Agent Specialization** (Chuyên biệt hóa đa tác tử dị thể). Thay vì sử dụng các mô hình ngôn ngữ lớn vạn năng (General LLM 7B) cho mọi công việc, kiến trúc chia bài toán thành 3 chặng khép kín:

```
[ Natural Language Question Q + Full Database Schema S ]
                           │
                           ▼
             ┌───────────────────────────┐
             │  AGENT 1: SCHEMA GROUNDING │  (Embedding + Graph Algorithms)
             └─────────────┬─────────────┘
                           │ Compact Sub-Schema S' (with bridge tables preserved)
                           ▼
             ┌───────────────────────────┐
             │ AGENT 2: SQL GENERATION   │  (SLM + XGrammar PDA + LLM-JEPA)
             └─────────────┬─────────────┘
                           │ 8–16 Candidate SQL Trajectories {SQL_1, ..., SQL_K}
                           ▼
             ┌───────────────────────────┐
             │   AGENT 3: VALIDATION     │  (PV-SQL AST Probing + 0-Token System-1)
             └─────────────┬─────────────┘
                           │
                           ▼
              [ Verified Execution-Ready SQL* ]
```

---

## 2. Giải Phẫu Chi Tiết Tác Tử 1: Schema Grounding Agent

Nhiệm vụ của Agent 1 là nhận câu hỏi người dùng $Q$ và toàn bộ cơ sở dữ liệu $S$, trích xuất chính xác tập con các bảng và cột cần thiết $S' \subset S$, đảm bảo **bảo toàn 100% các đường nối khóa ngoại (PK-FK)**.

### 2.1. Bộ Nhúng Cột LitE-SQL (Qwen3-Embedding-0.6B)
- **Đóng gói dữ liệu cột**: Mỗi cột trong cơ sở dữ liệu được biểu diễn thành một văn bản mô tả chuẩn hóa gồm 6 trường thông tin:
  1. `table_name`: Tên bảng chứa cột (ví dụ: `schools`, `frpm`).
  2. `column_name`: Tên chính xác của cột (ví dụ: `CDSCode`, `County Name`).
  3. `data_type`: Kiểu dữ liệu (`TEXT`, `INTEGER`, `REAL`, `DATE`).
  4. `constraints`: Khóa chính (`PRIMARY KEY`) hoặc khóa ngoại (`FOREIGN KEY REFERENCES ...`).
  5. `column_description`: Ý nghĩa nghiệp vụ lấy từ từ điển dữ liệu.
  6. `sample_values`: 3–5 giá trị thực tế tiêu biểu lưu trong cột.
- **Học phân biệt cột trùng tên (HN-SupCon)**: Huấn luyện với hàm mất mát tương phản có giám sát:
  $$\mathcal{L}_{\text{HN-SupCon}} = - \log \frac{\exp(\text{sim}(q, c^+) / \tau)}{\exp(\text{sim}(q, c^+) / \tau) + \sum_{c^- \in \mathcal{H}} \exp(\text{sim}(q, c^-) / \tau)}$$
  Trong đó tập mẫu âm khó $\mathcal{H}$ (Hard Negatives) gồm các cột trong cùng DB có độ tương đồng văn bản cao nhưng không nằm trong đáp án (biên phân cách $\text{margin} = 0.1$).
- **Offline Indexing**: Đóng băng trọng số, mã hóa toàn bộ cột của các cơ sở dữ liệu thành file vector tĩnh lưu trên đĩa. Khi suy luận, chỉ tốn vài mili-giây để truy vấn Top-25 cột có cosine similarity cao nhất với vector câu hỏi.

### 2.2. Thuật Toán Cây Steiner Tĩnh (Graph-Link) Cứu Bảng Cầu Nối
- **Dựng đồ thị cơ sở dữ liệu $G = (V, E)$**:
  - $V$: Tập đỉnh gồm toàn bộ các Bảng và các Cột.
  - $E$: Tập cạnh gồm cạnh nội bộ (Bảng $\leftrightarrow$ Cột của bảng đó) và cạnh khóa ngoại (Cột PK $\leftrightarrow$ Cột FK giữa hai bảng khác nhau).
- **Giải bài toán Cây Steiner có trọng số nút (NWST)**:
  - Các cột được chọn từ LitE-SQL được xem là các "điểm bắt buộc phải đến" (Terminal Nodes).
  - Thuật toán Cây Steiner tự động tìm cây con có tổng trọng số nhỏ nhất nối liền tất cả các đỉnh này.
  - Bất kỳ bảng trung gian nào nằm trên đường đi ngắn nhất giữa hai bảng chính sẽ tự động được giữ lại, giải quyết triệt để lỗi mất **bảng cầu nối (Bridge Tables)** ở 48 câu hỏi BIRD-dev.
- **Mở rộng (Phase 2)**: Huấn luyện thêm mạng nơ-ron đồ thị nhẹ (*Relation-Aware Heterogeneous GNN*) để cập nhật vector nút trước khi chạy Cây Steiner.

---

## 3. Giải Phẫu Chi Tiết Tác Tử 2: SQL Generation Agent

Nhiệm vụ của Agent 2 là chuyển hóa câu hỏi và lược đồ rút gọn $S'$ thành câu lệnh SQL chuẩn xác về cả cú pháp và ngữ nghĩa.

### 3.1. Mô Hình Cốt Lõi & Định Dạng Suy Luận
- **Mô hình**: Hỗ trợ 2 phân khúc chính:
  - `FINER-SQL-3B`: Mô hình 3B tham số tối ưu hóa cho tốc độ và khả năng chạy trên GPU nhẹ.
  - `Qwen2.5-Coder-7B`: Mô hình 7B tham số tối ưu hóa cho độ chính xác SOTA.
- **Định dạng sinh có cấu trúc**: Bắt buộc sinh luồng suy luận trước rồi mới xuất câu lệnh:
  ```text
  <think>
  Phân tích yêu cầu câu hỏi, xác định bảng cần nối và điều kiện lọc...
  </think>
  <answer>
  SELECT ... FROM ... WHERE ...
  </answer>
  ```
- **Completion-Only Loss (Huấn luyện SFT)**: Gán nhãn `-100` cho toàn bộ các token đề bài, câu hỏi và DDL trong hàm Cross-Entropy, chỉ tính đạo hàm cập nhật trên phần mô hình tự viết ra (`<think>...<answer>...`), ngăn chặn triệt để hiện tượng học vẹt tên bảng.

### 3.2. Căn Chỉnh Não Bộ Text-to-Code Bằng LLM-JEPA (Giai đoạn nâng cao)
- Bổ sung hàm mục tiêu dự đoán trạng thái ẩn giữa khối Câu hỏi (Text) và khối Câu lệnh SQL (Code):
  $$\mathcal{L}_{\text{LLM-JEPA}} = \mathcal{L}_{\text{LLM}}(\text{Text}, \text{Code}) + \lambda \cdot d(\text{Pred}(\text{Enc}(\text{Text})), \text{Enc}(\text{Code}))$$
- Sử dụng token đặc biệt `[PRED]` ở cuối câu hỏi để dự đoán vector ẩn của câu lệnh SQL. Tấm chắn chú ý (2-block causal mask) ngăn khối Text nhìn trộm khối Code.
- Tăng độ hiểu ngữ nghĩa sâu sắc mà không phát sinh bất kỳ độ trễ nào lúc chạy thực tế ($0\%$ inference overhead).

### 3.3. Khóa Văn Phạm Thời Gian Thực XGrammar PDA
- **Nguyên lý Ôtômat đẩy xuống (Pushdown Automaton)**:
  - Chia từ điển token của mô hình thành 2 nhóm: >99% từ cố định được tính trước thành bảng mặt nạ bit lưu trong bộ nhớ đệm (bitmask cache); <1% từ phụ thuộc ngữ cảnh được kiểm tra với ngăn xếp (stack).
  - Tốc độ nạp và biên dịch lược đồ $S'$ chỉ mất **4–8 mili-giây** (nhanh hơn Outlines từ 250 đến 1,000 lần).
- **Hiệu quả**: Chặn 100% lỗi cú pháp SQL, xóa sổ lỗi gõ sai tên bảng/cột và lỗi ép sai kiểu dữ liệu số/chuỗi.
- **Giải pháp chống "Giải cứu rỗng" (Hollow Rescue)**: Phối hợp XGrammar với LLM-JEPA (hiểu đúng ở Agent 2) và PV-SQL (bắt lỗi logic ở Agent 3) để ngăn ngừa trường hợp câu lệnh đúng cú pháp nhưng sai ngữ nghĩa.

---

## 4. Giải Phẫu Chi Tiết Tác Tử 3: Validation Agent

Nhiệm vụ của Agent 3 là rà soát, kiểm định và chọn ra câu lệnh tốt nhất trong số 8–16 quỹ đạo ứng viên do Agent 2 sinh ra.

### 4.1. Thăm Dò Dữ Liệu & 10 Quy Tắc Cây Cú Pháp PV-SQL
- **Thăm dò dữ liệu thật (Probing)**: Cho phép chạy tối đa 5 truy vấn ngắn có `LIMIT` trên cơ sở dữ liệu thật để kiểm tra định dạng dữ liệu (ví dụ: cột `state` lưu `'CA'` hay `'California'`) mà tốn 0 GB VRAM.
- **10 Quy tắc AST tất định**: Phân tích câu lệnh thành cây cú pháp (AST) bằng `sqlglot` và đối chiếu với từ khóa câu hỏi:
  1. `Distinct`: Câu hỏi chứa `unique`, `distinct`, `different` $\rightarrow$ AST phải có `DISTINCT` hoặc `GROUP BY`.
  2. `Top-K`: Câu hỏi chứa `top N`, `highest`, `lowest` $\rightarrow$ AST phải có `ORDER BY ... LIMIT`.
  3. `Average`: Câu hỏi chứa `average`, `mean` $\rightarrow$ AST phải có hàm `AVG()`.
  4. `Count`: Câu hỏi hỏi `how many`, `number of` $\rightarrow$ AST phải có `COUNT()`.
  5. `Sum / Total`: Câu hỏi chứa `total`, `sum` $\rightarrow$ AST phải có `SUM()`.
  6. `Extreme`: Chứa `maximum`, `minimum` $\rightarrow$ AST phải có `MAX()`, `MIN()`.
  7. `Comparison`: So sánh hơn/kém $\rightarrow$ AST phải có mệnh đề `WHERE` chứa toán tử so sánh.
  8. `Null Handling`: Điều kiện lọc rỗng khi truy vấn thuộc tính tùy chọn.
  9. `Join Validity`: Các liên kết `JOIN` phải tương thích với đồ thị PK-FK.
  10. `Group By Completeness`: Các cột không tổng hợp trong `SELECT` phải xuất hiện trong `GROUP BY`.

### 4.2. Bộ Phán Quyết Nhanh 0-Token System-1 (Laya-421M / NanoJev-0.6B)
- **Laya-421M (ModernBERT-large Cross-Encoder)**:
  - Kích thước chỉ ~85 MB.
  - Chèn token `[MASK]` đại diện cho từng phương án SQL để chấm điểm song song toàn bộ 8–16 ứng viên trong **~1.2 ms** (tương đương 810 quyết định/giây trên GPU T4).
  - Huấn luyện bằng thuật toán **RLCD** (Reinforcement Learning for Calibrated Decisions), đạt sai số hiệu chuẩn kỳ vọng cực thấp ($ECE \le 0.081$).
- **Kỹ năng Từ chối đoán mò (Abstention - Kev-0.8B)**:
  - Khi cả 16 phương án sinh ra đều sai hoặc mâu thuẫn, mô hình chủ động từ chối phán quyết và yêu cầu sinh lại thay vì đoán mò bừa bãi.
- **Bảo mật Chống Prompt Injection**:
  - Tách biệt nghiêm ngặt: Chỉ đưa tên bảng/cột tin cậy, câu lệnh SQL và biên bản kiểm tra 10 quy tắc AST vào trạng thái `state` của mô hình System-1; tuyệt đối không nối chuỗi văn bản thô từ cơ sở dữ liệu vào câu lệnh điều khiển.

---

## 5. Bảng Ma Trận Các Tổ Hợp Kiến Trúc Toàn Tuyến

| Cấu Hình | Agent 1 (Grounding) | Agent 2 (Generation) | Agent 3 (Validation) | Phần Cứng Mục Tiêu | Mục Tiêu Hiệu Năng BIRD-dev |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Combo A** *(Ultra-Light)* | LitE-SQL (0.6B) + Cây Steiner | SLM-SQL-1.5B + XGrammar | PV-SQL + Von-395M / Laya-421M | Laptop CPU / <8GB RAM | 69.5% – 72.0% EX |
| **Combo B** *(Shared System-1)* | NanoJev-0.6B + Cây Steiner | FINER-SQL-3B + XGrammar | PV-SQL + NanoJev-0.6B (Domain SFT) | 1 GPU 4GB–8GB VRAM | 73.5% – 75.5% EX |
| **Combo C** *(Pareto SOTA)* | LitE-SQL (0.6B) + Cây Steiner + GNN | FINER-SQL-3B + LLM-JEPA + XGrammar | PV-SQL + Laya-421M (RLCD) | 1 GPU 8GB–16GB VRAM | **75.0% – 76.8% EX** (Chuẩn mục tiêu) |
| **Combo D** *(High Accuracy)* | LitE-SQL (0.6B) + Cây Steiner + GNN | Qwen2.5-Coder-7B + LLM-JEPA + XGrammar | PV-SQL + Laya-421M (RLCD) | 1 GPU 24GB VRAM | **> 77.84% EX** (Vượt MARS-SQL 21B) |
