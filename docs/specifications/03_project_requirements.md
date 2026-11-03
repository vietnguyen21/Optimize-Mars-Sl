# YÊU CẦU KỸ THUẬT & TIÊU CHÍ NGHIỆM THU (PROJECT REQUIREMENTS & ACCEPTANCE CRITERIA)

## 1. Định Nghĩa Các Chỉ Số Đo Lường Cốt Lõi (Core Metrics)

Dự án tuân thủ nghiêm ngặt các tiêu chuẩn đo lường học thuật của cộng đồng Text-to-SQL quốc tế:

### 1.1. Execution Accuracy (EX — Độ chính xác thực thi)
- **Định nghĩa**: Tỷ lệ phần trăm câu lệnh SQL sinh ra chạy thực tế trên cơ sở dữ liệu SQLite và trả về bảng kết quả (result set) trùng khớp hoàn toàn ($100\%$) với bảng kết quả của câu lệnh đáp án chuẩn (*Gold SQL*), không phân biệt thứ tự dòng nếu không có `ORDER BY`.
- **Công thức**:
  $$\text{EX} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{Exec}(SQL_i, DB_i) \equiv \text{Exec}(SQL_i^*, DB_i))$$
- **Ý nghĩa**: Thước đo quan trọng nhất phản ánh khả năng ứng dụng thực tế.

### 1.2. Valid Efficiency Score (VES — Điểm hiệu quả thực thi)
- **Định nghĩa**: Điểm số đánh giá mức độ tối ưu về thời gian chạy và chi phí tính toán của câu lệnh SQL hợp lệ trên hệ quản trị cơ sở dữ liệu.
- **Công thức**:
  $$\text{VES} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{EX}_i = 1) \cdot \sqrt{\frac{T(SQL_i^*)}{T(SQL_i)}}$$
  (Trong đó $T$ là thời gian thực thi trung bình qua nhiều lượt chạy).

### 1.3. Schema Linking Recall / TPR (Tỷ lệ thu hồi cột & bảng đúng)
- **Định nghĩa**: Tỷ lệ phần trăm các bảng và cột thực sự cần thiết trong câu SQL đáp án chuẩn được Tác tử 1 (Grounding) trích xuất thành công vào lược đồ rút gọn $S'$.
- **Mục tiêu**: Đạt $\text{TPR} \ge 95.23\%$ trên tập BIRD-dev với Top-25 cột.

### 1.4. Bridge Tables Recovery Rate (Tỷ lệ cứu bảng cầu nối)
- **Định nghĩa**: Tỷ lệ phần trăm bảng cầu nối trung gian được bảo toàn thành công trong tập con gồm **48 câu hỏi BIRD-dev** mà hệ thống MARS-SQL 21B gốc bị mất.
- **Mục tiêu**: Đạt **$100\%$ (48/48 câu)** khi kích hoạt thuật toán Cây Steiner.

### 1.5. Expected Calibration Error (ECE — Sai số hiệu chuẩn kỳ vọng)
- **Định nghĩa**: Đo lường mức độ chênh lệch giữa điểm số tự tin mà mô hình phán quyết System-1 đưa ra so với xác suất chạy đúng thực tế.
- **Mục tiêu**: Đạt $\text{ECE} \le 0.081$ trên Laya-421M (vượt trội so với MARS-SQL 7B và Jev gốc ở mức $0.246$).

---

## 2. Yêu Cầu Chức Năng (Functional Requirements)

### FR-01: Đường Ống Xử Lý & Làm Sạch Dữ Liệu (Phase 1)
- **FR-01.1 (Lọc nhãn sai BIRD)**: Tự động rà soát và loại bỏ các mẫu huấn luyện có SQL đáp án mâu thuẫn ngữ nghĩa hoặc trả về kết quả rỗng bất thường (lọc từ 9,428 mẫu xuống 8,036 mẫu sạch).
- **FR-01.2 (Đóng gói HN-SupCon)**: Tách 90,102 thẻ thông tin cột, tự động gắn nhãn dương ($y=1$) và mẫu âm khó ($y=0$, biên $\text{margin}=0.1$) cho từng cơ sở dữ liệu.
- **FR-01.3 (Trích xuất đồ thị PK-FK)**: Quét toàn bộ câu lệnh DDL của 95 DB BIRD và 138 DB Spider để tạo đồ thị tĩnh $G=(V, E)$ lưu trữ dưới định dạng nhẹ (JSON / Pickle).
- **FR-01.4 (Làm sạch SynSQL-2.5M)**: Áp dụng 4 bộ lọc tự động: loại bỏ câu thiếu `SELECT`, lặp lại code trong `<think>`, chứa chú thích `--`, và câu đề bài vượt quá 7,000 token; chỉ giữ lại 916k mẫu chuẩn.
- **FR-01.5 (Chế độ dữ liệu nhỏ mini-dataset)**: Hỗ trợ tạo tập mẫu 100 câu BIRD và 1,000 câu SynSQL trong `data/mini/` để phục vụ kiểm tra hệ thống huấn luyện nhanh (smoke test) trong vòng <2 phút.

### FR-02: Kịch Bản Huấn Luyện Đa Tác Tử (Phase 2 & 3)
- **FR-02.1 (Huấn luyện bộ nhúng Agent 1)**: Huấn luyện `Qwen3-Embedding-0.6B` bằng hàm mất mát HN-SupCon; hỗ trợ xuất file vector nhúng cột offline.
- **FR-02.2 (Huấn luyện SFT Completion-Only Agent 2)**: Huấn luyện `FINER-SQL-3B` và `Qwen2.5-Coder-7B` trên cấu trúc `<think>...</think><answer>...</answer>`, che toàn bộ đề bài bằng nhãn `-100`.
- **FR-02.3 (Căn chỉnh LLM-JEPA)**: Cài đặt mặt nạ chú ý 2 khối (2-block causal mask) và hàm khoảng cách cosine giữa token `[PRED]` và vector ẩn câu SQL chuẩn.
- **FR-02.4 (Huấn luyện Học tăng cường GRPO)**: Cài đặt hàm thưởng 4 đầu điểm chi tiết (Format, Exec, Mem, Atomic) trên FINER-SQL-3B để chống sụp đổ lợi thế.
- **FR-02.5 (Huấn luyện Phán quyết 0-Token Agent 3)**: Huấn luyện `Domain SFT` trên NanoJev-0.6B (2 tốc độ học tách biệt: $10^{-5}$ cho khung xương, $10^{-4}$ cho đầu phân loại) và huấn luyện `RLCD` cho Laya-421M.

### FR-03: Thực Thi Đánh Giá & Thí Nghiệm So Sánh (Phase 4)
- **FR-03.1 (Hệ thống sinh quỹ đạo)**: Cho phép sinh đồng thời $N=8$ hoặc $N=16$ câu lệnh SQL ứng viên cho mỗi câu hỏi với tham số nhiệt độ $T=0.7$, Top-$p=0.9$, Top-$k=50$.
- **FR-03.2 (Bộ khóa XGrammar PDA)**: Rào chắn 100% cú pháp SQL thời gian thực, có cơ chế fallback sang Native Decoding kèm cảnh báo khi môi trường không có XGrammar.
- **FR-03.3 (Thẩm định PV-SQL 10 quy tắc AST)**: Chạy thử tối đa 5 câu ngắn có `LIMIT` và đối chiếu cây cú pháp với 10 quy tắc nghiệp vụ suy ra từ câu hỏi.
- **FR-03.4 (Nhật ký thí nghiệm học thuật)**: Tự động ghi log chi tiết từng quỹ đạo sinh ra (kết quả thực thi, điểm AST, điểm phán quyết 0-token) để phục vụ phân tích phương sai và vẽ biểu đồ bài báo.

---

## 3. Yêu Cầu Phi Chức Năng (Non-Functional Requirements)

- **NFR-01 (Tài nguyên tính toán)**:
  - Bản Combo C phải chạy vừa vặn trên 1 GPU NVIDIA có từ 16GB đến 24GB VRAM (như Tesla T4, L4, A100 trên Google Colab).
  - Bản Combo A phải hỗ trợ chạy trực tiếp trên CPU laptop thông thường với bộ nhớ RAM < 8 GB.
- **NFR-02 (Thời gian phản hồi)**: Độ trễ trung bình toàn tuyến cho mỗi câu hỏi không vượt quá **6.5 giây** (so với mốc 22.12 giây của MARS-SQL 21B).
- **NFR-03 (Độ an toàn dữ liệu & Checkpoint)**: Toàn bộ quá trình huấn luyện trên Colab phải có cơ chế ngắt quãng an toàn (checkpoint recovery) và tự động đồng bộ trọng số sang Google Drive để phòng ngừa mất phiên làm việc.
- **NFR-04 (Bảo mật)**: Tách biệt tuyệt đối giữa dữ liệu bảng thô của người dùng và câu lệnh điều khiển hệ thống, ngăn chặn 100% tấn công Prompt Injection theo cảnh báo của Wang et al. (2026).
- **NFR-05 (Tính tái lập)**: Mã nguồn phải cố định toàn bộ hạt giống ngẫu nhiên (random seeds: `{82, 23, 37, 84, 4}`) để đảm bảo kết quả đo đạc khoa học có thể tái lặp 100%.

---

## 4. Tiêu Chí Nghiệm Thu Thực Nghiệm (Final Acceptance Criteria)

| Hạng Mục Kiểm Tra | Hệ Thống Cơ Sở (MARS-SQL 21B) | Mục Tiêu Nghiệm Thu Combo C (~4.0B) | Mục Tiêu Nghiệm Thu Combo D (~8.0B) |
| :--- | :---: | :---: | :---: |
| **Độ chính xác BIRD-dev (EX)** | 77.84% | **75.00% – 76.80%** | **> 77.84%** (Vượt mốc 21B) |
| **Độ chính xác Spider-test (EX)**| 89.75% | **88.50% – 89.50%** | **$\ge$ 89.75%** |
| **Độ chính xác Spider-DK (OOD)**| 78.13% | **77.50% – 79.00%** | **> 79.50%** |
| **Bài test 48 câu mất bảng cầu nối**| Thất bại 48/48 câu (0%) | **Cứu thành công 100% (48/48)** | **Cứu thành công 100% (48/48)** |
| **Tỷ lệ đúng cấu trúc (Syntax/Table)**| 78.6% – 92.9% | **100%** (nhờ XGrammar) | **100%** (nhờ XGrammar) |
| **Độ trễ trung bình / câu hỏi** | 22.12 giây | **4.2 – 5.8 giây** (Giảm >75%) | **6.5 – 8.5 giây** (Giảm >60%) |
| **Bộ nhớ VRAM đỉnh** | 41.0 – 47.0 GB | **3.5 – 8.5 GB** (Giảm >80%) | **< 20.0 GB** (Vừa 1 GPU 24GB) |
| **Sai số hiệu chuẩn (ECE)** | ~0.246 | **$\le$ 0.081** | **$\le$ 0.081** |
