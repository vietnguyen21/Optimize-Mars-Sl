# LỘ TRÌNH TRIỂN KHAI THEO GIAI ĐOẠN (ROADMAP & PHASES SPECIFICATION)

> **Nguyên tắc bất biến**: Lộ trình được thiết kế theo quy trình tuần tự khép kín gồm **4 Phase chức năng**. Mỗi Phase chỉ được bắt đầu khi Phase trước đã hoàn thành đầy đủ các tiêu chí nghiệm thu (Definition of Done - DoD). Tuyệt đối không nhảy cóc sang giai đoạn sau khi chưa có dữ liệu hoặc mã nguồn kiểm chứng.

```
┌─────────────────────────┐
│ PHASE 1: DATA PIPELINE  │ ──► [4 Gói dữ liệu sạch + Mini-dataset]
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ PHASE 2: TRAIN SCRIPTS  │ ──► [Bộ script PyTorch/TRL hoàn chỉnh trên Colab]
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ PHASE 3: TRAIN EXECUTION│ ──► [Checkpoints mô hình lưu an toàn trên Drive]
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ PHASE 4: EXPERIMENTS    │ ──► [Kết quả thực nghiệm toàn tuyến & Báo cáo SOTA]
└─────────────────────────┘
```

---

## 1. PHASE 1: CHUẨN HÓA & ĐÓNG GÓI DỮ LIỆU (DATA PIPELINE)

### 1.1. Mục Tiêu
Biến toàn bộ các tập dữ liệu thô (BIRD 33.4 GB, Spider 1.0, Spider-DK, SynSQL-2.5M) thành 4 gói dữ liệu sạch chuyên biệt phục vụ huấn luyện 3 tác tử, đồng thời tạo bộ mẫu thử nhỏ (`data/mini/`) để kiểm thử hệ thống trong vòng 2 phút.

### 1.2. Nhiệm Vụ Chi Tiết
1. **Task 1.1 (Lọc dữ liệu gốc BIRD)**:
   - Dùng script kiểm tra chéo loại bỏ các mẫu có SQL đáp án mâu thuẫn ngữ nghĩa hoặc trả về kết quả rỗng bất thường, thu được 8,036 mẫu huấn luyện chuẩn.
2. **Task 1.2 (Đóng gói DATA-G1-EMB)**:
   - Tách từng bảng của 8,036 mẫu BIRD, tạo 90,102 thẻ thông tin cột gồm 6 trường dữ liệu.
   - Gán nhãn dương ($y=1$) cho các cột trong Gold SQL; tạo mẫu âm khó ($y=0$, $\text{margin}=0.1$) cho các cột trùng tên hoặc có độ tương đồng ngữ nghĩa cao trong cùng database.
3. **Task 1.3 (Trích xuất đồ thị DATA-G1-GRAPH)**:
   - Viết parser quét câu lệnh `CREATE TABLE` DDL của 95 DB BIRD và 138 DB Spider.
   - Dựng cấu trúc đồ thị tĩnh $G=(V, E)$ gồm nút Bảng, nút Cột, cạnh nội bộ và cạnh khóa ngoại PK-FK, xuất ra file `data/processed/graphs.pkl`.
4. **Task 1.4 (Làm sạch kho suy luận DATA-G2-SFT)**:
   - Áp dụng 4 bộ lọc trên SynSQL-2.5M: bỏ câu thiếu `SELECT`, lặp lại code trong `<think>`, chứa chú thích `--`, và đề bài > 7,000 token.
   - Bọc phần giải thích vào thẻ `<think>...</think>` và câu SQL vào thẻ `<answer>...</answer>`, xuất ra 916,000 mẫu `SynSQL-Think-916K`.
   - Thu thập 8,036 lời giải đúng từ mô hình lớn (DeepSeek-R1) để lập thành `Reasoning Bank`.
5. **Task 1.5 (Tạo bộ mẫu thử nghiệm data/mini/)**:
   - Trích xuất tự động 100 câu BIRD (kèm 2 SQLite databases tương ứng) và 1,000 câu SynSQL-Think.
   - Đặt tại `data/mini/` để dùng cho mọi kiểm thử smoke test.

### 1.3. Đầu Vào & Đầu Ra
- **Đầu vào**: Dữ liệu thô đặt tại `data/raw/` (BIRD, Spider, SynSQL).
- **Đầu ra**: 
  - `data/processed/DATA-G1-EMB.jsonl`
  - `data/processed/DATA-G1-GRAPH.pkl`
  - `data/processed/DATA-G2-SFT.jsonl`
  - `data/mini/` (dữ liệu mẫu kiểm thử nhanh)

### 1.4. Tiêu Chí Nghiệm Thu (DoD Phase 1)
- [ ] Chạy lệnh `python scripts/make_mini_dataset.py` thành công không lỗi, xuất đầy đủ 100 câu mẫu.
- [ ] Cấu trúc thẻ `<think>...</think><answer>...</answer>` của toàn bộ tập SFT được xác thực 100% bằng regex.
- [ ] File đồ thị PK-FK chứa đủ 95 đồ thị BIRD và 138 đồ thị Spider, không bị gãy cạnh nối.

---

## 2. PHASE 2: PHÁT TRIỂN MÃ NGUỒN HUẤN LUYỆN (TRAINING SCRIPTS)

### 2.1. Mục Tiêu
Xây dựng toàn bộ mã nguồn huấn luyện (PyTorch / HuggingFace Transformers / TRL) dưới dạng các script độc lập, tham số hóa linh hoạt, tối ưu cho môi trường Colab GPU, hỗ trợ cả hai mô hình sinh SQL 3B và 7B.

### 2.2. Nhiệm Vụ Chi Tiết
1. **Task 2.1 (Script Huấn luyện Agent 1)**:
   - Viết `src/training/train_embedding.py` huấn luyện `Qwen3-Embedding-0.6B` với hàm mất mát HN-SupCon.
   - Hỗ trợ xuất và lưu bảng vector nhúng cột xuống ổ cứng (`offline_column_index.npy`).
2. **Task 2.2 (Script Huấn luyện SFT Completion-Only Agent 2)**:
   - Viết `src/training/train_sft_generator.py` hỗ trợ `--model_type {3b, 7b}` (`FINER-SQL-3B` hoặc `Qwen2.5-Coder-7B`).
   - Tự động gắn nhãn `-100` cho toàn bộ prompt đề bài và DDL, chỉ tính loss trên thẻ `<think>` và `<answer>`.
   - Hỗ trợ huấn luyện LoRA ở rank 512 hoặc full fine-tuning trên Colab GPU.
3. **Task 2.3 (Script Căn chỉnh Não bộ LLM-JEPA)**:
   - Viết module `src/training/jepa_module.py` cài đặt 2-block causal mask và hàm mục tiêu khoảng cách cosine giữa token `[PRED]` và vector ẩn SQL chuẩn.
   - Tích hợp kỹ thuật Random JEPA-Loss Dropout ($LD=0.5$) để giảm 50% thời gian huấn luyện.
4. **Task 2.4 (Script Huấn luyện Học tăng cường Dense GRPO)**:
   - Viết `src/training/train_grpo_generator.py` cài đặt thuật toán GRPO với 4 đầu điểm thưởng (Format, Exec, Mem, Atomic) trên FINER-SQL-3B để chống sụp đổ lợi thế.
5. **Task 2.5 (Script Huấn luyện Phán quyết 0-Token Agent 3)**:
   - Viết `src/training/train_decision_model.py` hỗ trợ:
     - Domain SFT trên `NanoJev-0.6B` với 2 learning rates ($10^{-5}$ cho backbone, $10^{-4}$ cho choice head).
     - Thuật toán RLCD trên `Laya-421M` (ModernBERT-large) với phạt lệch hiệu chuẩn.
6. **Task 2.6 (Tích hợp Cơ chế Lưu Tự Động Google Drive)**:
   - Viết callback kiểm tra và tự động đồng bộ checkpoint từ `/content/checkpoints/` sang `/content/drive/MyDrive/Text2SQL_Checkpoints/`.

### 2.3. Đầu Vào & Đầu Ra
- **Đầu vào**: Dữ liệu từ Phase 1 (`data/processed/` hoặc `data/mini/`).
- **Đầu ra**: Toàn bộ mã nguồn huấn luyện hoàn chỉnh trong `src/training/`.

### 2.4. Tiêu Chí Nghiệm Thu (DoD Phase 2)
- [ ] Chạy thử nghiệm smoke test trên tập `data/mini/` với 1 batch thành công cho từng script huấn luyện mà không gặp lỗi cú pháp hay tràn bộ nhớ (OOM).
- [ ] Cơ chế callback lưu thử nghiệm 1 checkpoint lên Google Drive hoạt động trơn tru.

---

## 3. PHASE 3: THỰC THI HUẤN LUYỆN TRÊN GOOGLE COLAB (TRAINING EXECUTION)

### 3.1. Mục Tiêu
Tiến hành huấn luyện đầy đủ các mô hình trên môi trường Google Colab (sử dụng GPU T4/L4/A100), theo dõi biểu đồ hàm mất mát và lưu trữ an toàn các checkpoint trọng số cuối cùng vào Google Drive.

### 3.2. Nhiệm Vụ Chi Tiết
1. **Task 3.1 (Tạo Google Colab Notebooks)**:
   - Soạn thảo các notebook hoàn chỉnh trong thư mục `notebooks/`:
     - `01_colab_data_prep.ipynb`: Tải dữ liệu và chạy Phase 1 trên Colab.
     - `02_colab_train_grounding.ipynb`: Huấn luyện Qwen3-Embedding-0.6B.
     - `03_colab_train_generator_3b.ipynb`: Huấn luyện FINER-SQL-3B (SFT + JEPA + GRPO).
     - `03b_colab_train_generator_7b.ipynb`: Huấn luyện Qwen2.5-Coder-7B (Dành cho GPU Colab cao cấp A100).
     - `04_colab_train_validator.ipynb`: Huấn luyện Laya-421M RLCD.
2. **Task 3.2 (Thực thi huấn luyện lần lượt)**:
   - Chạy huấn luyện mô hình nhúng Agent 1 $\rightarrow$ xuất chỉ mục vector cột offline.
   - Chạy huấn luyện SFT Generator $\rightarrow$ căn chỉnh LLM-JEPA $\rightarrow$ chạy GRPO 4 đầu điểm.
   - Chạy huấn luyện bộ phán quyết System-1 Agent 3.
3. **Task 3.3 (Theo dõi & Ghi nhận)**:
   - Ghi lại nhật ký loss, learning rate, GPU memory utilization và thời gian chạy của mỗi epoch.
   - Đóng gói trọng số mô hình tốt nhất (best checkpoints) vào Google Drive.

### 3.3. Đầu Vào & Đầu Ra
- **Đầu vào**: Các script Phase 2 và dữ liệu đầy đủ Phase 1.
- **Đầu ra**: Các thư mục trọng số mô hình đã huấn luyện hoàn chỉnh:
  - `checkpoints/lite_sql_embedding_0.6b/`
  - `checkpoints/finer_sql_3b_sota/` (hoặc `qwen2.5_coder_7b_sota/`)
  - `checkpoints/laya_421m_rlcd/`

### 3.4. Tiêu Chí Nghiệm Thu (DoD Phase 3)
- [ ] Tất cả các mô hình hoàn thành đủ số epoch quy định, đồ thị loss hội tụ ổn định.
- [ ] Checkpoint đầy đủ được lưu trữ an toàn và kiểm tra nạp lại (model reload) thành công không lỗi.

---

## 4. PHASE 4: THÍ NGHIỆM ĐO ĐẠC & ĐỐI SÁNH KHOA HỌC (EXPERIMENTS & BENCHMARKING)

### 4.1. Mục Tiêu
Xây dựng kịch bản kiểm thử toàn tuyến (Evaluation Harness), sinh đủ 8–16 quỹ đạo ứng viên cho mỗi câu hỏi, chạy đo đạc trên 3 bộ dữ liệu chuẩn (BIRD-dev, Spider-test, Spider-DK), ghi nhận số liệu phân tích sâu (Ablation Study) và xuất báo cáo đối sánh với MARS-SQL 21B.

### 4.2. Nhiệm Vụ Chi Tiết
1. **Task 4.1 (Xây dựng Evaluation Harness)**:
   - Viết `src/evaluation/evaluate_end_to_end.py` phối hợp cả 3 tác tử:
     - Agent 1: LitE-SQL lấy top-25 cột $\rightarrow$ Cây Steiner nối bảng.
     - Agent 2: FINER-SQL-3B (hoặc 7B) sinh 8–16 phương án với rào chắn XGrammar PDA (kèm fallback).
     - Agent 3: PV-SQL chạy thử + kiểm tra 10 quy tắc AST $\rightarrow$ Laya-421M chấm điểm chọn phương án.
2. **Task 4.2 (Thí nghiệm 1: Đo tỷ lệ tìm đúng cột & 48 câu bảng cầu nối)**:
   - Đo chỉ số Recall/TPR của LitE-SQL trên toàn bộ BIRD-dev.
   - Chạy riêng biệt trên 48 câu hỏi mất bảng cầu nối của MARS-SQL 21B để chứng minh tỷ lệ cứu thành công 100% nhờ Cây Steiner.
3. **Task 4.3 (Thí nghiệm 2: Đo hiệu quả XGrammar & LLM-JEPA)**:
   - So sánh 3 chế độ sinh ở Agent 2: Giải mã tự nhiên (Native) vs Outlines vs XGrammar PDA.
   - Ghi nhận tỷ lệ đúng cú pháp (mục tiêu 100%), thời gian nạp DDL (4–8 ms) và mức giảm thông lượng (<3.7%).
4. **Task 4.4 (Thí nghiệm 3: Đo đạc bộ kiểm định System-1 & Sai số ECE)**:
   - Chạy so sánh giữa Majority Voting vs PV-SQL đơn lẻ vs Laya-421M 0-token trên 8 quỹ đạo ứng viên.
   - Tính toán sai số hiệu chuẩn kỳ vọng $ECE$ và kiểm thử chống đoán mò (Abstention).
5. **Task 4.5 (Thí nghiệm 4: Tổng hợp đối sánh toàn tuyến)**:
   - Đo toàn diện các Combo A, B, C, D trên BIRD-dev, Spider-test và Spider-DK.
   - Xuất bảng kết quả cuối cùng (Execution Accuracy EX, Valid Efficiency Score VES, Độ trễ, VRAM đỉnh).

### 4.3. Đầu Vào & Đầu Ra
- **Đầu vào**: Checkpoints đã huấn luyện từ Phase 3 và tập kiểm thử BIRD-dev, Spider.
- **Đầu ra**: 
  - File log chi tiết từng câu hỏi: `logs/evaluation_results.jsonl`
  - Bảng tổng hợp số liệu khoa học: `reports/benchmark_summary.md`
  - Các đồ thị trực quan hóa phục vụ bài báo nghiên cứu: `reports/figures/`

### 4.4. Tiêu Chí Nghiệm Thu (DoD Phase 4)
- [ ] Bản Combo C đạt độ chính xác thực thi $\text{EX} \ge 75.0\%$ trên BIRD-dev với thời gian phản hồi trung bình $\le 5.8$ giây.
- [ ] Thuật toán Cây Steiner cứu được trọn vẹn 48/48 câu lỗi mất bảng cầu nối.
- [ ] Xuất bản đầy đủ bảng số liệu khoa học chứng minh sự ưu việt toàn diện so với MARS-SQL 21B.
