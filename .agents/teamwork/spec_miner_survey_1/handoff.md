# BÁO CÁO BÀN GIAO KHẢO SÁT ĐẶC TẢ VÀ KIẾN TRÚC (HANDOFF REPORT)

> **Tác tử báo cáo:** `spec_miner_survey_1`  
> **Người nhận:** `3dfd88ef-5709-46d6-ba23-355dc6f4af89` (Orchestrator)  
> **Ngày lập:** 2026-11-02T04:14:00Z  
> **Loại bàn giao:** Hard Handoff (Hoàn thành đầy đủ nhiệm vụ khảo sát)

---

## 1. Quan Sát Trực Tiếp (Observation)

Tôi đã trực tiếp đọc và phân tích toàn bộ các tài liệu đặc tả có thẩm quyền trong không gian làm việc `/mnt/c/Workspace/optimize`:

1. **`ORIGINAL_REQUEST.md` (Dòng 12-44):**
   - R1: Triển khai kiến trúc đội ngũ theo `AGENTS_RESEARCH.md` gồm 4 vai trò, lập trình toàn bộ mã nguồn cho 4 Phase (`data/mini/`, `src/training/`, `notebooks/` cho 3B và 7B kèm Google Drive checkpointing, `src/evaluation/` sinh 8–16 quỹ đạo, 0-token decoding scoring, XGrammar PDA kèm fallback).
   - R2: Xây dựng bộ kiểm thử và nghiệm thu tự động (Smoke Test) trên `data/mini/` độc lập với quá trình huấn luyện kéo dài.
   - R3: Lập tài liệu bàn giao kỹ thuật `HANDOVER.md` tại thư mục gốc.

2. **`AGENTS_RESEARCH.md` (Dòng 7-63):**
   - Triết lý "Director-Led Single Source of Truth": Quyền lực tuyệt đối của `[AGENT-DIRECTOR-LEADER]`, không có dòng code hay dữ liệu nào được chuyển giao giữa các kỹ sư nếu chưa qua phê duyệt; phối hợp bù trừ điểm mạnh/yếu (Synergy Matrix); vòng lặp phản hồi khép kín (Closed-Loop Feedback).
   - 4 Tác tử định danh: `[AGENT-DIRECTOR-LEADER]`, `[AGENT-DATA-G1]`, `[AGENT-GEN-G2]`, `[AGENT-VAL-BENCH]`.

3. **`docs/specifications/01_project_specification.md` (Dòng 13-91):**
   - Nêu rõ 4 điểm nghẽn của MARS-SQL 21B: Quá nặng (41-47 GB VRAM, 22.12s), Mù cấu trúc ở Agent 1 (mất bảng cầu nối ở 48 câu BIRD-dev), Ảo giác tên bảng/cú pháp ở Agent 2, Phán quyết cảm tính chưa hiệu chuẩn ở Agent 3 ($ECE \approx 0.246$).
   - Giải pháp 3 chặng: LitE-SQL 0.6B + Cây Steiner $\rightarrow$ FINER-SQL-3B / Qwen2.5-Coder-7B + LLM-JEPA + XGrammar PDA (sinh 8-16 rollouts) $\rightarrow$ PV-SQL 10 AST rules + Laya-421M (ModernBERT) 0-token System-1 ($ECE \le 0.081$).

4. **`docs/specifications/02_architecture_design.md` (Dòng 31-131):**
   - Agent 1: LitE-SQL Qwen3-Embedding-0.6B với thẻ cột 6 trường (`table_name`, `column_name`, `data_type`, `constraints`, `column_description`, `sample_values`), hàm mất mát HN-SupCon với $\text{margin}=0.1$, đồ thị PK-FK $G=(V, E)$, Cây Steiner Node-Weighted Steiner Tree (NWST) bảo toàn 100% bảng cầu nối.
   - Agent 2: FINER-SQL-3B / Qwen2.5-Coder-7B, thẻ `<think>...</think><answer>...</answer>`, Completion-Only Loss (gán $-100$ cho prompt/DDL), LLM-JEPA căn chỉnh token $[PRED]$ qua cosine distance với 2-block causal mask và Random JEPA Dropout $LD=0.5$, XGrammar PDA nạp DDL trong 4-8 ms chặn 100% lỗi cú pháp và bảng.
   - Agent 3: Probing tối đa 5 câu có `LIMIT` (0 GB VRAM), 10 quy tắc cây cú pháp tất định PV-SQL (`sqlglot`), Laya-421M ModernBERT-large (~85 MB) chấm điểm song song ~1.2 ms qua token `[MASK]`, kỹ năng Abstention, và lọc bảo mật chống Prompt Injection.
   - 4 Cấu hình Combo: Combo A (~2.5B), Combo B (~3.6B), Combo C (~4.0B SOTA), Combo D (~8.0B).

5. **`docs/specifications/04_roadmap_and_phases.md` (Dòng 25-165):**
   - Quy định tuần tự nghiêm ngặt 4 Phase: Phase 1 (Data Pipeline) $\rightarrow$ Phase 2 (Training Scripts) $\rightarrow$ Phase 3 (Training Execution trên Colab) $\rightarrow$ Phase 4 (Experiments & Benchmark).
   - Nêu chi tiết danh mục file cần tạo: `scripts/make_mini_dataset.py`, `src/training/train_embedding.py`, `src/training/train_sft_generator.py`, `src/training/jepa_module.py`, `src/training/train_grpo_generator.py`, `src/training/train_decision_model.py`, `notebooks/01` đến `04`, `src/evaluation/evaluate_end_to_end.py`.

6. **`docs/specifications/05_environment_and_colab_guide.md` (Dòng 28-63):**
   - Lớp `DriveCheckpointCallback` kế thừa `TrainerCallback`, tự động đồng bộ `/content/checkpoints/` sang `/content/drive/MyDrive/Text2SQL_Colab/checkpoints/`.
   - Cơ chế khôi phục phiên `--resume_from_checkpoint latest`.

7. **Bốn tài liệu PDF kỹ thuật nền tảng (`docs/*.pdf`):**
   - Xác nhận cơ chế GRPO 4 đầu thưởng chi tiết ($R_{\text{format}}$, $R_{\text{exec}}$, $R_{\text{mem}}$, $R_{\text{atomic}}$) của FINER-SQL chống sụp đổ lợi thế (Advantage Collapse).
   - Hiện tượng Hollow Rescue (Chavan, 2026) khi dùng bộ khóa văn phạm và lý do bắt buộc ghép đôi XGrammar với LLM-JEPA và PV-SQL.
   - Hiện tượng OOD Drop trên NanoJev-0.6B từ 52% xuống 17% nếu không có Domain SFT với 2 tốc độ học tách biệt ($10^{-5}$ backbone, $10^{-4}$ choice head).

---

## 2. Chuỗi Suy Luận Logic (Logic Chain)

1. Từ **Quan sát 1 & 2**: Yêu cầu người dùng đòi hỏi triển khai toàn bộ mã nguồn của dự án dựa trên sự phân công của 4 vai trò tác tử và mô hình "Director-Led Single Source of Truth". Do đó, mọi mô-đun mã nguồn và quá trình kiểm thử cần phải được thiết kế có ranh giới rõ ràng tương ứng với 4 vai trò này:
   - `[AGENT-DIRECTOR-LEADER]`: Điều phối trung tâm, kiểm soát giao diện hợp đồng I/O, cổng phê duyệt và tổng hợp kết quả sang `HANDOVER.md`.
   - `[AGENT-DATA-G1]`: Chịu trách nhiệm toàn bộ `src/grounding/` (LitE-SQL, PK-FK Graph, Steiner Tree) và `data/mini/`.
   - `[AGENT-GEN-G2]`: Chịu trách nhiệm toàn bộ `src/training/` (Completion-Only SFT, LLM-JEPA, GRPO, XGrammar PDA & Fallback) và `notebooks/`.
   - `[AGENT-VAL-BENCH]`: Chịu trách nhiệm `src/validation/` (PV-SQL 10 AST rules, System-1 0-token decision, Abstention, Security Filter) và `src/evaluation/` (Evaluation Harness, Benchmark reports).

2. Từ **Quan sát 3, 4, 7**: Mục tiêu cốt lõi là hạ kích thước mô hình từ 21B xuống $2.5\text{B} - 4.0\text{B}$ mà không làm giảm độ chính xác. Để làm được điều này, sự kết hợp giữa các kỹ thuật không thể tách rời:
   - Nếu chỉ dùng vector embedding ở Agent 1 thì vẫn mất 48 câu bảng cầu nối; bắt buộc phải có thuật toán Cây Steiner trên đồ thị PK-FK để cứu đường nối JOIN.
   - Nếu chỉ dùng XGrammar ở Agent 2 thì vướng hiện tượng "Hollow Rescue" (đúng cú pháp nhưng sai ngữ nghĩa); do đó bắt buộc phải có LLM-JEPA ở Agent 2 để căn chỉnh tư duy và 10 quy tắc PV-SQL ở Agent 3 để chặn lỗi logic.
   - Nếu huấn luyện RL thông thường trên SLM 3B thì bị sụp đổ lợi thế (Advantage Collapse); do đó hàm thưởng GRPO bắt buộc phải gồm 4 đầu điểm chi tiết (Format, Exec, Mem từ Reasoning Bank, Atomic clause match).

3. Từ **Quan sát 5 & 6**: Quá trình triển khai thực tế trên Google Colab có nguy cơ mất session và mất trọng số. Do đó, việc xây dựng `DriveCheckpointCallback` và hỗ trợ `--resume_from_checkpoint latest` là yêu cầu kiến trúc bắt buộc của Phase 2 và Phase 3.

4. Từ **Quan sát 1 & 5**: Để các bài test chạy cục bộ không bị phụ thuộc vào việc tải 33GB dữ liệu BIRD hay phải chạy GPU hàng giờ, bộ dữ liệu mẫu `data/mini/` (100 câu BIRD, 2 SQLite DBs, 1,000 câu SynSQL-Think) và kịch bản smoke test toàn tuyến độc lập là điều kiện tiên quyết (DoD) trước khi bàn giao sang đội nghiên cứu.

---

## 3. Các Điểm Lưu Ý & Giả Định (Caveats)

1. **Giả định về môi trường phụ thuộc C++/CUDA của XGrammar:** XGrammar yêu cầu trình biên dịch C++/CUDA phù hợp. Do đó, mã nguồn ở Agent 2 bắt buộc phải có khối `try...except` bao bọc để fallback sang `Native Decoding` kèm cảnh báo log rõ ràng nếu môi trường thiếu thư viện này.
2. **Giới hạn phạm vi của Specification Miner:** Tác tử `spec_miner_survey_1` hoạt động hoàn toàn ở chế độ Read-Only, không trực tiếp tạo mã nguồn trong `src/`, `notebooks/`, hoặc `data/mini/`. Toàn bộ đặc tả, hợp đồng giao diện, và bảng tính năng đã được tổng hợp chi tiết trong `survey_report.md` để các tác tử kỹ thuật tiếp theo kế thừa trực tiếp.
3. **Các tài nguyên mô hình nền:** Việc huấn luyện thực tế phụ thuộc vào việc tải các mô hình gốc từ HuggingFace (`Qwen3-Embedding-0.6B`, `FINER-SQL-3B`, `Qwen2.5-Coder-7B`, `ModernBERT-large`). Trong bộ kiểm thử smoke test, các mô hình này cần được cấu hình chạy ở chế độ mô phỏng (stub/mock) hoặc tải các checkpoint nhỏ để đảm bảo hoàn thành trong < 2 phút.

---

## 4. Kết Luận (Conclusion)

1. Quá trình khảo sát đặc tả và kiến trúc dự án đã hoàn thành toàn diện 100%. Toàn bộ bối cảnh học thuật, 4 vai trò tác tử, mô hình điều phối SSoT, 4 Phase triển khai, các thuật toán then chốt (LitE-SQL HN-SupCon, Steiner Tree NWST, Completion-Only SFT, LLM-JEPA, Dense GRPO, XGrammar PDA, PV-SQL 10 AST rules, Laya-421M RLCD), cấu trúc notebook Colab và hợp đồng giao diện Pydantic đã được phân tích và văn bản hóa đầy đủ.
2. Sản phẩm bàn giao chính của phiên làm việc này là:
   - `/mnt/c/Workspace/optimize/.agents/teamwork/spec_miner_survey_1/survey_report.md`: Báo cáo khảo sát toàn diện (đầy đủ bảng Features Discovered 25 tính năng, bảng Edge Cases 13 trường hợp biên, ma trận Combo, hợp đồng Pydantic).
   - `/mnt/c/Workspace/optimize/.agents/teamwork/spec_miner_survey_1/handoff.md`: Báo cáo bàn giao 5 thành phần chuẩn hóa.
   - `/mnt/c/Workspace/optimize/.agents/teamwork/spec_miner_survey_1/progress.md`: Nhật ký tiến độ cập nhật.

---

## 5. Phương Pháp Xác Minh Độc Lập (Verification Method)

Người nhận bàn giao hoặc Orchestrator có thể xác minh độc lập các kết quả trên bằng các bước sau:

1. **Kiểm tra sự tồn tại và tính toàn vẹn của báo cáo khảo sát:**
   ```bash
   test -f /mnt/c/Workspace/optimize/.agents/teamwork/spec_miner_survey_1/survey_report.md && echo "✅ survey_report.md tồn tại"
   test -f /mnt/c/Workspace/optimize/.agents/teamwork/spec_miner_survey_1/handoff.md && echo "✅ handoff.md tồn tại"
   ```

2. **Xác minh các phần nội dung bắt buộc trong `survey_report.md`:**
   ```bash
   grep -q "Features Discovered" /mnt/c/Workspace/optimize/.agents/teamwork/spec_miner_survey_1/survey_report.md && echo "✅ Có bảng Features Discovered"
   grep -q "Edge Cases" /mnt/c/Workspace/optimize/.agents/teamwork/spec_miner_survey_1/survey_report.md && echo "✅ Có bảng Edge Cases"
   grep -q "Director-Led Single Source of Truth" /mnt/c/Workspace/optimize/.agents/teamwork/spec_miner_survey_1/survey_report.md && echo "✅ Có phân tích SSoT"
   grep -q "CompactSubSchema" /mnt/c/Workspace/optimize/.agents/teamwork/spec_miner_survey_1/survey_report.md && echo "✅ Có hợp đồng dữ liệu Pydantic"
   ```

3. **Điều kiện vô hiệu hóa kết luận (Invalidation Conditions):**
   - Nếu phát hiện tài liệu đặc tả nào trong `docs/specifications/` có yêu cầu trái ngược với bảng hợp đồng giao diện Pydantic hoặc cấu hình Combo mà chưa được ghi chú.
   - Nếu bảng Features Discovered bỏ sót bất kỳ tính năng cốt lõi nào được nêu trong `AGENTS_RESEARCH.md` hoặc 4 tài liệu PDF kỹ thuật.
