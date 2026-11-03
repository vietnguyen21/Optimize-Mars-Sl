# BÁO CÁO BÀN GIAO KHẢO SÁT ĐẶC TẢ (HANDOFF REPORT)

**Tác tử**: `spec_miner_survey_2`  
**Vai trò**: Specification Miner (Khai thác & Hệ thống hóa Đặc tả Kỹ thuật)  
**Nhiệm vụ**: Khai thác chi tiết yêu cầu, lộ trình và tiêu chí triển khai cho 4 Phase (Data Pipeline, Training Scripts, Colab Notebooks, Evaluation & Benchmark)  
**Người nhận**: `3dfd88ef-5709-46d6-ba23-355dc6f4af89` (`orchestrator`)  
**Ngày thực hiện**: 2026-11-02  
**Loại bàn giao**: Hard Handoff (Hoàn thành trọn vẹn nhiệm vụ)

---

## 1. Quan Sát (Observation)

Tôi đã tiến hành rà soát, đọc toàn văn và trích xuất dữ liệu từ toàn bộ các tài liệu đặc tả học thuật và hướng dẫn kỹ thuật trong kho lưu trữ `/mnt/c/Workspace/optimize`:
1. **Tài liệu đặc tả Markdown**:
   - `ORIGINAL_REQUEST.md`: R1 (Mã nguồn 4 Phase), R2 (Smoke test tự động trên `data/mini/`), R3 (Tài liệu HANDOVER.md).
   - `docs/specifications/01_project_specification.md` (lines 1–92): Bối cảnh MARS-SQL 21B, 4 nút thắt cổ chai, giải pháp thay thế SLM 2.5B–4.0B.
   - `docs/specifications/02_architecture_design.md` (lines 1–132): Cấu trúc 3 Agent, công thức $\mathcal{L}_{\text{HN-SupCon}}$, $\mathcal{L}_{\text{LLM-JEPA}}$, thuật toán Cây Steiner tĩnh (NWST), khóa văn phạm XGrammar PDA, 10 quy tắc PV-SQL AST.
   - `docs/specifications/03_project_requirements.md` (lines 1–81): Công thức EX, VES, Recall/TPR, Bridge Tables Recovery, ECE, và bảng chỉ tiêu nghiệm thu.
   - `docs/specifications/04_roadmap_and_phases.md` (lines 1–166): Lộ trình tuần tự khép kín 4 Phase và Definition of Done (DoD).
   - `docs/specifications/05_environment_and_colab_guide.md` (lines 1–109): Cấu hình GPU Colab (T4, L4, A100), mount Drive, class `DriveCheckpointCallback`, phục hồi session `--resume_from_checkpoint latest`.
   - `docs/specifications/requirements.md` & `requirements.txt`: 14 thư viện lõi (torch, transformers, trl, peft, xgrammar, networkx, sqlglot, sentence-transformers, etc.).
   - `AGENTS_RESEARCH.md` & `GLOSSARY.md`: Cơ chế điều phối "Director-Led Single Source of Truth" và thuật ngữ chuẩn hóa.
2. **Tài liệu PDF kỹ thuật chuyên sâu**:
   - `docs/Ke Hoach Xu Ly Du Lieu.pdf` (6 trang): Quy trình lọc BIRD 9,428 $\rightarrow$ 8,036 mẫu sạch, 90,102 thẻ thông tin cột (6 trường: `table_name`, `column_name`, `data_type`, `constraints`, `column_description`, `sample_values`), 4 bộ lọc SynSQL (916k CoT, 310k Merge), quy tắc lọc bảo mật Prompt Injection cho System-1 (Wang et al., 2026).
   - `docs/Ke Hoach Huan Luyen Mo Hinh.pdf` (5 trang): Hàm mục tiêu Completion-Only Loss (nhãn `-100`), LLM-JEPA 2-block causal mask ($\lambda=1.0, k=1, lr=2\times 10^{-5}, LD=0.5$), Dense GRPO 4 đầu thưởng ($R_{\text{format}}, R_{\text{exec}}, R_{\text{mem}}, R_{\text{atomic}}$) trên FINER-SQL-3B, Domain SFT NanoJev-0.6B (2 learning rates: $10^{-5}$ và $10^{-4}$), RLCD Laya-421M ($ECE \le 0.081$).
   - `docs/Ke Hoach Thi Nghiem Va Do Luong.pdf` (5 trang): 3 tổ hợp Demo, 4 tổ hợp Full Combos (A, B, C, D), bài test cứu 48 câu mất bảng cầu nối, tốc độ biên dịch XGrammar 4–8 ms vs Outlines 2.0–4.9s, lấy mẫu 8–16 quỹ đạo ($T=0.7, p=0.9, k=50$), Cổng dừng sớm 0.85.
   - `docs/Phuong Phap Luan Kien Truc Candidate.pdf` (5 trang): Phân tích cặn kẽ 4 điểm nghẽn của MARS-SQL 21B và cơ chế giải quyết của từng Candidate (LitE-SQL 0.6B, Graph-Link, FINER-SQL 3B, XGrammar PDA, PV-SQL, Laya-421M 0-token).

Toàn bộ phát hiện đã được tổng hợp chi tiết thành báo cáo:
- Đường dẫn: `/mnt/c/Workspace/optimize/.agents/teamwork/spec_miner_survey_2/survey_report.md` (9 phần, 23 tính năng, 12 edge cases, công thức toán học và checklist nghiệm thu DoD).

---

## 2. Chuỗi Suy Luận (Logic Chain)

Từ các quan sát thực nghiệm trên, tôi rút ra chuỗi logic kỹ thuật chi phối việc lập trình toàn dự án:
1. **Tại sao phải đi tuần tự theo 4 Phase mà không được làm song song?**
   - Theo nguyên tắc bất biến trong Tài liệu 04: Phase 2 (Training scripts) phụ thuộc vào định dạng dữ liệu chuẩn của Phase 1 (`DATA-G1-EMB`, `DATA-G1-GRAPH`, `DATA-G2-SFT`, `data/mini/`).
   - Phase 3 (Thực thi huấn luyện Colab) cần script của Phase 2 và dữ liệu Phase 1.
   - Phase 4 (Evaluation Harness) cần checkpoints đã huấn luyện từ Phase 3 và schema graph của Phase 1.
   - Do đó, việc hoàn thành Phase 1 và `data/mini/` là điều kiện tiên quyết (Prerequisite Zero).
2. **Tại sao phải có bộ dữ liệu mẫu `data/mini/`?**
   - Dữ liệu thô BIRD nặng tới 33.4 GB và 95 database; không thể nạp và chạy kiểm thử smoke test nhanh trên môi trường phát triển cục bộ.
   - `data/mini/` gồm 100 câu BIRD, 1,000 câu SynSQL và 2 SQLite database (`california_schools.sqlite`, `financial.sqlite`) cho phép chạy smoke test toàn tuyến từ Data $\rightarrow$ SFT $\rightarrow$ GRPO $\rightarrow$ XGrammar $\rightarrow$ PV-SQL trong vòng < 2 phút.
3. **Tại sao MARS-SQL 21B bị mất bảng cầu nối và cách giải quyết?**
   - MARS-SQL Agent 1 duyệt từng bảng rời rạc theo từ khóa câu hỏi; bảng cầu nối trung gian không chứa từ khóa người dùng nên bị loại bỏ (hỏng 48 câu trên BIRD-dev).
   - Thuật toán Node-Weighted Steiner Tree (NWST) trên đồ thị tĩnh $G=(V, E)$ kết nối các cột Top-25 từ LitE-SQL sẽ tự động giữ lại toàn bộ các đỉnh trung gian nằm trên đường đi ngắn nhất giữa hai bảng chính, đảm bảo bảo toàn 100% (48/48 câu).
4. **Tại sao GRPO thông thường gây sụp đổ lợi thế (Advantage Collapse) trên mô hình nhỏ?**
   - Với phần thưởng nhị phân $\{0, 1\}$, khi mô hình 3B sinh 8 câu đều bị lỗi thực thi trên câu hỏi khó, tất cả nhận điểm 0 $\rightarrow$ phương sai bằng 0 $\rightarrow$ lợi thế $A_j = 0$ $\rightarrow$ gradient biến mất.
   - Hàm thưởng 4 đầu điểm chi tiết của FINER-SQL ($R_{\text{format}} + R_{\text{exec}} + R_{\text{mem}} + R_{\text{atomic}}$) luôn tạo ra chênh lệch điểm số $A_j \neq 0$ nhờ điểm tư duy hợp lý $R_{\text{mem}}$ và tỷ lệ mảnh ghép toán tử $R_{\text{atomic}}$, giúp mô hình 3B tiếp tục học và đạt 67.73% EX.
5. **Tại sao XGrammar PDA vượt trội Outlines và cách khắc phục Hollow Rescue?**
   - XGrammar dùng bộ nhớ đệm bitmask (>99% token tĩnh), biên dịch DDL chỉ mất 4–8 ms (nhanh hơn Outlines 250–1,000 lần) và chỉ giảm 1.6%–3.7% throughput.
   - Để chống "Giải cứu rỗng" (câu lệnh đúng cú pháp nhưng sai ngữ nghĩa câu hỏi), XGrammar bắt buộc phải đi kèm LLM-JEPA (hiểu đúng ở Agent 2) và PV-SQL 10 quy tắc AST (bắt lỗi logic ở Agent 3).
6. **Tại sao kiểm định 0-token là chìa khóa giảm độ trễ?**
   - Thay vì nạp thêm một mô hình 7B thứ ba để đọc lại câu lệnh và sinh chữ "Yes" (tốn thêm 5–10s và hàng nghìn token), Laya-421M (ModernBERT) chấm điểm song song 8–16 ứng viên trong ~1.2 ms qua token `[MASK]`, kết hợp Cổng dừng sớm 0.85 giúp giảm 77% thời gian toàn tuyến.

---

## 3. Điều Kiện Giới Hạn & Cảnh Báo (Caveats)

1. **Phụ thuộc thư viện XGrammar trên CPU**:
   - `xgrammar` được tối ưu hóa cho CUDA GPU. Khi chạy trên môi trường CPU cục bộ (như WSL hoặc máy tính không có card NVIDIA), một số hàm C++ bitmask có thể không tương thích. Mã nguồn bắt buộc phải có khối lệnh `try...except` để chuyển sang `Native Decoding` kèm cảnh báo.
2. **Cơ sở dữ liệu SQLite trong `data/mini/`**:
   - Phải đảm bảo file SQLite là database chuẩn, có đầy đủ dữ liệu thực tế trong các bảng và tương thích với 100 câu hỏi mẫu BIRD trích xuất, để phép chạy thử (probing và execution accuracy) không bị trả về bảng rỗng do thiếu dữ liệu bản ghi.
3. **Giới hạn thời gian phiên làm việc Google Colab**:
   - Colab Free/Pro ngắt phiên sau 12 giờ hoặc khi mất kết nối mạng. Cơ chế `DriveCheckpointCallback` và `--resume_from_checkpoint latest` là bắt buộc phải được kiểm thử cẩn thận trong smoke test.

---

## 4. Kết Luận (Conclusion)

- Toàn bộ các yêu cầu, lộ trình, thuật toán, công thức toán học và thông số kỹ thuật cho cả 4 Phase đã được khai thác và hệ thống hóa trọn vẹn trong báo cáo `/mnt/c/Workspace/optimize/.agents/teamwork/spec_miner_survey_2/survey_report.md`.
- Các câu hỏi khảo sát từ Orchestrator về:
  - Phase 1 (Data models, schemas, PK-FK Graph, Steiner Tree, vector embeddings, `data/mini/`);
  - Phase 2 (Completion-Only Loss, LLM-JEPA 2-block mask, Dense GRPO 4 phần thưởng, LoRA/QLoRA, Drive sync);
  - Phase 3 (Google Colab notebooks 3B/7B, phân cấp phần cứng, tự động mount Drive, khôi phục session);
  - Phase 4 (Lấy mẫu 8–16 quỹ đạo, chấm điểm 0-token Laya-421M, XGrammar PDA & fallback, 10 quy tắc PV-SQL, bộ metrics EX, VES, VSR, TPR, ECE)
  đều đã được giải trình cặn kẽ với đầy đủ thông số định lượng và mã nguồn tham chiếu.
- Đội ngũ triển khai (Data-G1, Gen-G2, Val-Bench) có thể sử dụng trực tiếp báo cáo này làm hợp đồng kỹ thuật để bắt tay lập trình mà không gặp bất kỳ điểm mù nào về kiến trúc.

---

## 5. Phương Pháp Xác Minh Độc Lập (Verification Method)

Người nhận hoặc các tác tử tiếp theo có thể độc lập xác minh tính chuẩn xác của báo cáo bằng các bước sau:
1. **Kiểm tra sự tồn tại và tính đầy đủ của báo cáo**:
   ```bash
   test -f /mnt/c/Workspace/optimize/.agents/teamwork/spec_miner_survey_2/survey_report.md && echo "Report exists"
   wc -l /mnt/c/Workspace/optimize/.agents/teamwork/spec_miner_survey_2/survey_report.md
   ```
2. **Đối chiếu các thông số kỹ thuật với tài liệu gốc**:
   - Kiểm tra thông số LLM-JEPA: `grep -i "LD = 0.5" docs/specifications/*.md docs/*.pdf` (hoặc đối chiếu trang 2 của `Ke Hoach Huan Luyen Mo Hinh.pdf`).
   - Kiểm tra 4 đầu thưởng GRPO: đối chiếu mục 3.2 trang 3 của `Ke Hoach Huan Luyen Mo Hinh.pdf`.
   - Kiểm tra 10 quy tắc AST: đối chiếu mục 4.1 của `docs/specifications/02_architecture_design.md`.
   - Kiểm tra tốc độ biên dịch XGrammar (4–8 ms): đối chiếu trang 2 của `Ke Hoach Thi Nghiem Va Do Luong.pdf`.
   - Kiểm tra 6 trường của thẻ cột: đối chiếu mục 3.1 trang 2 của `Ke Hoach Xu Ly Du Lieu.pdf`.
