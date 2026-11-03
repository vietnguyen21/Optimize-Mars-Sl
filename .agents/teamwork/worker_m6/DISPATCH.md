## 2026-11-02T07:54:59Z
Bạn là teamwork_worker đảm nhận vai trò [DIRECTOR-LEADER & SYSTEM WRITER] Tác tử Soạn thảo Tài liệu Bàn giao Kỹ thuật (Technical Handover Author).

Thư mục làm việc của bạn: /mnt/c/Workspace/optimize/.agents/teamwork/worker_m6/
(Mọi tệp tin ghi chú, kế hoạch, tiến độ, và bản nháp của bạn PHẢI lưu trong thư mục này).

Quyền sở hữu tệp tin độc quyền (Exclusive File Ownership):
- /mnt/c/Workspace/optimize/HANDOVER.md (tạo mới tài liệu bàn giao kỹ thuật tại thư mục gốc của dự án)
- /mnt/c/Workspace/optimize/scripts/run_smoke_tests.py (bổ sung `os.chdir(REPO_ROOT)` theo khuyến nghị của Reviewer M5)

CẢNH BÁO LIÊM CHÍNH BẮT BUỘC (MANDATORY INTEGRITY WARNING):
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Nhiệm vụ:
1. Đọc bắt buộc các tài liệu nền tảng:
   - /mnt/c/Workspace/optimize/.agents/teamwork/ORIGINAL_REQUEST.md (BẮT BUỘC đọc)
   - /mnt/c/Workspace/optimize/AGENTS_RESEARCH.md
   - /mnt/c/Workspace/optimize/PROJECT.md
   - /mnt/c/Workspace/optimize/TEST_INFRA.md
   - /mnt/c/Workspace/optimize/.agents/teamwork/reviewer_m5/handoff.md (đặc biệt Finding 1 về os.chdir(REPO_ROOT))
   - /mnt/c/Workspace/optimize/.agents/teamwork/worker_m5/handoff.md
2. Áp dụng tinh chỉnh nhỏ cho `scripts/run_smoke_tests.py`:
   - Thêm `os.chdir(REPO_ROOT)` ngay sau khi định nghĩa `REPO_ROOT` để script chạy an toàn kể cả khi người dùng đứng ở bất kỳ thư mục nào ngoài repo.
   - Chạy thử nghiệm để đảm bảo script vẫn chạy mượt mà 100%.
3. Soạn thảo tài liệu bàn giao kỹ thuật toàn diện `HANDOVER.md` tại thư mục gốc `/mnt/c/Workspace/optimize/HANDOVER.md` (viết bằng tiếng Việt chuẩn mực, chuyên nghiệp, cấu trúc rõ ràng) bao gồm đầy đủ các phần:
   - **Phần 1: Giới thiệu & Kiến trúc Đội ngũ Triển khai**:
     * Tổng quan dự án: Thay thế kiến trúc MARS-SQL 21B cồng kềnh bằng pipeline SLM Text-to-SQL (2.5B–4.0B) hiệu năng cao.
     * Phân vai và trách nhiệm của 4 tác tử theo `AGENTS_RESEARCH.md`:
       1. `Director & Project Leader`: Điều phối SSoT, phân rã kiến trúc, hợp đồng giao diện, kiểm soát liêm chính.
       2. `[AGENT-DATA-G1] Kỹ sư Data Pipeline & Schema Grounding`: Đồ thị PK-FK, Node-Weighted Steiner Tree, Column Cards, Vector Embedding Retriever, dữ liệu `data/mini/`.
       3. `[AGENT-GEN-G2] Kỹ sư SLM Training & Constrained Decoding`: Completion-Only Loss (-100 masking), mô-đun LLM-JEPA (2-block causal mask, token `[PRED]`), Dense GRPO (4 reward heads: EX, AST, Column, Table), tự động mount Google Drive và checkpointing, cùng 5 Colab Notebooks (01-05) hỗ trợ cả 3B (FINER-SQL) và 7B (Qwen2.5-Coder-7B).
       4. `[AGENT-VAL-BENCH] Kỹ sư System-1 Validation & Benchmark`: XGrammar PDA & Fallback 100% crash-free, PV-SQL 10 AST rules, DB Probe (0 GB VRAM, `LIMIT 5`), ModernBERT System-1 Laya-421M 0-token parallel scorer ($ECE \le 0.081$, early exit gate $\ge 0.85$), sinh 8–16 ứng viên, multiset EX metrics, và bộ smoke test `scripts/run_smoke_tests.py`.
   - **Phần 2: Danh mục Toàn bộ Tệp Mã Nguồn (Codebase Inventory & File Map)**:
     * Bảng danh mục chi tiết mọi tệp tin trong `src/`, `notebooks/`, `scripts/`, `data/mini/`, `tests/`.
     * Cung cấp thông tin: Đường dẫn tệp, Agent sở hữu, Mô tả chức năng chính, Giao diện/Hàm cốt lõi, Đầu vào (Inputs) & Đầu ra (Outputs).
   - **Phần 3: Hướng dẫn Thiết lập Môi trường (Environment Setup Guide)**:
     * Hai tầng môi trường rõ ràng:
       a. Cục bộ CPU (Linux/WSL2, Python 3.12, virtualenv `.venv`, requirements.txt, zero-download offline test).
       b. Google Colab (GPU T4/L4/A100, cài đặt gói phụ thuộc, flash-attn, xgrammar, cấu hình đường dẫn Google Drive `/content/drive/MyDrive/SLM_Text2SQL_Checkpoints/`).
   - **Phần 4: Quy trình Vận hành & Huấn luyện Từng bước (Step-by-Step Reproduction Guide)**:
     * Hướng dẫn chi tiết thứ tự chạy 5 Notebooks trên Colab:
       - Notebook 01: Chuẩn bị dữ liệu và huấn luyện Embedding Retriever.
       - Notebook 02: Huấn luyện SFT Generator và LLM-JEPA (cấu hình 3B vs 7B).
       - Notebook 03: Tinh chỉnh tăng cường Dense GRPO với 4 hàm phần thưởng.
       - Notebook 04: Huấn luyện mô hình ra quyết định Laya-421M và hiệu chuẩn nhiệt độ ECE.
       - Notebook 05: Đánh giá toàn tuyến End-to-End Benchmark.
     * Hướng dẫn cơ chế tự động lưu Checkpoint sang Google Drive và phục hồi khi bị ngắt kết nối (`--resume_from_checkpoint latest`).
   - **Phần 5: Hướng dẫn Kiểm thử Tự động & Nghiệm thu Kỹ thuật (Smoke Testing & Acceptance)**:
     * Cách thực thi `python3 scripts/run_smoke_tests.py` (<60s, thực tế ~5s).
     * Cách chạy toàn bộ 158 unit tests: `python3 -m unittest discover tests/ -v`.
     * Giải thích chi tiết các chỉ số khoa học: EX (Multiset match), VSR (100%), ECE ($\le 0.081$), End-to-End Latency, GPU VRAM Footprint.
   - **Phần 6: Bảng Đối sánh Khoa học với các Baseline (MARS-SQL 21B, Combo A, Combo C)**:
     * Phân tích số liệu thực nghiệm, độ trễ, mức tiêu thụ VRAM và chi phí tính toán.
   - **Phần 7: Khuyến nghị & Lộ trình Nghiên cứu Tiếp theo cho Đội ngũ Phân tích Số liệu**:
     * Mở rộng từ mini dataset sang BIRD-dev đầy đủ (1,534 câu).
     * Các kịch bản Ablation Study đề xuất (Loại bỏ JEPA, loại bỏ GRPO, biến thiên $K \in \{1, 4, 8, 16\}$, tắt XGrammar).
4. Kiểm tra độc lập:
   - Đảm bảo tệp `/mnt/c/Workspace/optimize/HANDOVER.md` được tạo lập đầy đủ, đẹp mắt, có mục lục Markdown rõ ràng.
   - Chạy `python3 scripts/run_smoke_tests.py` và `python3 -m unittest discover tests/ -v` để xác nhận không phát sinh lỗi.
5. Viết báo cáo chuyển giao kỹ thuật vào `/mnt/c/Workspace/optimize/.agents/teamwork/worker_m6/handoff.md`.
6. Gửi tin nhắn hoàn thành về cho Orchestrator (Conversation ID: 3dfd88ef-5709-46d6-ba23-355dc6f4af89) bằng công cụ send_message.
