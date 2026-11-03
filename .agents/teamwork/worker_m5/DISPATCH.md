## 2026-11-02T07:13:15Z
Bạn là teamwork_worker đảm nhận vai trò [AGENT-VAL-BENCH] Kỹ sư Kiểm thử và Nghiệm thu Toàn diện (Comprehensive Smoke Test & Verification Engineer).

Thư mục làm việc của bạn: /mnt/c/Workspace/optimize/.agents/teamwork/worker_m5/
(Mọi tệp tin ghi chú, kế hoạch, tiến độ, và báo cáo của bạn PHẢI lưu trong thư mục này).

Quyền sở hữu tệp tin độc quyền (Exclusive File Ownership):
- scripts/run_smoke_tests.py (tạo mới kịch bản nghiệm thu độc lập toàn diện)
- src/evaluation/evaluate_end_to_end.py (tinh chỉnh kết nối AbstentionManager và bổ sung --model_path theo khuyến nghị Reviewer M4)
- src/validation/pv_sql_rules.py (tinh chỉnh ranh giới từ \bDISTINCT\b và Rule 8 theo khuyến nghị Reviewer M4)
- src/evaluation/__init__.py (export bổ sung normalize_query_result)
- tests/ (nếu cần bổ sung test case nghiệm thu)

CẢNH BÁO LIÊM CHÍNH BẮT BUỘC (MANDATORY INTEGRITY WARNING):
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Nhiệm vụ:
1. Đọc bắt buộc các tài liệu nền tảng:
   - /mnt/c/Workspace/optimize/.agents/teamwork/ORIGINAL_REQUEST.md (BẮT BUỘC đọc)
   - /mnt/c/Workspace/optimize/PROJECT.md
   - /mnt/c/Workspace/optimize/TEST_INFRA.md
   - /mnt/c/Workspace/optimize/.agents/teamwork/reviewer_m4/handoff.md (đọc các khuyến nghị Finding 1, 2, 3)
   - /mnt/c/Workspace/optimize/.agents/teamwork/challenger_m4/handoff.md
2. Áp dụng các cải tiến chất lượng từ Reviewer/Challenger:
   - Trong `src/evaluation/evaluate_end_to_end.py`: Kết nối `self.abstention_manager.should_abstain(...)` khi các ứng viên có độ tự tin thấp hoặc vi phạm an toàn; ghi nhận số lượng `total_abstentions` vào kết quả; bổ sung tham số CLI `--model_path` (mặc định None) để hỗ trợ nạp checkpoint mô hình trên Colab.
   - Trong `src/validation/pv_sql_rules.py`: Sử dụng regex `re.search(r"\bDISTINCT\b", ...)` thay cho substring `in` để tránh false positive khi tên bảng/cột có chữ "distinct"; loại bỏ "WHERE" thuần túy khỏi danh sách kiểm tra của Rule 8 (`check_null_handling_rule`).
   - Trong `src/evaluation/__init__.py`: Export hàm `normalize_query_result`.
3. Xây dựng kịch bản nghiệm thu tự động toàn tuyến `scripts/run_smoke_tests.py`:
   - Chạy độc lập từ dòng lệnh: `python3 scripts/run_smoke_tests.py`
   - Thực thi tuần tự và đo lường thời gian của các khối:
     * Suite 1: Data Pipeline & Schema Grounding (`tests/test_grounding.py` - PK-FK Graph, Steiner Tree, Column Cards, Embedding Retriever, data/mini/)
     * Suite 2: SLM Training Scripts (`tests/test_training.py` - Completion-Only Loss Masking, LLM-JEPA 2-block causal mask với [PRED], Dense GRPO 4 reward heads, Callbacks)
     * Suite 3: Colab Training Notebooks (`tests/test_notebooks.py` - dynamic cell execution toàn bộ 49 code cells qua nbformat/exec)
     * Suite 4: Grammar PDA & Validation Rules (`tests/test_grammar_and_validation.py` - XGrammar PDA, Fallback Engine 100% crash-free, 10 PV-SQL AST rules, DB Probe 0 GB VRAM & LIMIT 5, Laya-421M Scorer)
     * Suite 5: Evaluation & Benchmark Suite (`tests/test_evaluation.py` - Candidate Generator 8-16 rollouts, Multiset EX comparison, VSR, ECE, Latency, Memory)
     * Suite 6: End-to-End Benchmark Execution on `data/mini/mini_bird.jsonl` (gọi trực tiếp `EvaluationHarness` với 24 câu hỏi mẫu, kiểm tra `EX >= 75%`, `VSR = 100%`, `ECE <= 0.081`, `Latency < 5.8s`)
   - Toàn bộ quá trình chạy phải hoàn tất trong < 60 giây trên môi trường CPU cục bộ.
   - Định dạng hiển thị chuyên nghiệp: In bảng tổng kết ANSI/ASCII đẹp mắt liệt kê từng Suite, số lượng test case, thời gian chạy, và trạng thái PASSED / FAILED.
   - Trả về mã thoát (exit code) 0 nếu tất cả đều thành công; mã khác 0 nếu có lỗi.
4. Chạy thực nghiệm kiểm chứng:
   - Chạy `python3 scripts/run_smoke_tests.py`
   - Chạy `python3 -m unittest discover tests/ -v`
   - Xác nhận không có lỗi cú pháp, không thiếu hụt thư viện, 100% test pass.
5. Viết báo cáo chuyển giao kỹ thuật chi tiết vào `/mnt/c/Workspace/optimize/.agents/teamwork/worker_m5/handoff.md` theo cấu trúc: Observation, Logic Chain, Caveats, Conclusion, Verification Method.
6. Gửi tin nhắn hoàn thành về cho Orchestrator (Conversation ID: 3dfd88ef-5709-46d6-ba23-355dc6f4af89) bằng công cụ send_message.
