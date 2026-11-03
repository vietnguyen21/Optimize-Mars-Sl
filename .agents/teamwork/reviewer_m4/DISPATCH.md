## 2026-11-02T06:58:33Z
Bạn là teamwork_reviewer chịu trách nhiệm rà soát, đánh giá độc lập mã nguồn và chất lượng kỹ thuật của Milestone M4 (Phase 4: Validation, Grammar PDA & Benchmark) theo yêu cầu trong AGENTS_RESEARCH.md và docs/specifications/.

Thư mục làm việc của bạn: /mnt/c/Workspace/optimize/.agents/teamwork/reviewer_m4/
(Mọi tệp tin trạng thái, tiến độ và báo cáo của bạn PHẢI lưu trong thư mục này. TUYỆT ĐỐI không sửa đổi mã nguồn trong src/ hay tests/).

Nhiệm vụ:
1. Đọc bắt buộc các tài liệu nền tảng:
   - /mnt/c/Workspace/optimize/.agents/teamwork/ORIGINAL_REQUEST.md (BẮT BUỘC đọc)
   - /mnt/c/Workspace/optimize/PROJECT.md
   - /mnt/c/Workspace/optimize/TEST_INFRA.md
   - /mnt/c/Workspace/optimize/.agents/teamwork/worker_m4/handoff.md
2. Kiểm tra chi tiết các tệp mã nguồn M4 do worker_m4 tạo:
   - src/grammar/: base.py, fallback_engine.py, xgrammar_engine.py, __init__.py
   - src/validation/: pv_sql_rules.py, db_probe.py, laya_scorer.py, security_filter.py, __init__.py
   - src/evaluation/: candidate_generator.py, metrics.py, evaluate_end_to_end.py, __init__.py
   - tests/: test_grammar_and_validation.py, test_evaluation.py
3. Thực hiện kiểm tra thực nghiệm độc lập:
   - Chạy toàn bộ 154 tests: `python3 -m unittest discover tests/ -v`
   - Chạy benchmark harness: `python3 src/evaluation/evaluate_end_to_end.py --dataset data/mini/mini_bird.jsonl --db_dir data/mini --num_candidates 8`
4. Rà soát chuyên sâu các tiêu chí kỹ thuật:
   - XGrammar PDA & Fallback: Pushdown Automaton, bitmask caching theo SHA256 DDL, fallback an toàn không bao giờ crash.
   - 10 quy tắc PV-SQL AST: Distinct, Top-K, Average, Count, Sum, Extreme, Comparison, Null Handling, Join Validity, Group By Completeness.
   - DBProbeEngine: 0 GB VRAM, Read-Only tuyệt đối, giới hạn LIMIT 5, quota max_queries=5.
   - Laya-421M Scorer: ModernBERT System-1 0-token parallel scoring, temperature scaling, ECE <= 0.081, Adaptive Early Exit Gate >= 0.85.
   - Candidate rollouts (8-16 candidates) & metrics (EX multiset matching, VSR, ECE, latency, memory).
5. Viết báo cáo chuyển giao chi tiết vào `/mnt/c/Workspace/optimize/.agents/teamwork/reviewer_m4/handoff.md` theo cấu trúc: Observation, Logic Chain, Caveats, Conclusion, Verification Method. Đưa ra phán quyết rõ ràng: APPROVE hoặc REQUEST_CHANGES.
6. Gửi thông điệp hoàn thành kèm phán quyết về cho Orchestrator (Conversation ID: 3dfd88ef-5709-46d6-ba23-355dc6f4af89) bằng công cụ send_message.
