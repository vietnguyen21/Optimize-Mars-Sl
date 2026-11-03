## 2026-11-02T07:31:26Z

Bạn là teamwork_challenger chịu trách nhiệm kiểm thử đối kháng (Adversarial Verification & Stress Testing) cho Milestone M5 (Automated Smoke Testing & Comprehensive Verification).

Thư mục làm việc của bạn: /mnt/c/Workspace/optimize/.agents/teamwork/challenger_m5/
(Mọi tệp tin kiểm thử đối kháng, kịch bản stress-test, và báo cáo của bạn PHẢI lưu trong thư mục này. TUYỆT ĐỐI không sửa đổi mã nguồn trong src/, scripts/ hay tests/).

Nhiệm vụ:
1. Đọc bắt buộc các tài liệu nền tảng:
   - /mnt/c/Workspace/optimize/.agents/teamwork/ORIGINAL_REQUEST.md (BẮT BUỘC đọc)
   - /mnt/c/Workspace/optimize/PROJECT.md
   - /mnt/c/Workspace/optimize/TEST_INFRA.md
   - /mnt/c/Workspace/optimize/.agents/teamwork/worker_m5/handoff.md
2. Thiết kế và thực thi kịch bản kiểm thử đối kháng cho M5:
   - Stress-test `scripts/run_smoke_tests.py`:
     * Kiểm tra hành vi khi chạy trên terminal khác nhau hoặc khi stdout bị redirect sang pipe (`python3 scripts/run_smoke_tests.py | cat`).
     * Kiểm tra xem script có bắt đúng lỗi và trả về exit code khác 0 khi một test suite bị cố tình làm fail (có thể test bằng mock hoặc test riêng trong thư mục của challenger).
     * Kiểm tra thời gian thực thi: đo lường thời gian chạy và xác nhận tổng thời gian < 60 giây (mục tiêu thực tế ~5s).
   - Kiểm tra hồi quy quy tắc PV-SQL sau tinh chỉnh:
     * Kiểm tra câu lệnh có `distinct_clients` không còn bị Rule 1 coi là có `DISTINCT`.
     * Kiểm tra câu lệnh có `WHERE age > 20` nhưng không có `IS NOT NULL` / `IS NULL` khi câu hỏi hỏi về null có bị Rule 8 bắt lỗi chính xác không.
   - Kiểm tra tham số CLI `--model_path` của `evaluate_end_to_end.py`:
     * Chạy thử với `--model_path non_existent_model` xem có thông báo lỗi hợp lệ hoặc xử lý an toàn không.
   - Chạy toàn bộ 158 tests hiện có: `python3 -m unittest discover tests/ -v`.
3. Viết báo cáo đối kháng chi tiết vào `/mnt/c/Workspace/optimize/.agents/teamwork/challenger_m5/handoff.md` theo cấu trúc: Observation, Logic Chain, Caveats, Conclusion, Verification Method. Đưa ra phán quyết rõ ràng: APPROVE hoặc REQUEST_CHANGES.
4. Gửi thông điệp hoàn thành kèm phán quyết về cho Orchestrator (Conversation ID: 3dfd88ef-5709-46d6-ba23-355dc6f4af89) bằng công cụ send_message.
