## 2026-11-02T06:58:36Z

Bạn là teamwork_challenger chịu trách nhiệm kiểm thử đối kháng (Adversarial Verification & Stress Testing) cho Milestone M4 (Phase 4: Validation, Grammar PDA & Benchmark) theo AGENTS_RESEARCH.md và docs/specifications/.

Thư mục làm việc của bạn: /mnt/c/Workspace/optimize/.agents/teamwork/challenger_m4/
(Mọi tệp tin kiểm thử đối kháng, kịch bản stress-test, và báo cáo của bạn PHẢI lưu trong thư mục này. TUYỆT ĐỐI không sửa đổi mã nguồn trong src/ hay tests/).

Nhiệm vụ:
1. Đọc bắt buộc các tài liệu nền tảng:
   - /mnt/c/Workspace/optimize/.agents/teamwork/ORIGINAL_REQUEST.md (BẮT BUỘC đọc)
   - /mnt/c/Workspace/optimize/PROJECT.md
   - /mnt/c/Workspace/optimize/TEST_INFRA.md
   - /mnt/c/Workspace/optimize/.agents/teamwork/worker_m4/handoff.md
2. Thiết kế và thực thi kịch bản kiểm thử đối kháng (stress tests, fuzzing, corner cases):
   - Stress-test XGrammar PDA & Fallback: Bơm các chuỗi SQL dị dạng (dấu ngoặc đơn lồng nhau mất cân bằng, chuỗi nháy chưa đóng, chấm phẩy kép lặp, toán tử treo cuối câu, mệnh đề lộn xộn). Xác minh fallback bảo đảm 100% crash-free và chỉ trả về SQL cú pháp hợp lệ.
   - Thử thách 10 quy tắc PV-SQL AST: Thử nghiệm các câu hỏi mẹo và truy vấn thiếu điều kiện (thiếu DISTINCT, thiếu ORDER BY LIMIT, thiếu AVG/COUNT/SUM, thiếu điều kiện JOIN ON, cột không tổng hợp thiếu GROUP BY). Kiểm tra xem validator có bắt lỗi chính xác 100% không.
   - Kiểm thử bảo mật DB Probe: Cố tình tiêm các truy vấn phá hủy (DROP TABLE, DELETE, UPDATE, CREATE, stacked queries, PRAGMA) và các truy vấn trả về hàng triệu dòng. Xác minh DBProbeEngine tuyệt đối từ chối và luôn cưỡng chế LIMIT 5.
   - Kiểm thử Laya-421M Scorer: Kiểm tra tính toán phân phối xác suất, nhiệt độ, sai số hiệu chuẩn ECE <= 0.081, và ngưỡng thoát sớm early exit gate >= 0.85.
   - Kiểm thử Candidate Generator & Metrics: Kiểm thử multiset comparison trong EX (thay đổi thứ tự hàng ngẫu nhiên, số thực làm tròn), VSR, độ trễ và bộ nhớ.
3. Chạy toàn bộ 154 tests hiện có trong dự án: `python3 -m unittest discover tests/ -v`.
4. Viết báo cáo đối kháng chi tiết vào `/mnt/c/Workspace/optimize/.agents/teamwork/challenger_m4/handoff.md` theo cấu trúc: Observation, Logic Chain, Caveats, Conclusion, Verification Method. Đưa ra phán quyết rõ ràng: APPROVE hoặc REQUEST_CHANGES.
5. Gửi thông điệp hoàn thành kèm phán quyết về cho Orchestrator (Conversation ID: 3dfd88ef-5709-46d6-ba23-355dc6f4af89) bằng công cụ send_message.
