# Dispatch History

## 2026-11-02T04:04:31Z

Tổ chức đội ngũ triển khai theo file AGENTS_RESEARCH.md (gồm Director & Project Leader điều phối trung tâm, Kỹ sư Data Pipeline & Schema Grounding, Kỹ sư SLM Training & Constrained Decoding, và Kỹ sư System-1 Validation & Benchmark) để lập trình toàn bộ mã nguồn cho dự án theo các tài liệu đặc tả trong thư mục docs/specifications/. Sau khi kết thúc triển khai, tạo file HANDOVER.md tại thư mục gốc giải thích những gì các agent đã làm cũng như hướng dẫn chi tiết cách sử dụng các file code để bàn giao từ đội ngũ lập trình sang đội ngũ phân tích và nghiên cứu số liệu.

Working directory: /mnt/c/Workspace/optimize
Integrity mode: development

## Requirements

### R1. Triển khai kiến trúc đội ngũ và toàn bộ mã nguồn dự án
- Vận hành theo mô hình phân vai và cơ chế điều phối "Director-Led Single Source of Truth" theo AGENTS_RESEARCH.md.
- Triển khai đầy đủ mã nguồn cho 4 Phase theo đúng kiến trúc và lộ trình trong docs/specifications/:
  - Phase 1: Data Pipeline & Schema Grounding (PK-FK Graph, Steiner Tree, vector embedding) cùng bộ dữ liệu mẫu kiểm thử nhỏ tại data/mini/.
  - Phase 2: Training Scripts trong src/training/ (Completion-Only Loss, LLM-JEPA, GRPO).
  - Phase 3: Notebooks huấn luyện trên Google Colab trong notebooks/, hỗ trợ cả 3B (FINER-SQL) và 7B (Qwen2.5-Coder-7B), tích hợp cơ chế lưu checkpoint tự động sang Google Drive.
  - Phase 4: Evaluation & Benchmark trong src/evaluation/ (sinh 8–16 quỹ đạo ứng viên phục vụ Ablation Study, chấm điểm 0-token decoding, cơ chế ràng buộc ngữ pháp XGrammar PDA kèm fallback).

### R2. Xây dựng bộ kiểm thử và nghiệm thu tự động (Smoke Test)
- Xây dựng bộ dữ liệu nhỏ gọn tại data/mini/ và kịch bản kiểm thử tự động (smoke tests / unit tests) để xác minh độc lập toàn bộ luồng xử lý từ tiền xử lý dữ liệu, chuẩn bị huấn luyện, cơ chế giải mã ràng buộc XGrammar PDA (kèm fallback), đến chấm điểm và đánh giá.
- Toàn bộ các bài kiểm thử cục bộ phải vượt qua thành công mà không phụ thuộc vào việc phải chạy huấn luyện mô hình thực tế kéo dài.

### R3. Lập tài liệu bàn giao kỹ thuật (Handover Documentation)
- Tạo file HANDOVER.md tại thư mục gốc của dự án.
- Tài liệu phải tổng hợp và giải thích rõ ràng những gì từng vai trò tác tử (Director-Leader, Data-G1, Gen-G2, Val-Bench) đã thực hiện.
- Hướng dẫn chi tiết, rõ ràng cách sử dụng từng file code, cách thiết lập môi trường, và quy trình từng bước để đội ngũ nghiên cứu có thể tiếp quản, chạy huấn luyện trên Google Colab và thực hiện phân tích số liệu thí nghiệm.

## Acceptance Criteria

### Cấu trúc mã nguồn và Pipeline
- [ ] Cấu trúc thư mục dự án tuân thủ đầy đủ đặc tả tại docs/specifications/ (gồm src/, notebooks/, tests/ hoặc script kiểm thử, data/mini/).
- [ ] Các notebook trong notebooks/ sẵn sàng chạy trên Google Colab, có sẵn cấu hình cho cả 3B và 7B và tự động mount Google Drive để lưu checkpoint.
- [ ] Logic sinh SQL hỗ trợ cấu hình 8–16 ứng viên và tích hợp cơ chế XGrammar PDA cùng fallback xử lý lỗi cú pháp.

### Kiểm thử và Xác minh độc lập
- [ ] Thư mục data/mini/ chứa dữ liệu mẫu hợp lệ cho smoke test.
- [ ] Bộ smoke test chạy thành công từ đầu đến cuối trên dữ liệu data/mini/ mà không phát sinh lỗi cú pháp hay thiếu hụt phụ thuộc.

### Tài liệu bàn giao
- [ ] File HANDOVER.md tồn tại tại thư mục gốc của dự án.
- [ ] HANDOVER.md liệt kê đầy đủ danh mục các file code đã tạo, vai trò thực hiện của các agent theo AGENTS_RESEARCH.md.
- [ ] HANDOVER.md cung cấp hướng dẫn từng bước (setup môi trường, chuẩn bị dữ liệu, chạy Colab notebooks, chạy benchmark) để đội ngũ nghiên cứu vận hành độc lập.
