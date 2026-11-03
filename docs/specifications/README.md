# BỘ TÀI LIỆU ĐẶC TẢ VÀ KIẾN TRÚC DỰ ÁN (PROJECT SPECIFICATION SUITE)

> **Dự án**: Nghiên cứu hệ thống đa tác tử ngôn ngữ nhỏ (SLM Multi-Agent Text-to-SQL) thay thế MARS-SQL 21B  
> **Mục tiêu**: Đạt hiệu năng tương đương hoặc vượt qua hệ thống 21 tỷ tham số (3 × 7B) với chi phí tính toán giảm >80%, bộ nhớ VRAM tối ưu và giải quyết dứt điểm các lỗi cố hữu (mất bảng cầu nối, ảo giác cú pháp, phán quyết cảm tính).

---

## 📚 Danh Mục Tài Liệu Kỹ Thuật

Bộ tài liệu này được thiết kế nhằm cung cấp bức tranh toàn cảnh, chi tiết và có thể chuyển giao cho toàn bộ đội ngũ nghiên cứu, kỹ sư AI và các bên liên quan:

| STT | Tài Liệu | Nội Dung Chính |
| :---: | :--- | :--- |
| **01** | [01_project_specification.md](./01_project_specification.md) | **Đặc tả dự án toàn diện**: Bối cảnh học thuật, bài toán Text-to-SQL, 4 nút thắt của MARS-SQL 21B, giải pháp cốt lõi và các mốc bàn giao. |
| **02** | [02_architecture_design.md](./02_architecture_design.md) | **Thiết kế kiến trúc hệ thống**: Chi tiết giải phẫu 3 Tác tử (Grounding, Generation, Validation), cơ chế Cây Steiner, XGrammar PDA và mô hình quyết định 0-token. |
| **03** | [03_project_requirements.md](./03_project_requirements.md) | **Yêu cầu kỹ thuật & Tiêu chí nghiệm thu**: Các chỉ số cốt lõi (EX, VES, Recall), ma trận 4 cấu hình Combo A/B/C/D và chuẩn bài báo nghiên cứu. |
| **04** | [04_roadmap_and_phases.md](./04_roadmap_and_phases.md) | **Lộ trình 4 Phase chi tiết**: Quy định nghiêm ngặt thứ tự thực hiện trước/sau (Phase 1: Data → Phase 2: Scripts → Phase 3: Training → Phase 4: Experiments). |
| **05** | [05_environment_and_colab_guide.md](./05_environment_and_colab_guide.md) | **Hướng dẫn môi trường & Google Colab**: Thiết lập GPU Colab, quy trình đồng bộ tự động Google Drive, logging và cơ chế phục hồi checkpoint. |
| **06** | [requirements.md](./requirements.md) | **Đặc tả thư viện & Gói phụ thuộc**: Danh mục chi tiết các gói Python (PyTorch, Transformers, XGrammar, TRL, NetworkX, SQLGlot) và ma trận tương thích CUDA. |

---

## 🧭 Hướng Dẫn Tiếp Cận Nhanh Cho Thành Viên Mới

1. **Hiểu bài toán và mục tiêu**: Bắt đầu bằng [01_project_specification.md](./01_project_specification.md) và [GLOSSARY.md](../../GLOSSARY.md).
2. **Nắm bắt luồng kỹ thuật**: Đọc [02_architecture_design.md](./02_architecture_design.md) để hiểu cách 3 tác tử phối hợp với nhau.
3. **Tuân thủ quy trình triển khai**: Xem kỹ [04_roadmap_and_phases.md](./04_roadmap_and_phases.md) để biết giai đoạn hiện tại cần làm gì, tuyệt đối không nhảy cóc sang giai đoạn sau khi giai đoạn trước chưa nghiệm thu.
4. **Chuẩn bị môi trường huấn luyện**: Đọc [05_environment_and_colab_guide.md](./05_environment_and_colab_guide.md) để thiết lập notebook và Google Drive trên Google Colab.
