# ĐẶC TẢ DỰ ÁN TOÀN DIỆN (PROJECT SPECIFICATION)

## 1. Tóm Tắt Dự Án (Executive Summary)

Dự án này là một công trình **nghiên cứu khoa học thực nghiệm (Empirical Research Paper)** nhằm giải quyết bài toán chuyển đổi câu hỏi ngôn ngữ tự nhiên thành câu lệnh truy vấn cơ sở dữ liệu có cấu trúc (**Text-to-SQL**) trên các bộ dữ liệu quy mô lớn và phức tạp hàng đầu hiện nay (**BIRD Benchmark** và **Spider Benchmark**).

Hệ thống SOTA hiện nay là **MARS-SQL** (Yang et al., ICML 2026) sử dụng cấu trúc đa tác tử gồm 3 mô hình lớn 7B giống hệt nhau ($3 \times 7\text{B} = 21\text{B}$ tham số), tiêu tốn từ 41 đến 47 GB VRAM và mất trung bình 22.12 giây cho mỗi câu hỏi. Dự án này đặt mục tiêu thay thế toàn bộ hệ thống 21B cồng kềnh bằng một **hệ thống đa tác tử ngôn ngữ nhỏ chuyên biệt (SLM Multi-Agent System)** có tổng kích thước chỉ từ **2.5B đến 4.0B tham số** (hoặc mở rộng lên 8.0B với mô hình 7B đơn lẻ).

Hệ thống mới cắt giảm hơn **80% tham số và chi phí tính toán**, có thể chạy hoàn toàn trên các môi trường điện toán đám mây như **Google Colab** (với 1 GPU đơn lẻ 16GB–24GB), đồng thời giải quyết triệt để 4 điểm nghẽn cố hữu của kiến trúc 21B cũ.

---

## 2. Bối Cảnh Nghiên Cứu & Các Điểm Nghẽn Của MARS-SQL 21B

Trong các cơ sở dữ liệu thực tế (như bộ dữ liệu BIRD gồm 95 cơ sở dữ liệu trải dài trên 37 lĩnh vực chuyên sâu), các câu hỏi thường đòi hỏi phải kết nối qua nhiều bảng và hiểu biết sâu sắc về cấu trúc lược đồ (schema). Hệ thống MARS-SQL 21B đã bộc lộ 4 điểm nghẽn nghiêm trọng:

### 2.1. Quá nặng, quá chậm và lãng phí tài nguyên (Compute & Latency Bottleneck)
- **Hiện trạng**: Phải nạp đồng thời 3 mô hình `Qwen2.5-Coder-7B` (mỗi mô hình chiếm 15.22 GB ở định dạng BF16), ngốn tới 41–47 GB VRAM.
- **Hệ quả**: Chi phí vận hành đắt đỏ, tiêu tốn trung bình 13,325 token và mất 22.12 giây để trả lời một câu hỏi, không khả thi khi huấn luyện thử nghiệm trên phần cứng nghiên cứu thông thường.

### 2.2. Hiện tượng "Mù cấu trúc" ở tác tử chọn bảng (Structural Blindness at Agent 1)
- **Hiện trạng**: Tác tử Grounding 7B của MARS-SQL duyệt từng bảng một cách tuần tự và độc lập xem bảng đó có chứa từ vựng giống câu hỏi hay không.
- **Hệ quả**: Những **bảng cầu nối (Bridge Tables)** — các bảng trung gian không hề chứa từ khóa người dùng nhắc tới nhưng bắt buộc phải có để thực hiện liên kết khóa ngoại (`FOREIGN KEY JOIN`) giữa hai bảng chính — bị loại bỏ oan uổng. Điều này làm hệ thống trả lời sai hoàn toàn ở **48 câu hỏi** trên tập BIRD-dev. Trên tập dữ liệu Spider-test, việc tắt hẳn Agent 1 7B chỉ làm giảm 0.56% độ chính xác, chứng minh việc dùng LLM 7B để lọc bảng là sự lãng phí tài nguyên rất lớn.

### 2.3. Ảo giác tên bảng và lỗi cú pháp ở tác tử sinh câu lệnh (Hallucination & Syntax Failure at Agent 2)
- **Hiện trạng**: Tác tử Generation 7B viết SQL theo phương thức sinh văn bản tự do không có rào chắn văn phạm.
- **Hệ quả**: Mô hình thường xuyên viết sai chính tả tên bảng/cột (ví dụ viết nhầm `fprm` thay vì `frpm`), ép sai kiểu dữ liệu số/chuỗi, gây lãng phí nhiều vòng lặp chạy thử chỉ để sửa các lỗi chính tả cơ bản.

### 2.4. Phán quyết cảm tính chưa hiệu chuẩn ở tác tử kiểm định (Uncalibrated Validation at Agent 3)
- **Hiện trạng**: Tác tử Validation 7B đọc lại các phương án SQL và lấy xác suất sinh ra từ `"Yes"` làm điểm tin cậy để xếp hạng.
- **Hệ quả**: Xác suất này chưa được hiệu chuẩn (uncalibrated), mô hình có xu hướng tự tin thái quá vào các câu lệnh viết sai, không chỉ ra được lỗi vi phạm quy tắc logic nào và làm tiêu tốn thêm một mô hình 7B thứ ba.

---

## 3. Giải Pháp Kỹ Thuật Đề Xuất (Core Technical Solution)

Dự án áp dụng nguyên lý: **"Thay thế mỗi khâu bằng đúng công cụ chuyên biệt nhẹ nhất"**, kết hợp toán rời rạc trên đồ thị, mô hình ngôn ngữ nhỏ (0.5B–3B), cơ chế khóa văn phạm thời gian thực và mô hình quyết định 0-token:

```
                              CÂU HỎI & CƠ SỞ DỮ LIỆU
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 1: GROUNDING AGENT (Bộ Tìm Bảng & Cột Nhẹ)                            │
│ 1. Vector Search: LitE-SQL (Qwen3-Embedding-0.6B + HN-SupCon) lấy Top-25 cột.   │
│ 2. Graph Algorithm: Cây Steiner (Graph-Link) trên đồ thị PK-FK cứu 100%         │
│    bảng cầu nối (Bridge Tables).                                                │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │ Lược đồ rút gọn S'
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 2: GENERATION AGENT (Mô Hình Nhỏ Viết SQL Chuẩn)                       │
│ 1. Core Model: FINER-SQL-3B (hoặc Qwen2.5-Coder-7B) học từ Reasoning Bank.      │
│ 2. Alignment: Huấn luyện LLM-JEPA căn chỉnh tư duy Text-to-Code.                │
│ 3. Constrained Decoding: XGrammar PDA khóa cứng 100% cú pháp & tên bảng.        │
│ 4. Sampling: Sinh 8 đến 16 quỹ đạo ứng viên (Rollouts).                         │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │ 8-16 phương án SQL ứng viên
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 3: VALIDATION AGENT (Kiểm Định Nhanh 0-Token)                         │
│ 1. Rule-based Probing: PV-SQL chạy thử mẫu và kiểm tra 10 quy tắc AST tất định. │
│ 2. System-1 Classifier: Laya-421M (ModernBERT) chấm điểm song song (~1.2 ms).   │
│ 3. Robustness: Khả năng chủ động từ chối khi phương án toàn sai (Abstention).  │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
                             CÂU LỆNH SQL TỐI ƯU NHẤT
```

---

## 4. Mục Tiêu Nghiên Cứu & Chỉ Số Đóng Góp (Contributions)

Dự án hướng tới việc xuất bản bài báo khoa học chất lượng cao với các đóng góp cụ thể:
1. **Đóng góp kiến trúc**: Chứng minh rằng sự kết hợp giữa mô hình nhúng siêu nhẹ (0.6B) + Toán đồ thị Cây Steiner hoàn toàn vượt trội mô hình 7B trong nhiệm vụ Schema Linking và bảo toàn 100% đường nối khóa ngoại.
2. **Đóng góp về huấn luyện**: Áp dụng thành công kỹ thuật **Completion-Only Loss**, căn chỉnh tư duy **LLM-JEPA**, và thuật toán **Dense GRPO 4 đầu điểm** trên mô hình nhỏ 3B, vượt qua hiện tượng sụp đổ lợi thế (*advantage collapse*).
3. **Đóng góp về kiểm định**: Đưa chi phí kiểm định câu lệnh từ hàng nghìn token xuống mức **0-token** với mô hình phân loại song song dựa trên ModernBERT, đạt sai số hiệu chuẩn cực thấp ($ECE \le 0.081$).
4. **Hiệu quả toàn tuyến**: Đạt độ chính xác thực thi **$75.0\% - 76.8\%$ EX** trên BIRD-dev với mô hình 4B (Combo C) và vượt **$77.84\%$ EX** với mô hình 8B (Combo D), cắt giảm thời gian xử lý từ 22.12s xuống còn 4.2s–6.5s.

---

## 5. Phạm Vi & Các Sản Phẩm Bàn Giao (Scope & Deliverables)

Dự án được thực hiện nghiêm ngặt và bàn giao đầy đủ các cấu phần sau:
- **Bộ dữ liệu chuẩn hóa**: Mã nguồn và dữ liệu sạch của 4 gói `DATA-G1-EMB`, `DATA-G1-GRAPH`, `DATA-G2-SFT`, `DATA-G3-VAL` (kèm gói mẫu thử `data/mini/`).
- **Mã nguồn huấn luyện**: Toàn bộ script huấn luyện (`src/training/`) cho Google Colab, hỗ trợ checkpoint tự động sang Google Drive.
- **Mã nguồn tác tử**: Bộ mã nguồn suy luận độc lập (`src/agents/`) cho 3 tác tử.
- **Hệ thống đo kiểm (Evaluation Harness)**: Bộ công cụ sinh 8–16 quỹ đạo, tính toán điểm số EX, VES, Recall, ECE và xuất bảng số liệu khoa học.
- **Google Colab Notebooks**: Các file `.ipynb` cấu hình sẵn, có thể chạy 1-click trên môi trường đám mây.
- **Tài liệu học thuật & Báo cáo kết quả**: Bộ tài liệu kiến trúc, đặc tả và nhật ký thực nghiệm phục vụ viết bài báo.
