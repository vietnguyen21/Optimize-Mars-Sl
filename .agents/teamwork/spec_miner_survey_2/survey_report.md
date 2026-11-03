# BÁO CÁO ĐẶC TẢ CHI TIẾT ROADMAP VÀ TIÊU CHÍ TRIỂN KHAI 4 GIAI ĐOẠN (SPECIFICATION SURVEY REPORT)
**Dự Án**: Hệ Thống Đa Tác Tử Ngôn Ngữ Nhỏ (SLM Multi-Agent Text-to-SQL System) Thay Thế MARS-SQL 21B  
**Tác Tử Khảo Sát**: `spec_miner_survey_2` (`teamwork_spec_miner`)  
**Ngày Thực Hiện**: 2026-11-02  
**Nguồn Tham Chiếu Tối Cao**:
- `docs/specifications/01_project_specification.md` (Đặc tả dự án toàn diện)
- `docs/specifications/02_architecture_design.md` (Thiết kế kiến trúc hệ thống)
- `docs/specifications/03_project_requirements.md` (Yêu cầu kỹ thuật & Tiêu chí nghiệm thu)
- `docs/specifications/04_roadmap_and_phases.md` (Lộ trình 4 Phase chi tiết)
- `docs/specifications/05_environment_and_colab_guide.md` (Hướng dẫn môi trường & Google Colab)
- `docs/specifications/requirements.md` & `requirements.txt` (Danh mục thư viện & phụ thuộc)
- `AGENTS_RESEARCH.md` & `GLOSSARY.md` (Hệ điều hành nghiên cứu & Thuật ngữ chuẩn)
- 4 Tài liệu Kỹ thuật gốc: `Ke Hoach Xu Ly Du Lieu.pdf`, `Ke Hoach Huan Luyen Mo Hinh.pdf`, `Ke Hoach Thi Nghiem Va Do Luong.pdf`, `Phuong Phap Luan Kien Truc Candidate.pdf`

---

## 1. Tổng Quan Kiến Trúc & Sứ Mệnh Nghiên Cứu

Hệ thống SOTA hiện nay là **MARS-SQL** (Yang et al., ICML 2026) sử dụng cấu trúc gồm 3 mô hình lớn 7B giống hệt nhau ($3 \times 7\text{B} = 21\text{B}$ tham số), tiêu tốn từ 41 đến 47 GB VRAM, 13,325 token và mất 22.12 giây cho mỗi câu hỏi. Hệ thống cũ bộc lộ 4 điểm nghẽn nghiêm trọng:
1. **Compute & Latency Bottleneck**: Đòi hỏi phần cứng khổng lồ, không thể chạy trên môi trường nghiên cứu phổ thông.
2. **Structural Blindness ở Agent 1**: Lọc bảng tuần tự, làm mất **Bảng cầu nối (Bridge Tables)** ở 48 câu hỏi BIRD-dev.
3. **Hallucination & Syntax Failure ở Agent 2**: Sinh SQL tự do không có rào chắn cú pháp, bịa tên bảng (`fprm` thay vì `frpm`), ép sai kiểu dữ liệu.
4. **Uncalibrated Validation ở Agent 3**: Chấm điểm cảm tính bằng xác suất sinh chữ "Yes" chưa qua hiệu chuẩn ($ECE \approx 0.246$).

**Mục tiêu của dự án**: Thay thế toàn bộ hệ thống 21B bằng kiến trúc **SLM Multi-Agent Chuyên Biệt Hóa Dị Thể (2.5B – 4.0B tham số, mở rộng lên 8.0B)**, cắt giảm >80% tham số và chi phí tính toán, tương thích 1 GPU thông thường trên Google Colab (T4 / L4 / A100), giải quyết triệt để 4 điểm nghẽn và đạt hiệu năng thực nghiệm ngang bằng hoặc vượt trội MARS-SQL 21B.

Dự án được triển khai tuần tự khép kín qua **4 Phase chức năng**:
- **Phase 1**: Chuẩn hóa & Đóng gói dữ liệu (Data Pipeline & Schema Grounding) + tạo bộ dữ liệu mẫu `data/mini/`.
- **Phase 2**: Phát triển mã nguồn huấn luyện (SLM Training Scripts trong `src/training/`).
- **Phase 3**: Thực thi huấn luyện và đóng gói trọng số trên Google Colab (`notebooks/`).
- **Phase 4**: Đo đạc thực nghiệm, kiểm thử toàn tuyến & đối sánh khoa học (`src/evaluation/`).

---

## 2. Danh Sách Tính Năng Khảo Sát (Features Discovered)

| # | Phân Loại (Category) | Tính Năng (Feature) | Mô Tả Kỹ Thuật (Description) | Đầu Vào (Inputs) | Đầu Ra (Outputs) | Cơ Chế Xử Lý Lỗi (Error Behavior) | Phương Thức Khám Phá |
|---|---|---|---|---|---|---|---|
| F01 | Phase 1 (Data) | BIRD Label Cleansing | Lọc bỏ các mẫu nhãn sai, mâu thuẫn ngữ nghĩa hoặc trả về rỗng bất thường bằng dual-LLM cross-check (Gemini-2.5-pro + GPT-4o) | 9,428 mẫu BIRD thô (33.4 GB) | 8,036 mẫu huấn luyện BIRD sạch chuẩn hóa | Loại bỏ hoàn toàn mẫu lỗi khỏi tập train, ghi log ID bị loại | Tài liệu 01 (PDF), Doc 03, Doc 04 |
| F02 | Phase 1 (Data) | Column Card Extraction & HN-SupCon Packaging | Tách 90,102 thẻ thông tin cột (6 trường) từ 8,036 mẫu BIRD; gán nhãn dương ($y=1$) và mẫu âm khó ($y=0$, margin = 0.1) | 8,036 mẫu BIRD sạch + DDL + từ điển dữ liệu | `DATA-G1-EMB.jsonl` (90,102 thẻ cột) | Bỏ qua cột thiếu thông tin DDL tối thiểu, cảnh báo thiếu description | Tài liệu 01 (PDF), Doc 02, Doc 04 |
| F03 | Phase 1 (Data) | Static PK-FK Graph Construction | Parser quét câu lệnh `CREATE TABLE` DDL, trích xuất nút Bảng, nút Cột, cạnh nội bộ và cạnh khóa ngoại PK-FK | 95 DB DDL (BIRD) + 138 DB DDL (Spider) | `DATA-G1-GRAPH.pkl` (NetworkX graph dictionary) | Bắt lỗi cú pháp DDL lạ, bỏ qua quan hệ PK-FK không phân giải được | Tài liệu 01 (PDF), Doc 02, Doc 04 |
| F04 | Phase 1 (Data) | SynSQL 4-Filter Cleansing | 4 bộ lọc tự động: bỏ câu thiếu `SELECT`, lặp lại code trong CoT, chứa chú thích `--`, và đề bài > 7,000 token | 2.5 triệu mẫu SynSQL thô | `SynSQL-Think-916K` và `SynSQL-Merge-Think-310K` | Loại bỏ ngay mẫu vi phạm bất kỳ 1 trong 4 điều kiện | Tài liệu 01 (PDF), Doc 03, Doc 04 |
| F05 | Phase 1 (Data) | Reasoning Bank Construction | Trích xuất luồng suy luận từng bước chất lượng cao từ DeepSeek-R1 trên 8,036 câu BIRD có SQL chạy đúng 100% | 8,036 câu BIRD + DeepSeek-R1 CoT traces | Ngân hàng lời giải mẫu (Reasoning Bank) bọc trong `<think>...</think>` | Chỉ lưu mẫu có SQL khớp kết quả thực thi 100% với Gold SQL | Tài liệu 01 (PDF), Doc 04 |
| F06 | Phase 1 (Data) | Mini-dataset Creation (`data/mini/`) | Script sinh gói dữ liệu nhỏ gọn gồm 100 câu BIRD (kèm 2 DB SQLite thật) và 1,000 câu SynSQL-Think cho smoke test < 2 phút | `data/processed/` hoặc data nguồn | `data/mini/bird_mini.json`, `data/mini/synsql_mini.json`, SQLite files | Tự động tạo thư mục và xác thực tính toàn vẹn của 2 DB SQLite | Tài liệu 01 (PDF), Doc 03, Doc 04 |
| F07 | Phase 1 (Grounding) | LitE-SQL Column Embedding | Mô hình `Qwen3-Embedding-0.6B` huấn luyện tương phản có giám sát HN-SupCon; hỗ trợ trích xuất offline index | Thẻ cột 6 trường + câu hỏi tự nhiên | Vector biểu diễn cột và `offline_column_index.npy` | Chuẩn hóa L2 vector, cosine similarity truy xuất Top-25 cột | Tài liệu 02 (PDF), Doc 02, Doc 04 |
| F08 | Phase 1 (Grounding) | Node-Weighted Steiner Tree (Graph-Link) | Thuật toán toán học trên đồ thị PK-FK tìm cây con trọng số nhỏ nhất nối các cột Top-25, cứu 100% bảng cầu nối | Top-25 cột từ LitE-SQL + đồ thị $G=(V, E)$ | Lược đồ rút gọn $S'$ bảo toàn đường nối JOIN | Fallback sang hợp nhất trực tiếp (union) các bảng nếu đồ thị phân rã | Tài liệu 02, 03 (PDF), Doc 02, Doc 04 |
| F09 | Phase 2 (Training) | Completion-Only Cross Entropy Loss | Gán nhãn `-100` cho toàn bộ token đề bài, câu hỏi và DDL trong Cross-Entropy; chỉ tính loss trên `<think>` và `<answer>` | Tokenized input sequence + prompt length offset | Scalar Cross-Entropy Loss tính riêng trên phần output | Nếu không tìm thấy tag `<think>` / `<answer>`, cảnh báo và mask theo delimiter | Tài liệu 02 (PDF), Doc 02, Doc 04 |
| F10 | Phase 2 (Training) | LLM-JEPA Joint-Embedding Alignment | Dự đoán vector ẩn khối Code từ token `[PRED]` cuối khối Text; sử dụng 2-block causal mask và hàm khoảng cách cosine | Cặp Text ($Q, E, S', \text{[PRED]}$) và Code (Gold SQL) | $\mathcal{L}_{\text{LLM-JEPA}} = \mathcal{L}_{\text{LLM}} + \lambda \cdot d_{\cos}(\text{Pred}, \text{Enc})$ | Ngăn Text nhìn trộm Code bằng mặt nạ $-\infty$; bật Random Loss Dropout ($LD=0.5$) | Tài liệu 02, 03 (PDF), Doc 02, Doc 04 |
| F11 | Phase 2 (Training) | Dense GRPO 4-Reward Training | Tối ưu hóa chính sách nhóm không cần Critic với 4 đầu thưởng chi tiết ($R_{\text{format}}, R_{\text{exec}}, R_{\text{mem}}, R_{\text{atomic}}$) chống sụp đổ lợi thế | Nhóm $G$ câu SQL ứng viên + DB SQLite + Reasoning Bank | Điểm thưởng $R(s \vert q)$ và độ vượt trội chuẩn hóa $A_j$ | Khi tất cả câu SQL chạy lỗi, $R_{\text{mem}}$ và $R_{\text{atomic}}$ vẫn tạo độ lệch điểm $A_j \neq 0$ | Tài liệu 02 (PDF), Doc 02, Doc 04 |
| F12 | Phase 2 (Training) | System-1 Domain SFT (NanoJev-0.6B) | Tinh chỉnh có giám sát trên 16 quỹ đạo SQL với 2 learning rate tách biệt: $10^{-5}$ cho backbone và $10^{-4}$ cho choice head | 128,576 quỹ đạo từ `DATA-G3-VAL` | Trọng số NanoJev-0.6B chuyên dụng cho SQL selection | Phạt gradient nếu backbone bị trôi dạt quá mức (weight decay) | Tài liệu 02, 04 (PDF), Doc 03, Doc 04 |
| F13 | Phase 2 (Training) | RLCD Calibrated Training (Laya-421M) | Huấn luyện ModernBERT-large (~85 MB) chèn token `[MASK]` chấm điểm song song, phạt nặng trường hợp tự tin ảo | Quỹ đạo SQL + kết quả thực thi nhị phân | Mô hình Laya-421M với $ECE \le 0.081$ | Phạt loss hiệu chuẩn khi xác suất dự đoán cao mà kết quả thực thi sai | Tài liệu 02, 03 (PDF), Doc 02, Doc 04 |
| F14 | Phase 2 (Training) | DriveCheckpointCallback | Callback tích hợp PyTorch/Transformers tự động sao lưu checkpoint từ ổ cứng Colab sang Google Drive an toàn | Trainer state, checkpoint folder cục bộ | Bản sao lưu vĩnh viễn trên `MyDrive/Text2SQL_Colab/` | Thử lại copy nếu lỗi I/O; không chặn đứng luồng train chính nếu Drive đầy | Doc 04, Doc 05 |
| F15 | Phase 3 (Colab) | Multi-Tier Notebook Suite | 5 Jupyter notebooks cấu hình sẵn chạy 1-click cho việc chuẩn bị dữ liệu, train nhúng, train generator 3B/7B, train validator | Mã nguồn `src/` + dữ liệu | Checkpoint mô hình lưu trên Drive | Kiểm tra tính tương thích GPU và tự động bật FP16/BF16 tương ứng | Doc 04, Doc 05 |
| F16 | Phase 3 (Colab) | Checkpoint Session Recovery | Khôi phục phiên làm việc bị ngắt kết nối trên Colab bằng cách tự động nạp checkpoint mới nhất từ Google Drive | Đường dẫn Drive checkpoint + cờ `--resume_from_checkpoint` | Mô hình tiếp tục huấn luyện từ step đã lưu | Nếu không tìm thấy checkpoint trên Drive, báo lỗi rõ ràng và yêu cầu xác nhận | Doc 05 |
| F17 | Phase 4 (Evaluation) | Trajectory Sampling Harness | Sinh đồng thời $N=8$ hoặc $N=16$ câu SQL ứng viên với tham số lấy mẫu đa dạng ($T=0.7, p=0.9, k=50$) phục vụ Ablation | Câu hỏi $Q$ + Lược đồ $S'$ | Tập ứng viên $\{SQL_1, \dots, SQL_K\}$ kèm log suy luận | Tự động loại bỏ chuỗi rỗng; nếu sinh thất bại, kích hoạt cơ chế retry | Doc 02, Doc 03, Doc 04 |
| F18 | Phase 4 (Evaluation) | XGrammar PDA Constrained Decoding | Khóa cứng 100% cú pháp SQL và danh sách bảng/cột từ $S'$ tại thời điểm sinh token bằng ôtômat đẩy xuống; biên dịch 4–8 ms | Token vocabulary + BNF/EBNF schema grammar | Token logit mask thời gian thực | Fallback tức thì sang Native Decoding kèm warning log khi thiếu thư viện | Tài liệu 03, 04 (PDF), Doc 02, Doc 04 |
| F19 | Phase 4 (Evaluation) | PV-SQL Database Probing & 10 AST Rules | Chạy tối đa 5 truy vấn ngắn có `LIMIT` kiểm tra định dạng dữ liệu thực tế và đối chiếu cây cú pháp AST với 10 quy tắc nghiệp vụ | Ứng viên SQL + SQLite DB + Câu hỏi $Q$ | Biên bản kiểm tra AST (Passed/Failed) + Kết quả probing | Nếu kết nối SQLite bị timeout/khóa, hủy lệnh probing sau 2 giây | Tài liệu 01, 03 (PDF), Doc 02, Doc 04 |
| F20 | Phase 4 (Evaluation) | 0-Token Parallel Scoring (Laya-421M) | Chấm điểm song song 8–16 ứng viên SQL bằng đầu phân loại Cross-Encoder trong ~1.2 ms mà không sinh thêm token nào | Cặp (Context, Candidate SQL) | Điểm số xác suất chuẩn hóa của từng ứng viên | Nếu tất cả ứng viên < 0.5, kích hoạt cơ chế Abstention hoặc Majority Vote | Tài liệu 03 (PDF), Doc 02, Doc 04 |
| F21 | Phase 4 (Evaluation) | Adaptive Early-Exit Gate | Cho phép chốt đáp án ngay ở phương án đầu tiên nếu thỏa mãn 10 quy tắc PV-SQL và điểm tự tin Laya-421M $\ge 0.85$ | Phương án $SQL_1$ + điểm kiểm định | Quyết định dừng sớm hoặc tiếp tục sinh đủ 8–16 phương án | Nếu không đạt ngưỡng, tiếp tục chạy đủ chu trình đánh giá toàn bộ ứng viên | Tài liệu 03 (PDF), Doc 02, Doc 04 |
| F22 | Phase 4 (Evaluation) | Security Prompt Injection Isolation | Tách biệt tuyệt đối dữ liệu người dùng và lệnh điều khiển; cấm nối chuỗi bảng thô vào câu lệnh System-1 | Kết quả truy vấn bảng thô | Trạng thái an toàn chỉ gồm tên bảng/cột tin cậy và báo cáo AST | Cắt ngắn hoặc lọc bỏ các chuỗi văn bản lạ nghi ngờ là prompt injection | Tài liệu 01 (PDF), Doc 02, Doc 03 |
| F23 | Phase 4 (Evaluation) | Comprehensive Benchmark Metric Suite | Bộ công cụ tính toán tự động các chỉ số học thuật: Execution Accuracy (EX), Valid Efficiency Score (VES), ECE, Latency, Peak VRAM | File log kết quả thực thi `evaluation_results.jsonl` | Báo cáo Markdown tổng hợp và bảng số liệu khoa học | Xử lý an toàn các giá trị NaN/Inf, đo thời gian thực thi trung bình 3 lượt | Doc 03, Doc 04 |

---

## 3. Các Tình Huống Biên & Ngoại Lệ (Edge Cases)

| # | Tính Năng (Feature) | Tình Huống Biên / Đầu Vào Đặc Biệt (Input / Condition) | Hành Vi Hệ Thống Được Đặc Tả (Observed / Expected Behavior) |
|---|---|---|---|
| E01 | PK-FK Graph Construction | Cơ sở dữ liệu không khai báo rõ ràng ràng buộc `FOREIGN KEY` trong câu lệnh `CREATE TABLE` (DDL ngầm định). | Parser ghi nhận cảnh báo; kích hoạt thuật toán suy luận liên kết dựa trên tên cột tương đồng (ví dụ: `schools.id` và `frpm.school_id`) để bổ sung cạnh dự phòng với trọng số penalty. |
| E02 | Node-Weighted Steiner Tree | Các cột được chọn từ LitE-SQL thuộc về các bảng nằm trong 2 thành phần liên thông rời rạc (đồ thị bị đứt gãy, không có đường đi PK-FK). | Thuật toán Steiner Tree phát hiện không có đường đi; tự động suy biến thành thao tác lấy hợp nhất (union) các bảng thành phần, kèm cờ đánh dấu cần thêm liên kết `CROSS JOIN` hoặc kiểm tra lại schema. |
| E03 | BIRD Label Cleansing | Cả Gemini-2.5-pro và GPT-4o đều trả về kết quả mâu thuẫn nhau khi đánh giá câu Gold SQL của BIRD (1 mô hình bảo đúng, 1 mô hình bảo sai). | Mẫu được đưa vào danh sách nghi vấn trung gian (`quarantine`); nếu câu Gold SQL thực thi trên SQLite trả về kết quả rỗng (empty result set), mẫu bị loại bỏ ngay lập tức để bảo vệ tập sạch 8,036 mẫu. |
| E04 | Completion-Only Loss | Đầu vào không tìm thấy cặp thẻ `<think>` hoặc `<answer>` (do dữ liệu lỗi định dạng hoặc cắt cụt chuỗi). | Hàm tính loss bắt ngoại lệ; tự động fallback sang tính loss từ token thứ `prompt_len` trở đi dựa trên vị trí phân tách câu hỏi, đồng thời ghi log cảnh báo dữ liệu không chuẩn. |
| E05 | Dense GRPO Reward | Cả $G$ phương án sinh ra của mô hình 3B đều bị lỗi cú pháp hoặc lỗi cơ sở dữ liệu trên câu hỏi cực khó ($R_{\text{exec}} = 0$). | Thay vì sụp đổ lợi thế ($A_j = 0$), hàm thưởng kích hoạt $R_{\text{mem}}$ (đo cosine similarity giữa `<think>` và Reasoning Bank) và $R_{\text{atomic}}$ (đo tỷ lệ đúng của các mệnh đề `SELECT`, `FROM`), tạo ra độ chênh lệch điểm để gradient không biến mất. |
| E06 | XGrammar Compilation | Cấu trúc lược đồ rút gọn $S'$ có tên bảng hoặc tên cột chứa ký tự đặc biệt, dấu cách, hoặc từ khóa trùng với SQL reserved words (ví dụ: `Group`, `Order`, `Table-1`). | XGrammar PDA tự động bọc tên bảng/cột vào dấu ngoặc kép `"` hoặc dấu backtick ``` ` ``` trong tập luật ngữ pháp BNF/EBNF, ngăn chặn lỗi biên dịch văn phạm. |
| E07 | XGrammar Fallback | Môi trường máy chủ không cài đặt được thư viện C++ `xgrammar`, chạy trên CPU không hỗ trợ GPU bitmask cache, hoặc lỗi biên dịch ngữ pháp lúc runtime. | Hệ thống bắt lỗi ngoại lệ, ghi log cảnh báo `[XGrammar Warning] Falling back to Native Decoding`, và chuyển đổi trong suốt sang sinh văn bản tự do (Native Decoding) mà không làm sập pipeline đánh giá. |
| E08 | Hollow Rescue | Câu lệnh SQL được sinh ra bởi SLM đúng 100% về mặt cú pháp và bảng/cột nhờ XGrammar, nhưng mô hình hiểu sai logic (câu hỏi yêu cầu tính `AVG` nhưng mô hình viết `SUM`). | Bộ lọc 10 quy tắc AST của PV-SQL phát hiện câu hỏi chứa từ khóa `average` nhưng AST thiếu hàm `AVG()`; bác bỏ câu lệnh ngay lập tức và chuyển quyền phán quyết cho ứng viên khác. |
| E09 | 0-Token Abstention | Cả 16 phương án sinh ra trong tập ứng viên đều vi phạm quy tắc AST hoặc nhận điểm tự tin từ Laya-421M thấp hơn ngưỡng chấp nhận ($< 0.5$). | Bộ phán quyết System-1 kích hoạt kỹ năng Từ chối đoán mò (Abstention); trả về cờ `ABSTAIN` yêu cầu sinh lại với tham số nhiệt độ mới hoặc chuyển sang chế độ giải cứu tập thể thay vì đoán mò bừa bãi. |
| E10 | Google Colab Disconnection | Phiên làm việc Google Colab bị mất kết nối đột ngột giữa chừng khi đang chạy epoch thứ 2 của mô hình 7B. | Trọng số đã được lưu tự động từng checkpoint qua `DriveCheckpointCallback` trên Google Drive. Khi kết nối lại, script đọc `--resume_from_checkpoint latest` và tiếp tục chạy từ step gần nhất mà không mất mát dữ liệu. |
| E11 | Prompt Injection qua Dữ Liệu | Cơ sở dữ liệu của người dùng chứa một ô văn bản lưu chuỗi độc hại: `'; DROP TABLE users; --` hoặc chỉ thị điều khiển mô hình. | Hệ thống áp dụng quy tắc lọc bảo mật nghiêm ngặt (Wang et al., 2026): Kết quả dữ liệu thô của truy vấn bị cấm tuyệt đối không được nối vào prompt của System-1; chỉ truyền metadata cấu trúc và mã lỗi. |
| E12 | SQLite Execution Timeout | Câu lệnh SQL sinh ra chứa tích Đề-các (Cartesian product không có điều kiện `ON`/`WHERE`) gây treo máy hoặc chạy vô tận trên cơ sở dữ liệu lớn. | Bộ thực thi SQLite đặt ngưỡng ngắt thời gian tối đa (`timeout = 10.0s`); nếu vượt quá thời gian này, câu lệnh bị gắn nhãn lỗi `Execution Timeout`, nhận $R_{\text{exec}} = 0$ và bị loại khỏi nhóm ứng viên tốt. |

---

## 4. Chi Tiết Phase 1: Data Pipeline & Schema Grounding

### 4.1. Các Nguồn Dữ Liệu & Quy Chuẩn Tiền Xử Lý
1. **BIRD (v1.0)**:
   - Quy mô: 12,751 cặp (Question, SQL) trên 95 cơ sở dữ liệu SQLite, 37 lĩnh vực chuyên sâu, dung lượng 33.4 GB.
   - Làm sạch: Rà soát bằng dual-LLM (Gemini-2.5-pro + GPT-4o) theo quy trình MARS-SQL. Loại bỏ mẫu mâu thuẫn ngữ nghĩa và mẫu có kết quả thực thi rỗng bất thường, thu được **8,036 mẫu sạch chuẩn hóa**.
2. **Spider 1.0**:
   - Quy mô: Tập Dev & Test gồm 2,147 mẫu trên 138 lĩnh vực cơ sở dữ liệu.
   - Đặc điểm: Không trùng lặp database giữa tập học và tập thi; phục vụ kiểm thử zero-shot cross-domain.
3. **Spider-DK (v1.0)**:
   - Kiểm tra khả năng suy luận ngoài miền (Out-of-Domain - OOD) với các câu hỏi đòi hỏi kiến thức ngầm định.
4. **SynSQL-2.5M**:
   - Kho dữ liệu suy luận tổng hợp từ OmniSQL. Áp dụng **4 bộ lọc nghiêm ngặt**:
     1. *Thiếu từ khóa truy vấn*: SQL không chứa mệnh đề `SELECT` hợp lệ.
     2. *Lặp lại code*: Đoạn suy luận từng bước (Chain-of-Thought) lặp lại nguyên văn câu lệnh SQL nhiều lần.
     3. *Chứa chú thích*: Câu lệnh SQL chứa ký tự comment `--` (gây gãy bộ phân tích cú pháp).
     4. *Đề bài quá dài*: Tổng độ dài prompt (câu hỏi + mô tả schema) vượt quá 7,000 token.
   - Đầu ra sau lọc:
     - `SynSQL-Think-916K` (916,000 mẫu): Bọc gọn CoT trong `<think>...</think>` và SQL trong `<answer>...</answer>`.
     - `SynSQL-Merge-Think-310K` (310,000 mẫu): Dạy mô hình gộp sửa lỗi CSC-SQL 0.5B.
5. **Reasoning Bank**:
   - 8,036 lời giải mẫu có CoT chuẩn xác từ DeepSeek-R1 (chỉ giữ lại những lời giải thực thi khớp 100% với Gold SQL) trên tập BIRD sạch.

### 4.2. Mô Hình Dữ Liệu & Định Dạng Lưu Trữ
- **`DATA-G1-EMB.jsonl`**:
  - Gồm 90,102 thẻ thông tin cột cấp độ bảng/cột từ 8,036 câu BIRD.
  - Cấu trúc 6 trường thông tin bắt buộc:
    ```json
    {
      "table_name": "frpm",
      "column_name": "CDSCode",
      "data_type": "TEXT",
      "constraints": "PRIMARY KEY",
      "column_description": "California Department of Education unique identifier code for schools",
      "sample_values": ["01611190130229", "01611270130450", "01611430136127"]
    }
    ```
  - Cặp dữ liệu huấn luyện: `(Question Q, Column Card)` kèm nhãn nhị phân:
    - $y = 1$ (Nhãn dương): Cột thực sự có mặt trong câu lệnh Gold SQL.
    - $y = 0$ (Mẫu âm khó - Hard Negatives): Cột trong cùng database không nằm trong Gold SQL nhưng có độ tương đồng ngữ nghĩa cao với câu hỏi hoặc trùng tên với cột đúng (áp dụng biên phân cách $\text{margin} = 0.1$).
- **`DATA-G1-GRAPH.pkl`**:
  - Lưu trữ 95 đồ thị BIRD và 138 đồ thị Spider dưới dạng từ điển các đối tượng `networkx.Graph`.
  - Cấu trúc đồ thị $G = (V, E)$:
    - Tập đỉnh $V = V_{\text{table}} \cup V_{\text{column}}$.
    - Tập cạnh $E = E_{\text{intra}} \cup E_{\text{pk-fk}}$, trong đó $E_{\text{intra}}$ nối Bảng với các Cột của chính bảng đó; $E_{\text{pk-fk}}$ nối Cột khóa chính với Cột khóa ngoại giữa hai bảng khác nhau.
- **`DATA-G2-SFT.jsonl`**:
  - Chứa 916,000 mẫu SynSQL + 8,036 mẫu Reasoning Bank.
  - Định dạng chuẩn hóa:
    ```text
    <prompt>
    [Instruction & Question Q & Sub-Schema S']
    </prompt>
    <think>
    Phân tích yêu cầu bài toán, các bảng cần nối và điều kiện gom nhóm...
    </think>
    <answer>
    SELECT T1.name, AVG(T2.score) FROM schools AS T1 JOIN frpm AS T2 ON T1.CDSCode = T2.CDSCode GROUP BY T1.name
    </answer>
    ```
- **`DATA-G3-VAL.jsonl`**:
  - Chứa 128,576 quỹ đạo ứng viên ($8,036 \times 16$).
  - Gắn nhãn nhị phân $\{0, 1\}$ dựa trên kết quả chạy thử trên SQLite, kèm biên bản kiểm tra 10 quy tắc AST của PV-SQL.

### 4.3. Thuật Toán Cốt Lõi Của Grounding Agent
1. **Bộ Nhúng Cột LitE-SQL (`Qwen3-Embedding-0.6B`)**:
   - Huấn luyện bằng hàm mất mát tương phản có giám sát lọc mẫu âm khó (HN-SupCon):
     $$\mathcal{L}_{\text{HN-SupCon}} = - \log \frac{\exp(\text{sim}(q, c^+) / \tau)}{\exp(\text{sim}(q, c^+) / \tau) + \sum_{c^- \in \mathcal{H}} \exp(\text{sim}(q, c^-) / \tau)}$$
     với $\tau$ là tham số nhiệt độ và $\mathcal{H}$ là tập các cột âm khó trong cùng database với khoảng cách biên $\text{margin} = 0.1$.
   - **Offline Indexing**: Mã hóa toàn bộ thẻ cột của 95 DB thành ma trận vector tĩnh lưu vào `offline_column_index.npy`. Khi có câu hỏi mới, chỉ tốn vài mili-giây để tính cosine similarity và trích xuất Top-25 cột có điểm cao nhất (đạt $\text{TPR} \ge 95.23\%$ trên BIRD-dev).
2. **Thuật Toán Cây Steiner Tĩnh (Node-Weighted Steiner Tree - NWST / Graph-Link)**:
   - Các bảng chứa 25 cột được chọn từ LitE-SQL đóng vai trò là các đỉnh bắt buộc phải đến (Terminal Nodes $T \subset V$).
   - Thuật toán giải bài toán cây Steiner trên đồ thị $G=(V, E)$ để tìm cây con $T_{\text{Steiner}}$ có tổng trọng số nút nhỏ nhất nối liền toàn bộ các đỉnh trong $T$.
   - **Cơ chế cứu bảng cầu nối (Bridge Tables Recovery)**: Bất kỳ bảng trung gian nào không chứa từ khóa trong câu hỏi nhưng nằm trên đường đi ngắn nhất giữa hai bảng chính sẽ tự động được đưa vào $T_{\text{Steiner}}$.
   - **Kết quả nghiệm thu**: Cứu thành công trọn vẹn **100% (48/48 câu hỏi)** bị mất bảng cầu nối của MARS-SQL 21B, giúp tăng độ chính xác thực thi trên nhóm câu hỏi khó thêm **+13.8% EX**.

### 4.4. Yêu Cầu Cho Bộ Mẫu Thử Nghiệm `data/mini/`
- **Mục tiêu**: Đảm bảo toàn bộ luồng kiểm thử (smoke test) chạy thành công từ đầu đến cuối trong vòng **< 2 phút** mà không cần GPU hay nạp dữ liệu khổng lồ 33.4 GB.
- **Thành phần cấu trúc thư mục**:
  ```text
  data/mini/
  ├── bird_mini.json          # 100 câu hỏi BIRD chuẩn kèm Gold SQL và DB ID
  ├── synsql_mini.jsonl       # 1,000 mẫu SynSQL-Think có thẻ <think> và <answer>
  ├── graphs_mini.pkl         # Đồ thị PK-FK trích xuất riêng cho 2 database kiểm thử
  ├── sqlite/
  │   ├── california_schools.sqlite  # Database thật 1 kèm toàn bộ bảng và dữ liệu mẫu
  │   └── financial.sqlite           # Database thật 2 kèm toàn bộ bảng và dữ liệu mẫu
  └── column_cards_mini.jsonl # Thẻ cột của 2 database trên phục vụ smoke test Grounding
  ```
- **Kịch bản kiểm tra**: Script `scripts/make_mini_dataset.py` trích xuất tự động và kiểm tra tính toàn vẹn của các file trên, đảm bảo thực thi câu lệnh SQL trả về kết quả hợp lệ.

---

## 5. Chi Tiết Phase 2: SLM Training Scripts (`src/training/`)

### 5.1. Kỹ Thuật Completion-Only Cross Entropy Loss
- **Mục tiêu**: Ngăn chặn mô hình ngôn ngữ nhỏ (0.5B – 3B) ghi nhớ vẹt cấu trúc bảng và tên cột của dữ liệu huấn luyện, ép mô hình tập trung năng lực vào việc suy luận logic và viết code SQL.
- **Cơ chế triển khai**:
  - Đầu vào chuỗi token $X = [x_1, x_2, \dots, x_N]$.
  - Điểm phân tách $L_{\text{prompt}}$ nằm ngay trước thẻ bắt đầu `<think>`.
  - Toàn bộ nhãn mục tiêu từ vị trí $1$ đến $L_{\text{prompt}}-1$ được gán giá trị đặc biệt `-100` (giá trị bỏ qua đạo hàm chuẩn trong PyTorch `CrossEntropyLoss`).
  - Hàm mất mát chỉ tính trên các token câu trả lời:
    $$\mathcal{L}_{\text{Completion-Only}} = - \frac{1}{\sum_{i=L_{\text{prompt}}}^N \mathbb{I}(y_i \neq -100)} \sum_{i=L_{\text{prompt}}}^N \log P(x_i \mid x_1, \dots, x_{i-1})$$
- **Hỗ trợ script**: `train_sft_generator.py` hỗ trợ cả 2 dòng mô hình `--model_type 3b` (`FINER-SQL-3B`) và `--model_type 7b` (`Qwen2.5-Coder-7B`).

### 5.2. Kiến Trúc & Hàm Mất Mát LLM-JEPA (Joint-Embedding Predictive Architecture)
- **Động lực**: Mô hình ngôn ngữ thông thường chỉ học dự đoán từ tiếp theo (next-token prediction) mà không thực sự liên kết không gian ngữ nghĩa giữa "câu hỏi tự nhiên" (Text) và "câu lệnh truy vấn cơ sở dữ liệu" (Code). LLM-JEPA (Huang, LeCun & Balestriero, ICLR 2026) ép hai khối này phải tương thích sâu sắc trong không gian biểu diễn ẩn.
- **Kiến trúc mạng & Mặt nạ chú ý 2 khối (2-Block Causal Mask)**:
  - Cửa sổ ngữ cảnh chứa 2 khối: Khối 1 (Text: $Q, E, S'$ nối thêm token đặc biệt $[PRED]$ ở cuối) và Khối 2 (Code: Câu SQL chuẩn).
  - Ma trận chú ý đặt giá trị $-\infty$ ở vùng giao thoa giữa Khối 1 và Khối 2: Khối Text không được phép nhìn sang Khối Code (đảm bảo tính nhân quả khi suy luận), trong khi Khối Code có thể nhìn Khối Text để học sinh từ.
- **Hàm mục tiêu kết hợp**:
  $$\mathcal{L}_{\text{LLM-JEPA}} = \mathcal{L}_{\text{LLM}}(\text{Text}_{1:\ell-1}, \text{Text}_\ell) + \lambda \cdot d_{\cos}(\text{Pred}(\text{Enc}(\text{Text})), \text{Enc}(\text{Code}))$$
  - Trong đó $d_{\cos}(u, v) = 1 - \frac{u \cdot v}{\|u\| \|v\|}$ là khoảng cách cosine giữa vector dự đoán tại token $[PRED]$ và vector ẩn trung bình của câu SQL chuẩn.
  - Tham số chuẩn: $\lambda = 1.0$, số token dự đoán $k = 1$, tốc độ học $lr = 2 \times 10^{-5}$.
- **Tối ưu hóa huấn luyện**:
  - **Random JEPA-Loss Dropout ($LD = 0.5$)**: Cứ 2 mini-batch thì ngẫu nhiên chỉ tính hàm mất mát JEPA ở 1 batch, cắt giảm 50% thời gian huấn luyện mà không làm suy giảm độ chính xác.
  - **0% Inference Overhead**: Khi suy luận, token $[PRED]$ và mạng dự đoán bị gỡ bỏ hoàn toàn; mô hình sinh SQL theo cách thông thường với biểu diễn bên trong đã được căn chỉnh tối ưu (+7.8% EX trên tập Spider).

### 5.3. Huấn Luyện Học Tăng Cường Dense GRPO (Group Relative Policy Optimization)
- **Vấn đề của GRPO truyền thống (Advantage Collapse)**:
  - MARS-SQL gốc dùng hàm thưởng nhị phân $\{0, 1\}$: câu lệnh chạy đúng được 1 điểm, chạy sai hoặc lỗi cú pháp được 0 điểm.
  - Với các mô hình nhỏ (0.5B – 3B) trên câu hỏi khó của BIRD, cả $G=8$ lần thử đều nhận 0 điểm ($r_1 = r_2 = \dots = r_G = 0$).
  - Độ vượt trội tương đối trong nhóm $A_j = \frac{r_j - \text{mean}(r)}{\text{std}(r)} = 0$, đạo hàm triệt tiêu hoàn toàn và mô hình ngừng học.
- **Giải pháp: Dense GRPO với 4 Đầu Thưởng Dày (FINER-SQL Hoang et al., 2026)**:
  $$R(s \mid q) = R_{\text{format}}(s) + R_{\text{exec}}(s \mid q) + R_{\text{mem}}(s) + R_{\text{atomic}}(s, s^*)$$
  1. **Thưởng Trình Bày ($R_{\text{format}} \in \{0, 1\}$)**: Thưởng $1.0$ điểm nếu cấu trúc đóng mở đầy đủ cặp thẻ `<think>...</think>` và `<answer>...</answer>`.
  2. **Thưởng Thực Thi ($R_{\text{exec}} \in \{0, 1, 2\}$)**:
     - $0$ điểm: Lỗi cú pháp SQL hoặc runtime database error.
     - $1.0$ điểm: Chạy thành công trên SQLite mà không gặp lỗi (kết quả có thể chưa khớp đáp án).
     - $2.0$ điểm: Kết quả thực thi trùng khớp $100\%$ với kết quả của Gold SQL.
  3. **Thưởng Tư Duy Hợp Lý / Ký Ức ($R_{\text{mem}} \in [0, 1]$)**: Khi câu SQL thực thi sai, trích xuất vector nhúng của đoạn `<think>` vừa sinh ra và tính độ tương đồng cosine với luồng suy luận mẫu của câu đó trong `Reasoning Bank`. Thưởng điểm cho việc mô hình có hướng tư duy đúng dù viết sai cú pháp nhỏ.
  4. **Thưởng Mảnh Ghép Toán Tử ($R_{\text{atomic}} \in [0, 1]$)**: Tách câu SQL sinh ra và Gold SQL thành các thành phần cấu trúc độc lập (`SELECT`, `FROM`, `JOIN`, `WHERE`, `GROUP BY`, `ORDER BY`), tính tỷ lệ phần trăm số mảnh ghép khớp đúng logic.
- **Hiệu năng**: Nhờ 4 đầu thưởng, luôn đảm bảo $A_j \neq 0$; FINER-SQL-3B đạt **67.73% EX** trên BIRD-dev, đánh bại mô hình 7B đơn lẻ của MARS-SQL (66.37%).
- **Các biến thể khác**:
  - Mô hình 0.5B – 1.5B: Huấn luyện đồng thời bộ sinh và bộ gộp sửa lỗi CSC-SQL.
  - Mô hình 7B (Combo D): Áp dụng hàm thưởng tối giản Arctic-Text2SQL-R1 (chỉ cần thưởng thực thi và cú pháp).

### 5.4. Huấn Luyện Mô Hình Phán Quyết System-1 (Agent 3)
1. **Domain SFT cho NanoJev-0.6B**:
   - Khắc phục hiện tượng sụt giảm độ chính xác từ 52% xuống 17% do mô hình gốc chỉ học chơi game.
   - Huấn luyện SFT trên 16 quỹ đạo SQL với **2 learning rates tách biệt**:
     - Khung xương ngôn ngữ (Backbone): $lr = 1 \times 10^{-5}$.
     - Đầu quyết định (Choice Head / Boolean Head): $lr = 1 \times 10^{-4}$.
   - Độ chính xác chọn câu lệnh đúng tăng vọt lên **93.8%** với độ trễ chỉ ~100 ms.
2. **Huấn Luyện RLCD cho Laya-421M (ModernBERT-large, ~85 MB)**:
   - Chèn token $[MASK]$ đại diện cho mỗi phương án SQL để chấm điểm song song toàn bộ 8–16 ứng viên trong **~1.2 ms** (tương đương 810 quyết định/giây trên GPU T4).
   - Thuật toán RLCD (Reinforcement Learning for Calibrated Decisions) phạt nặng trường hợp mô hình tự tin cao nhưng thực thi sai, đưa sai số hiệu chuẩn kỳ vọng xuống **$ECE \le 0.081$**, cho phép thiết lập Cổng dừng sớm an toàn tại ngưỡng tự tin **0.85**.
3. **Huấn Luyện Từ Chối Đoán Mò (Abstention RL) cho Kev-0.8B**:
   - Thưởng điểm khi mô hình chủ động chọn hành động "Từ chối / Yêu cầu viết lại" khi cả 16 phương án sinh ra đều sai, giảm tỷ lệ đoán mò từ 26% xuống 5%.

### 5.5. Cấu Hình Trainer, LoRA/QLoRA & Lưu Checkpoint
- **Cấu hình LoRA / QLoRA**:
  - Rank $r = 512$, $\alpha = 1024$, dropout = 0.05.
  - Áp dụng trên toàn bộ linear projection modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
  - Lượng tử hóa 4-bit (QLoRA) bằng `bitsandbytes` khi chạy trên GPU T4 16GB.
- **Bộ Hạt Giống Ngẫu Nhiên Cố Định (Reproducibility)**:
  - Toàn bộ script cố định tập 5 hạt giống: `seeds = [82, 23, 37, 84, 4]`.
- **Cơ Chế Đồng Bộ Google Drive (`DriveCheckpointCallback`)**:
  - Kế thừa `transformers.TrainerCallback`.
  - Tại mỗi sự kiện `on_save`, tự động sao lưu thư mục `checkpoint-<step>` từ thư mục tạm `/content/checkpoints/` sang thư mục vĩnh viễn `/content/drive/MyDrive/Text2SQL_Colab/checkpoints/` bằng `shutil.copytree(dirs_exist_ok=True)`.

---

## 6. Chi Tiết Phase 3: Notebooks Trong `notebooks/`

### 6.1. Danh Mục Các Notebook Google Colab
1. **`01_colab_data_prep.ipynb`**:
   - Kiểm tra GPU (`nvidia-smi`), mount Google Drive.
   - Cài đặt thư viện phụ thuộc từ `requirements.txt`.
   - Chạy pipeline tiền xử lý: Lọc BIRD (8,036 mẫu), lọc SynSQL (916k mẫu), dựng đồ thị PK-FK (`DATA-G1-GRAPH.pkl`), đóng gói thẻ cột (`DATA-G1-EMB.jsonl`), và tạo thư mục `data/mini/`.
2. **`02_colab_train_grounding.ipynb`**:
   - Huấn luyện `Qwen3-Embedding-0.6B` bằng HN-SupCon trên Colab T4/L4.
   - Xuất file chỉ mục vector tĩnh `offline_column_index.npy` và lưu vào Google Drive.
   - Kiểm tra đánh giá chỉ số Schema Linking Recall (Top-25 cột $\ge 95.23\%$).
3. **`03_colab_train_generator_3b.ipynb`**:
   - Huấn luyện mô hình sinh cốt lõi `FINER-SQL-3B` trên Colab T4 (QLoRA 4-bit) hoặc L4 (LoRA BF16).
   - Huấn luyện 3 bước: SFT Completion-Only $\rightarrow$ Căn chỉnh LLM-JEPA $\rightarrow$ Dense GRPO 4 đầu điểm.
   - Đồng bộ checkpoint an toàn vào Drive sau mỗi epoch hoặc 500 steps.
4. **`03b_colab_train_generator_7b.ipynb`**:
   - Dành riêng cho GPU Colab Pro+ (NVIDIA A100 40GB).
   - Huấn luyện `Qwen2.5-Coder-7B` nhằm xác lập kỷ lục vượt qua MARS-SQL 21B ($> 77.84\%$ EX trên BIRD-dev).
   - Tích hợp SFT Completion-Only, LLM-JEPA, và Arctic-style GRPO.
5. **`04_colab_train_validator.ipynb`**:
   - Huấn luyện mô hình phán quyết System-1: Domain SFT cho NanoJev-0.6B và RLCD cho Laya-421M.
   - Đo đạc sai số hiệu chuẩn $ECE \le 0.081$ và thử nghiệm Cổng dừng sớm 0.85.

### 6.2. Phân Cấp Phần Cứng & Yêu Cầu Mô Hình
- **Colab Free (Tesla T4 ~15GB VRAM)**:
  - Chạy Phase 1 Data Prep, smoke test trên `data/mini/`.
  - Huấn luyện `Qwen3-Embedding-0.6B`, `Laya-421M`.
  - Huấn luyện `FINER-SQL-3B` ở chế độ QLoRA 4-bit (chỉ tốn ~1.93 GB VRAM).
- **Colab Pro (NVIDIA L4 / V100 ~16–22.5GB VRAM)**:
  - Môi trường tiêu chuẩn khuyến nghị để chạy Combo C toàn tuyến.
  - Huấn luyện `FINER-SQL-3B` với LoRA rank 512 ở định dạng BF16/FP16 (chiếm ~6.18 GB VRAM).
- **Colab Pro+ (NVIDIA A100-SXM4 ~40GB VRAM)**:
  - Huấn luyện `Qwen2.5-Coder-7B` (Combo D) với LoRA/Full SFT, chiếm dưới 20.0 GB VRAM.

### 6.3. Cơ Chế Mount Drive & Phục Hồi Phiên Huấn Luyện (Session Recovery)
- **Cấu trúc thư mục chuẩn trên Google Drive**:
  ```text
  /content/drive/MyDrive/Text2SQL_Colab/
  ├── data/
  │   ├── processed/
  │   └── mini/
  ├── checkpoints/
  │   ├── lite_sql_embedding_0.6b/
  │   ├── finer_sql_3b/
  │   │   ├── checkpoint-500/
  │   │   ├── checkpoint-1000/
  │   │   └── best_model/
  │   ├── qwen2.5_coder_7b/
  │   └── laya_421m_rlcd/
  └── logs/
      └── train_history.json
  ```
- **Kịch bản phục hồi khi bị ngắt kết nối**:
  1. Mount lại Google Drive: `drive.mount('/content/drive')`.
  2. Truyền tham số `--resume_from_checkpoint latest`. Script tự động quét thư mục `checkpoints/` trên Drive, tìm checkpoint có số step lớn nhất, nạp lại trạng thái optimizer, scheduler, RNG seeds và tiếp tục huấn luyện mà không mất mát bước nào.

---

## 7. Chi Tiết Phase 4: Evaluation & Benchmark (`src/evaluation/`)

### 7.1. Lấy Mẫu 8–16 Quỹ Đạo Ứng Viên Phục Vụ Ablation Study
- **Phương thức lấy mẫu (Rollouts)**:
  - Đối với mỗi câu hỏi kiểm thử trong BIRD-dev hoặc Spider, Agent 2 sinh đồng thời $N = 8$ hoặc $N = 16$ câu lệnh SQL ứng viên.
  - Bộ tham số lấy mẫu đa dạng: Nhiệt độ $T = 0.7$, Top-$p = 0.9$, Top-$k = 50$.
- **Mục đích nghiên cứu sâu (Ablation Study)**:
  - So sánh đối đầu giữa các chiến lược chọn đáp án cuối cùng:
    1. *Majority Voting* (Bỏ phiếu theo kết quả chạy thử xuất hiện nhiều nhất).
    2. *PV-SQL đơn lẻ* (Chỉ dùng bộ lọc quy tắc AST và Probing).
    3. *System-1 0-Token Decision* (NanoJev-0.6B hoặc Laya-421M).
    4. *Adaptive Early-Exit Gate* (Dừng ngay ở phương án 1 nếu độ tự tin $\ge 0.85$).
  - Ghi nhận chi tiết lịch sử từng ứng viên vào file `logs/evaluation_results.jsonl`.

### 7.2. Phương Pháp Chấm Điểm Giải Mã 0-Token (0-Token Decoding Scoring)
- **Bản chất kỹ thuật**:
  - Các hệ thống cũ (như MARS-SQL Agent 3) bắt mô hình đọc lại toàn bộ câu hỏi và SQL rồi sinh ra các token giải thích hoặc từ "Yes/No", tiêu tốn hàng nghìn token và mất 5–10 giây cho mỗi lượt kiểm định.
  - Phương pháp 0-token đưa cặp (Context, Candidate SQL) vào mô hình Cross-Encoder (Laya-421M dựa trên ModernBERT-large, nặng ~85 MB), sử dụng đầu phân loại trực tiếp trên token `[MASK]`.
- **Tốc độ & Chi phí**:
  - Chấm điểm song song toàn bộ 8–16 phương án chỉ mất **~1.2 ms** (tương đương 810 quyết định/giây trên GPU T4).
  - Không sinh ra bất kỳ output token nào (0-token decoding overhead), giảm chi phí tính toán từ 40 đến 400 lần so với mô hình 7B.
- **Cổng Dừng Sớm Thích Ứng (Adaptive Early-Exit Gate)**:
  - Ngay khi Agent 2 sinh xong câu lệnh đầu tiên ($SQL_1$), nếu câu lệnh này vượt qua 10 quy tắc của PV-SQL và nhận điểm tự tin từ Laya-421M $\ge 0.85$, hệ thống dừng ngay lập tức và trả kết quả, tiết kiệm hơn 77% thời gian suy luận toàn tuyến.

### 7.3. Động Cơ Khóa Văn Phạm XGrammar PDA & Cơ Chế Fallback
- **Nguyên lý Ôtômat Đẩy Xuống (Pushdown Automaton - Dong et al., CMU/NVIDIA 2024)**:
  - Thay vì dịch lại toàn bộ ngữ pháp mỗi bước như Outlines CFG, XGrammar phân tách từ điển token của mô hình thành 2 nhóm:
    - *Nhóm từ cố định (>99% từ điển)*: Được tính trước thành bảng mặt nạ bit (bitmask cache) lưu sẵn trong bộ nhớ đệm.
    - *Nhóm từ phụ thuộc ngữ cảnh (<1% từ điển)*: Được kiểm tra trực tiếp với ngăn xếp (stack) lúc runtime.
  - **Tốc độ biên dịch**: Nạp và biên dịch lược đồ $S'$ chỉ mất **4–8 mili-giây** (nhanh hơn Outlines từ 250 đến 1,000 lần, vốn mất 2.0–4.9 giây).
  - **Mức giảm thông lượng (Throughput Penalty)**: Chỉ giảm nhẹ từ **1.6% đến 3.7%** tốc độ sinh token (so với mức sụt giảm 9%–13% của Outlines).
- **Định nghĩa Ngữ pháp SQL BNF/EBNF Dynamic**:
  - Tập luật văn phạm mô tả cú pháp chuẩn của SQLite (`SELECT ... FROM ... WHERE ... GROUP BY ... ORDER BY ... LIMIT ...`).
  - Các ký hiệu kết thúc (terminals) đại diện cho tên bảng và tên cột được chèn động dựa trên chính xác danh sách bảng và cột có trong lược đồ rút gọn $S'$ do Agent 1 cung cấp.
  - Ngăn chặn triệt để: Lỗi sai chính tả tên bảng (`fprm` vs `frpm`), gọi cột không tồn tại trong DB, và lỗi ép sai kiểu dữ liệu số/chuỗi (như viết `"4.98"` thay vì số `4.98`).
- **Cơ Chế Dự Phòng (Fallback Mechanism)**:
  - Nếu import thư viện `xgrammar` thất bại (chạy trên CPU không tương thích hoặc thiếu bản dựng C++ CUDA), hoặc quá trình biên dịch ngữ pháp gặp lỗi cú pháp lạ:
  - Hệ thống tự động bắt ngoại lệ, ghi log cảnh báo: `[XGrammar Warning] Falling back to Native Decoding` và chuyển sang chế độ giải mã tự nhiên (Native Decoding).
- **Phòng chống hiện tượng "Giải Cứu Rỗng" (Hollow Rescue)**:
  - Theo nghiên cứu của Chavan (2026), khóa văn phạm giúp mô hình nhỏ đạt 100% đúng cú pháp, nhưng câu lệnh có thể hoàn toàn sai ngữ nghĩa (ví dụ câu hỏi yêu cầu tính trung bình nhưng SQL lại dùng `SUM`).
  - Khắc phục bằng giải pháp 3 lớp: XGrammar khóa cú pháp ở Agent 2 + LLM-JEPA căn chỉnh ngữ nghĩa ở Agent 2 + PV-SQL bắt lỗi logic ở Agent 3.

### 7.4. Thăm Dò Cơ Sở Dữ Liệu & 10 Quy Tắc Cây Cú Pháp PV-SQL (Agent 3)
- **Thăm dò dữ liệu thật (Probing)**: Chạy tối đa 5 truy vấn ngắn có `LIMIT` trên cơ sở dữ liệu thật để kiểm tra định dạng dữ liệu thực tế (ví dụ: cột `state` lưu `'CA'` hay `'California'`) mà tốn 0 GB VRAM.
- **10 Quy tắc AST tất định (sử dụng thư viện `sqlglot`)**:
  1. `Distinct`: Câu hỏi chứa `unique`, `distinct`, `different` $\rightarrow$ AST bắt buộc phải có `DISTINCT` hoặc `GROUP BY`.
  2. `Top-K`: Câu hỏi chứa `top N`, `highest`, `lowest` $\rightarrow$ AST bắt buộc phải có `ORDER BY ... LIMIT`.
  3. `Average`: Câu hỏi chứa `average`, `mean` $\rightarrow$ AST bắt buộc phải có hàm `AVG()`.
  4. `Count`: Câu hỏi chứa `how many`, `number of` $\rightarrow$ AST bắt buộc phải có hàm `COUNT()`.
  5. `Sum / Total`: Câu hỏi chứa `total`, `sum` $\rightarrow$ AST bắt buộc phải có hàm `SUM()`.
  6. `Extreme`: Câu hỏi chứa `maximum`, `minimum` $\rightarrow$ AST bắt buộc phải có hàm `MAX()` hoặc `MIN()`.
  7. `Comparison`: Câu hỏi chứa so sánh hơn/kém $\rightarrow$ AST bắt buộc phải có mệnh đề `WHERE` chứa toán tử so sánh (`>`, `<`, `>=`, `<=`).
  8. `Null Handling`: Điều kiện kiểm tra `IS NOT NULL` khi truy vấn thuộc tính tùy chọn.
  9. `Join Validity`: Các liên kết `JOIN ... ON` trong AST phải tương thích với các cạnh trong đồ thị PK-FK.
  10. `Group By Completeness`: Mọi cột không nằm trong hàm tổng hợp ở mệnh đề `SELECT` bắt buộc phải xuất hiện trong mệnh đề `GROUP BY`.

### 7.5. Danh Mục Chỉ Số Đo Lường & Mục Tiêu Nghiệm Thu Benchmark
1. **Execution Accuracy (EX)**:
   - Tỷ lệ phần trăm câu lệnh SQL sinh ra thực thi trên SQLite trả về kết quả khớp $100\%$ với đáp án chuẩn Gold SQL (không xét thứ tự dòng nếu không có `ORDER BY`).
   - Công thức: $\text{EX} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{Exec}(SQL_i, DB_i) \equiv \text{Exec}(SQL_i^*, DB_i))$.
2. **Valid Efficiency Score (VES)**:
   - Đánh giá mức độ tối ưu về chi phí thực thi và tốc độ truy vấn của các câu lệnh chạy đúng:
     $$\text{VES} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{EX}_i = 1) \cdot \sqrt{\frac{T(SQL_i^*)}{T(SQL_i)}}$$
     (với $T$ là thời gian thực thi trung bình qua nhiều lượt chạy).
3. **Valid SQL Rate (VSR) / Syntax Validity**:
   - Tỷ lệ câu lệnh sinh ra hợp lệ về mặt cú pháp và cấu trúc bảng. Đạt **100%** nhờ động cơ XGrammar PDA.
4. **Schema Linking Recall (TPR)**:
   - Tỷ lệ thu hồi cột và bảng đúng trong Top-25 cột của Agent 1. Mục tiêu: đạt **$\text{TPR} \ge 95.23\%$** trên BIRD-dev.
5. **Bridge Tables Recovery Rate**:
   - Tỷ lệ cứu bảng cầu nối trên tập 48 câu hỏi BIRD-dev bị hỏng của MARS-SQL. Mục tiêu: đạt **100% (48/48 câu)** nhờ Cây Steiner.
6. **Expected Calibration Error (ECE)**:
   - Đo lường mức độ lệch giữa điểm tự tin dự đoán và xác suất chạy đúng thực tế. Mục tiêu: đạt **$ECE \le 0.081$** trên Laya-421M.
7. **Độ Trễ Toàn Tuyến (Latency)**:
   - Thời gian trung bình xử lý một câu hỏi từ lúc nhận đề bài đến lúc xuất kết quả SQL cuối cùng. Mục tiêu: **$\le 5.8$ giây** (Combo C) và **$\le 8.5$ giây** (Combo D), giảm >75% so với mốc 22.12 giây của MARS-SQL 21B.
8. **Bộ Nhớ VRAM Đỉnh (Peak VRAM)**:
   - Combo C: **3.5 – 8.5 GB VRAM** (giảm 81%–92% so với 41–47 GB của 21B).
   - Combo D: **< 20.0 GB VRAM** (chạy vừa trọn vẹn trên 1 card GPU 24GB như RTX 3090/4090 hoặc A100).

---

## 8. Bảng Ma Trận Các Tổ Hợp Kiến Trúc Toàn Tuyến (Combos A, B, C, D)

| Tổ Hợp (Combo) | Tổng Tham Số | Agent 1 (Grounding) | Agent 2 (Generation) | Agent 3 (Validation) | Phần Cứng Mục Tiêu | Mục Tiêu BIRD-dev (EX) | Mục Tiêu Spider-test (EX) | Độ Trễ TB | VRAM Đỉnh |
|---|:---:|---|---|---|---|:---:|:---:|:---:|:---:|
| **MARS-SQL Full (8,8)** *(Baseline 21B)* | 21.0B ($3 \times 7\text{B}$) | Grounding Agent 7B | Generation Agent 7B | Validation Agent 7B | $2 \times$ A100 (48GB) | 77.84% | 89.75% (DK: 78.13%) | 22.12s | 41.0 – 47.0 GB |
| **Combo A** *(Ultra-Light Edge)* | ~2.5B | LitE-SQL (0.6B) + Cây Steiner | SLM-SQL-1.5B + LLM-JEPA + XGrammar | PV-SQL + Von-395M (CPU) / Laya-421M | Laptop CPU / <8GB RAM | 69.5% – 72.0% | 83.0% – 85.5% | ~3.8s – 5.2s | < 8.0 GB RAM (hoặc ~2.5 GB Q4) |
| **Combo B** *(Shared System-1)* | ~3.6B | NanoJev-0.6B (dùng chung) + Cây Steiner | FINER-SQL-3B + LLM-JEPA + XGrammar | PV-SQL + NanoJev-0.6B (Domain SFT) | 1 GPU 4GB–8GB VRAM | 73.5% – 75.5% | 87.5% – 89.0% | ~4.5s – 6.2s | 3.2 GB (Q4) – 7.5 GB (F16) |
| **Combo C** *(Pareto SOTA)* | ~4.0B | LitE-SQL (0.6B) + Cây Steiner + GNN | FINER-SQL-3B + LLM-JEPA + XGrammar | PV-SQL + Laya-421M (RLCD, Early-Exit 0.85) | 1 GPU 8GB–16GB (Colab T4/L4) | **75.0% – 76.8%** | **88.5% – 89.5%** | **~4.2s – 5.8s** (Giảm 77%) | **3.5 GB (Q4) – 8.5 GB (F16)** |
| **Combo D** *(High Accuracy)* | ~8.0B | LitE-SQL (0.6B) + Cây Steiner + GNN | Qwen2.5-Coder-7B + LLM-JEPA + XGrammar | PV-SQL + Laya-421M (RLCD) | 1 GPU 24GB (Colab A100) | **> 77.84%** (Vượt 21B) | **$\ge$ 89.75%** | ~6.5s – 8.5s | < 20.0 GB VRAM |

---

## 9. Tiêu Chí Hoàn Thành (Definition of Done - DoD) Cho Từng Phase

```
┌────────────────────────────────────────────────────────────────────────┐
│ CHECKLIST NGHIỆM THU THEO ĐẶC TẢ TÀI LIỆU 04 (DEFINITION OF DONE)      │
├────────────────────────────────────────────────────────────────────────┤
│ PHASE 1: DATA PIPELINE & SCHEMA GROUNDING                              │
│ [ ] Xuất thành công DATA-G1-EMB.jsonl (90,102 thẻ cột, 6 trường, y∈{0,1})│
│ [ ] Dựng đủ 95 đồ thị BIRD và 138 đồ thị Spider trong DATA-G1-GRAPH.pkl│
│ [ ] Tập SFT (DATA-G2-SFT.jsonl) chứa 916k SynSQL + 8k Reasoning Bank   │
│ [ ] Thư mục data/mini/ đầy đủ 100 câu BIRD, 1k SynSQL, 2 SQLite DBs    │
│ [ ] Chạy scripts/make_mini_dataset.py hoàn thành < 2 phút không lỗi   │
├────────────────────────────────────────────────────────────────────────┤
│ PHASE 2: TRAINING SCRIPTS (src/training/)                              │
│ [ ] train_embedding.py huấn luyện HN-SupCon và xuất offline_index     │
│ [ ] train_sft_generator.py hỗ trợ 3B/7B, nhãn -100 trên đề bài        │
│ [ ] jepa_module.py tích hợp 2-block causal mask, λ=1.0, LD=0.5         │
│ [ ] train_grpo_generator.py tích hợp 4 đầu thưởng chống sụp đổ lợi thế │
│ [ ] train_decision_model.py hỗ trợ NanoJev Domain SFT và Laya RLCD    │
│ [ ] DriveCheckpointCallback sao lưu tự động và phục hồi checkpoint    │
│ [ ] Toàn bộ script pass smoke test 1 batch trên data/mini/            │
├────────────────────────────────────────────────────────────────────────┤
│ PHASE 3: NOTEBOOKS EXECUTION (notebooks/)                              │
│ [ ] 01_colab_data_prep.ipynb chạy thông suốt trên Colab               │
│ [ ] 02_colab_train_grounding.ipynb xuất offline_column_index.npy       │
│ [ ] 03_colab_train_generator_3b.ipynb hỗ trợ Colab T4 (QLoRA) & L4    │
│ [ ] 03b_colab_train_generator_7b.ipynb hỗ trợ Colab A100               │
│ [ ] 04_colab_train_validator.ipynb huấn luyện thành công Laya-421M     │
│ [ ] Checkpoints được lưu trữ an toàn và nạp lại thành công từ Drive    │
├────────────────────────────────────────────────────────────────────────┤
│ PHASE 4: EVALUATION & BENCHMARK (src/evaluation/)                      │
│ [ ] evaluate_end_to_end.py kết nối trơn tru 3 tác tử                  │
│ [ ] Sinh 8–16 quỹ đạo (T=0.7, p=0.9, k=50) cho ablation study         │
│ [ ] XGrammar PDA đạt 100% đúng cú pháp, biên dịch 4–8 ms, có fallback │
│ [ ] Cây Steiner cứu thành công 100% (48/48) câu mất bảng cầu nối       │
│ [ ] PV-SQL kiểm tra đủ 10 quy tắc AST và chạy 5 truy vấn probing      │
│ [ ] Laya-421M chấm điểm 0-token trong ~1.2 ms, ECE ≤ 0.081            │
│ [ ] Combo C đạt EX ≥ 75.0% trên BIRD-dev, độ trễ ≤ 5.8s                │
│ [ ] Xuất file log evaluation_results.jsonl và benchmark_summary.md    │
└────────────────────────────────────────────────────────────────────────┘
```

---
*Báo cáo được lập bởi `spec_miner_survey_2` dựa trên việc phân tích toàn diện bộ tài liệu kỹ thuật gốc của dự án. Tài liệu này đóng vai trò là kim chỉ nam đặc tả cho toàn bộ đội ngũ lập trình.*
