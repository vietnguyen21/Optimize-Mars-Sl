# Text-to-SQL SLM Multi-Agent System (Dự Án Thay Thế MARS-SQL)

Hệ thống đa tác tử nhẹ (Lightweight Multi-Agent SLM) nghiên cứu thay thế MARS-SQL 21B cho bài toán Text-to-SQL phức tạp trên quy mô cơ sở dữ liệu lớn (BIRD, Spider).

## Language

### Kiến Trúc Tác Tử & Luồng Xử Lý

**Grounding Agent (Agent 1)**:
Tác tử xác định và trích xuất đúng các bảng, cột và liên kết khóa liên quan mật thiết đến câu hỏi tự nhiên.
_Avoid_: Schema Retriever, Table Selector, Schema Filter

**Generation Agent (Agent 2)**:
Tác tử sinh câu lệnh SQL từ câu hỏi và lược đồ rút gọn, tích hợp cơ chế suy luận và khóa văn phạm cú pháp.
_Avoid_: SQL Writer, Code Generator, LLM Coder

**Validation Agent (Agent 3)**:
Tác tử kiểm định chất lượng, rà soát lỗi logic/AST và chấm điểm chọn ra câu lệnh SQL tối ưu nhất trong số các phương án ứng viên.
_Avoid_: SQL Evaluator, Ranker, Filter Agent

**Bridge Table (Bảng Cầu Nối)**:
Bảng trung gian không chứa từ khóa xuất hiện trong câu hỏi người dùng nhưng bắt buộc phải có để thực hiện liên kết khóa ngoại (JOIN) giữa các bảng chính.
_Avoid_: Intermediate Table, Junction Table, Intermediate Join

**Candidate Combos (Tổ Hợp Kiến Trúc)**:
Các cấu hình ghép nối toàn tuyến giữa 3 tác tử được thiết kế theo phân khúc phần cứng: Combo A (Siêu nhẹ ~2.5B), Combo B (Dùng chung ~3.6B), Combo C (Pareto SOTA ~4.0B), Combo D (Chính xác cao ~8.0B).
_Avoid_: Models, Pipelines, Architectures

### Cơ Chế Huấn Luyện & Mô Hình

**Completion-Only Loss**:
Kỹ thuật chỉ tính hàm mất mát (loss) trên phần chuỗi suy luận và câu lệnh SQL do mô hình tự sinh ra, che toàn bộ đề bài và DDL để ngăn học vẹt.
_Avoid_: Answer Loss, Response Masking

**LLM-JEPA (Joint Embedding Predictive Architecture)**:
Phương pháp căn chỉnh biểu diễn ẩn giữa câu hỏi (Text) và câu lệnh SQL (Code) trong quá trình huấn luyện nhằm tăng cường hiểu biết ngữ nghĩa mà không tốn chi phí suy diễn.
_Avoid_: Contrastive Learning, Multi-modal Alignment

**Hard Negatives (HN-SupCon)**:
Tập mẫu âm khó gồm các cột không có trong đáp án nhưng có độ tương đồng ngữ nghĩa cao hoặc trùng tên với cột đúng, dùng để huấn luyện bộ nhúng phân biệt cột chính xác.
_Avoid_: Negative Samples, Confusing Columns

**Reasoning Bank**:
Ngân hàng các luồng suy luận từng bước chất lượng cao được thu thập từ mô hình lớn trên các mẫu đúng 100%, phục vụ SFT và chấm điểm thưởng bước nghĩ.
_Avoid_: CoT Bank, Solution Database

**Dense GRPO (4 Phần Thưởng Dày)**:
Thuật toán tối ưu hóa chính sách nhóm không cần Critic với hàm thưởng đa thành phần (Định dạng, Thực thi, Nhớ tư duy, Mảnh ghép toán tử) để tránh sụp đổ lợi thế (advantage collapse) trên mô hình nhỏ.
_Avoid_: Multi-Reward RL, Standard GRPO

### Kiểm Định & Giới Hạn

**0-Token Decision**:
Cơ chế phán quyết hoặc chấm điểm song song các phương án SQL bằng đầu phân loại (Classification/Choice Head) mà không sinh thêm bất kỳ token giải thích nào.
_Avoid_: Zero-Shot Decision, Fast Classification

**XGrammar PDA (Pushdown Automaton)**:
Động cơ khóa văn phạm thời gian thực bằng ôtômat đẩy xuống, chặn 100% lỗi cú pháp và lỗi bịa tên bảng ngay tại bước sinh token.
_Avoid_: Outlines, Constrained Sampler, Grammar Mask

**Hollow Rescue (Giải Cứu Rỗng)**:
Hiện tượng câu lệnh SQL sinh ra đúng 100% về cú pháp và cấu trúc bảng nhờ khóa văn phạm nhưng kết quả thực thi sai do mô hình hiểu sai ngữ nghĩa câu hỏi.
_Avoid_: Semantic Gap, Syntax Trap

**Abstention (Từ Chối Phán Quyết)**:
Khả năng của mô hình kiểm định chủ động từ chối chọn hoặc yêu cầu sinh lại khi phát hiện toàn bộ các phương án ứng viên đều sai hoặc dữ liệu đầu vào không đủ tin cậy.
_Avoid_: Reject All, Fallback, Null Decision

**Calibrated Decision (Hiệu Chuẩn Xác Suất - RLCD)**:
Mô hình chấm điểm có xác suất đầu ra phản ánh trung thực xác suất chạy đúng trên thực tế (sai số hiệu chuẩn ECE thấp), triệt tiêu sự tự tin ảo.
_Avoid_: Confidence Calibration, Probability Tuning

### Thước Đo Thực Nghiệm

**Execution Accuracy (EX)**:
Tỷ lệ phần trăm các câu lệnh SQL sinh ra thực thi trên cơ sở dữ liệu trả về bảng kết quả trùng khớp hoàn toàn với đáp án chuẩn.
_Avoid_: SQL Accuracy, Match Rate

**Schema Linking Recall (TPR)**:
Tỷ lệ phần trăm các cột và bảng cần thiết trong đáp án chuẩn được tác tử Grounding truy xuất thành công.
_Avoid_: Linking Precision, Retrieval Rate

**Valid Efficiency Score (VES)**:
Điểm đo lường mức độ tối ưu về chi phí thực thi và tốc độ truy vấn của câu lệnh SQL đúng trên công cụ cơ sở dữ liệu.
_Avoid_: Execution Speed Score, SQL Efficiency
