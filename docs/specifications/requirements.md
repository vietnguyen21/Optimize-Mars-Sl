# DANH MỤC THƯ VIỆN & PHỤ THUỘC (DEPENDENCY SPECIFICATION)

Tài liệu này đặc tả chi tiết toàn bộ các thư viện Python, phiên bản kiểm chuẩn và mục đích sử dụng cho hệ thống đa tác tử Text-to-SQL trên Google Colab.

---

## 1. Bảng Chi Tiết Các Gói Thư Viện

| Tên Gói (Package) | Phiên Bản Khuyến Nghị | Mô-đun Sử Dụng | Mục Đích Kỹ Thuật |
| :--- | :---: | :--- | :--- |
| **`torch`** | `>=2.2.0` | Toàn bộ 3 Tác tử | Khung tính toán Tensor và huấn luyện Deep Learning trên CUDA. |
| **`transformers`** | `>=4.44.0` | Agent 1, 2, 3 | Nạp và tinh chỉnh các mô hình `Qwen3`, `FINER-SQL`, `ModernBERT`. |
| **`accelerate`** | `>=0.33.0` | `src/training/` | Quản lý phân bổ bộ nhớ GPU, phân tán và tối ưu hóa tính toán hỗn hợp (FP16/BF16). |
| **`peft`** | `>=0.12.0` | `src/training/` | Huấn luyện hiệu quả tham số (LoRA / QLoRA rank 512) giúp tiết kiệm VRAM trên Colab. |
| **`trl`** | `>=0.9.6` | `src/training/` | Bộ công cụ huấn luyện căn chỉnh `SFTTrainer` và thuật toán `GRPO`. |
| **`bitsandbytes`** | `>=0.43.0` | Agent 2 | Nạp lượng tử hóa 4-bit / 8-bit trên GPU T4 của Colab. |
| **`xgrammar`** | `>=0.1.5` | Agent 2 | Động cơ khóa văn phạm thời gian thực (PDA) chặn 100% lỗi cú pháp và ảo giác bảng. |
| **`sqlglot`** | `>=25.0.0` | Agent 3 (`PV-SQL`) | Phân tích câu lệnh SQL thành cây cú pháp (AST) để kiểm tra 10 quy tắc tất định. |
| **`networkx`** | `>=3.2.1` | Agent 1 (`Graph-Link`)| Dựng đồ thị PK-FK và giải bài toán Cây Steiner có trọng số nút (NWST) cứu bảng cầu nối. |
| **`sentence-transformers`**| `>=3.0.0` | Agent 1 | Huấn luyện bộ nhúng `Qwen3-Embedding-0.6B` với hàm mất mát HN-SupCon. |
| **`scipy`** | `>=1.12.0` | Agent 1, 3 | Tính toán ma trận khoảng cách cosine và thuật toán tối ưu trên đồ thị. |
| **`numpy`** | `>=1.26.0` | Toàn hệ thống | Xử lý mảng vector nhúng và ma trận trọng số. |
| **`pydantic`** | `>=2.7.0` | Toàn hệ thống | Xác thực cấu trúc dữ liệu I/O giữa các tác tử và file cấu hình. |
| **`tqdm`** | `>=4.66.0` | Pipeline | Hiển thị thanh tiến trình xử lý dữ liệu và đánh giá. |
| **`gdown`** | `>=5.1.0` | Phase 1 | Tự động tải các tập dữ liệu lớn từ Google Drive vào môi trường Colab. |

---

## 2. Ma Trận Tương Thích CUDA & PyTorch

- **Môi trường Colab chuẩn**: Python 3.10+, CUDA 12.1 hoặc 12.2.
- **Tùy chọn Precision**:
  - GPU Ampere/Ada/Hopper (A100, L4): Bật **BF16** (`bfloat16=True`) để đạt hiệu năng tối đa và tính ổn định gradient cao nhất.
  - GPU Turing (Tesla T4): Bật **FP16** (`fp16=True`).

---

## 3. Nội Dung Tệp `requirements.txt` Chuẩn

Dưới đây là nội dung chuẩn hóa của file `requirements.txt`:

```text
# Deep Learning & Core LLM
torch>=2.2.0
transformers>=4.44.0
datasets>=2.20.0
accelerate>=0.33.0
peft>=0.12.0
trl>=0.9.6
bitsandbytes>=0.43.0

# Constrained Decoding & Grammar
xgrammar>=0.1.5

# Graph & Mathematics
networkx>=3.2.1
scipy>=1.12.0
numpy>=1.26.0

# SQL Parsing & AST Rules
sqlglot>=25.0.0

# Embeddings & Retrieval
sentence-transformers>=3.0.0

# Utilities & Data
pydantic>=2.7.0
tqdm>=4.66.0
gdown>=5.1.0
```
