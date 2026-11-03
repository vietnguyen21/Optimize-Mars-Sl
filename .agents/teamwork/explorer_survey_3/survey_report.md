# BÁO CÁO KHẢO SÁT KỸ THUẬT: MÔI TRƯỜNG, SMOKE TEST & CHIẾN LƯỢC BÀN GIAO (SURVEY REPORT)

> **Tác tử thực hiện**: `explorer_survey_3` (Teamwork Explorer)  
> **Ngày thực hiện**: 2026-11-02  
> **Dự án**: Hệ thống Đa tác tử SLM Text-to-SQL thay thế MARS-SQL 21B  
> **Tài liệu tham chiếu**: `AGENTS_RESEARCH.md`, `ORIGINAL_REQUEST.md`, `docs/specifications/01–05`, `requirements.txt`, `GLOSSARY.md`.

---

## 1. Tóm Tắt Khảo Sát (Executive Summary)

Nhiệm vụ của `explorer_survey_3` là khảo sát toàn diện môi trường làm việc, phụ thuộc kỹ thuật, đề xuất kiến trúc bộ kiểm thử nhanh (**Smoke Test Architecture & Strategy**) trên tập dữ liệu nhỏ gọn `data/mini/`, và xây dựng cấu trúc tài liệu bàn giao kỹ thuật (**Handover Documentation Strategy**) cho file `HANDOVER.md` tại thư mục gốc theo yêu cầu R3 và Acceptance Criteria.

### Phát Hiện Nòng Cốt:
1. **Môi trường cục bộ**: Host chạy Linux (Ubuntu 24.04 LTS), Python 3.12.3, card đồ họa NVIDIA GeForce RTX 4050 Laptop (6GB VRAM, CUDA 13.0 driver). Môi trường Python hệ thống kích hoạt **PEP 668** (`externally-managed-environment`), ngăn chặn `pip install` trực tiếp ra ngoài virtualenv.
2. **Hiện trạng thư viện**: Các gói học sâu (`torch`, `transformers`, `trl`, `peft`, `xgrammar`, `bitsandbytes`) và công cụ đo kiểm (`pytest`, `sqlglot`, `networkx`) chưa được cài đặt trong môi trường gốc.
3. **Chiến lược kiểm thử độc lập (Zero Heavy Download)**: Bộ smoke test phải chạy được 100% bằng thư viện chuẩn (`unittest`) hoặc `pytest` trong virtualenv, không đòi hỏi tải mô hình 3B/7B hàng chục GB hay đòi hỏi GPU rời. Mọi luồng kiểm thử đều dùng Dummy/Mock Models, SQLite mẫu và tập dữ liệu mô phỏng chuẩn xác tại `data/mini/`.
4. **Cơ chế phòng hộ XGrammar**: `xgrammar` đòi hỏi C++/CUDA runtime có thể không sẵn sàng trên Python 3.12 cục bộ. Do đó, mã nguồn phải xây dựng lớp trừu tượng `GrammarEngine` với chế độ tự động kích hoạt `RegexASTFallbackEngine` kèm cảnh báo thân thiện, đảm bảo không bao giờ dừng đột ngột (crash-free).
5. **Cấu trúc HANDOVER.md chuẩn hóa**: Phản ánh đầy đủ 4 vai trò tác tử theo `AGENTS_RESEARCH.md`, danh mục toàn bộ file mã nguồn, hướng dẫn thiết lập 2 tầng (Cục bộ & Google Colab GPU), quy trình đồng bộ Google Drive và tái lập toàn bộ số liệu thực nghiệm.

---

## 2. Khảo Sát Môi Trường & Phụ Thuộc Kỹ Thuật (Environment & Dependencies)

### 2.1. Thông Số Hệ Thống Thực Tế
| Thông Số | Giá Trị Thực Tế Tại Workspace | Ý Nghĩa Kỹ Thuật |
| :--- | :--- | :--- |
| **Hệ Điều Hành** | Linux 6.6.x (Ubuntu 24.04 LTS - WSL2) | Tương thích cao với các script Bash/Python chuẩn POSIX. |
| **Phiên Bản Python** | Python 3.12.3 (`/usr/bin/python3`) | Phiên bản Python rất mới; một số wheel nhị phân C++ cũ có thể cần build từ source. |
| **Quản Lý Gói** | Pip 24.0 (PEP 668 active) | Bắt buộc sử dụng `python3 -m venv .venv` khi cài đặt cục bộ, tránh xung đột OS packages. |
| **GPU Cục Bộ** | NVIDIA RTX 4050 Laptop (6GB VRAM) | Đủ để chạy suy luận mô hình siêu nhẹ (0.6B) hoặc test thử nghiệm, nhưng không đủ để train full SFT 7B hoặc GRPO nặng. |
| **Môi Trường Target (Colab)** | Google Colab (T4 15GB, L4 22.5GB, A100 40GB) | Python 3.10/3.11, CUDA 12.x; ổ đĩa tạm `/content/` cần đồng bộ Drive tự động. |

### 2.2. Phân Tích Phụ Thuộc Trong `requirements.txt`
Nội dung file `requirements.txt` chuẩn hóa gồm 15 thư viện chính:
```text
# Deep Learning & Core LLM: torch, transformers, datasets, accelerate, peft, trl, bitsandbytes
# Constrained Decoding: xgrammar>=0.1.5
# Graph & Mathematics: networkx>=3.2.1, scipy>=1.12.0, numpy>=1.26.0
# SQL Parsing & AST: sqlglot>=25.0.0
# Embeddings & Retrieval: sentence-transformers>=3.0.0
# Utilities & Data: pydantic>=2.7.0, tqdm>=4.66.0, gdown>=5.1.0
```

### 2.3. Rủi Ro Tiềm Ẩn & Giải Pháp Xử Lý
1. **Rủi ro PEP 668 trên Host**:
   - *Rủi ro*: Người dùng chạy `pip install -r requirements.txt` trực tiếp sẽ bị lỗi `error: externally-managed-environment`.
   - *Giải pháp*: Trong `HANDOVER.md` và hướng dẫn cài đặt, quy định rõ lệnh chuẩn:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     pip install --upgrade pip
     pip install -r requirements.txt
     ```
2. **Rủi ro XGrammar Không Khả Dụng Cục Bộ**:
   - *Rủi ro*: `xgrammar>=0.1.5` phụ thuộc vào pre-built wheel tương thích CUDA/PyTorch. Nếu chạy trên máy không có CUDA 12 hoặc Python 3.12 chưa có wheel, lệnh `import xgrammar` sẽ ném lỗi `ImportError`.
   - *Giải pháp*: Bắt buộc cài đặt cơ chế **Graceful Fallback**:
     ```python
     try:
         import xgrammar
         HAS_XGRAMMAR = True
     except ImportError:
         HAS_XGRAMMAR = False
     ```
     Khi `HAS_XGRAMMAR is False`, hệ thống tự chuyển sang bộ giải mã bản địa (Native Decoding) kết hợp bộ lọc AST của `sqlglot` mà không gây gián đoạn chương trình.
3. **Rủi ro Thiếu Công Cụ Test (`pytest`)**:
   - *Rủi ro*: Môi trường chưa cài `pytest`.
   - *Giải pháp*: Toàn bộ các test case trong thư mục `tests/` phải được viết tương thích với cả `unittest` chuẩn của Python (`python3 -m unittest discover tests`) và `pytest tests/`. Cung cấp thêm script thực thi nhanh `python scripts/run_smoke_tests.py` không cần cài thêm bất kỳ gói ngoài nào.
4. **Rủi ro Gián Đoạn Phiên Google Colab**:
   - *Rủi ro*: Hết thời gian cấp GPU Colab (disconnect) làm mất checkpoint.
   - *Giải pháp*: Sử dụng callback `DriveCheckpointCallback` (đã quy định trong tài liệu 05) tự động copy mỗi checkpoint sau `save_steps` sang `/content/drive/MyDrive/Text2SQL_Colab/checkpoints/` và cờ phục hồi `--resume_from_checkpoint latest`.

---

## 3. Kiến Trúc & Chiến Lược Bộ Kiểm Thử Nhanh (Smoke Test Architecture & Strategy)

### 3.1. Nguyên Tắc Thiết Kế (Design Principles)
- **Zero-Heavy Download**: Tuyệt đối không yêu cầu tải trước các checkpoint mô hình nặng (3B, 7B hay 0.6B) khi chạy smoke test. Sử dụng Mock/Dummy Tokenizer và Dummy Model có kích thước mini (vài KB đến vài MB) để kiểm thử luồng tensor và đạo hàm loss.
- **Tốc độ dưới 60 giây**: Toàn bộ chuỗi smoke test từ Phase 1 đến Phase 4 phải hoàn tất trong vòng dưới 1 phút trên CPU thông thường.
- **Dữ liệu thật thu nhỏ (Realistic Miniature Data)**: Dữ liệu trong `data/mini/` gồm các file SQLite thực tế, có bảng chính, bảng cầu nối và khóa ngoại PK-FK thật để các câu lệnh SQL chạy thực thi và so sánh kết quả thực tế.
- **Bao phủ 100% 4 Phase Chức Năng**: Xác minh tính đúng đắn về thuật toán đồ thị, hàm mất mát, cấu trúc notebook và cơ chế khóa văn phạm/phán quyết.

### 3.2. Cấu Trúc Thư Mục Dữ Liệu Thu Nhỏ (`data/mini/`)
Thư mục `data/mini/` sẽ được thiết kế với dung lượng < 2 MB, tự đầy đủ (self-contained):
```
data/mini/
├── sqlite/
│   ├── financial_mini.sqlite      # SQLite DB mẫu có 3 bảng: accounts, transactions, transfers (chứa PK-FK và bridge table)
│   └── california_schools.sqlite  # SQLite DB mẫu mô phỏng BIRD với bảng: schools, frpm, satscores
├── bird_mini.json                 # 20 mẫu câu hỏi BIRD chuẩn, gắn nhãn Gold SQL, DB ID, evidence
├── synsql_mini.jsonl              # 50 mẫu SynSQL có thẻ <think>...</think><answer>...</answer>
├── columns_mini.jsonl             # 30 thẻ thông tin cột chuẩn 6 trường (positive & hard negative)
├── graphs_mini.pkl                # File pickle lưu trữ đồ thị NetworkX PK-FK của 2 DB trên
└── candidates_mini.jsonl          # 10 mẫu câu hỏi, mỗi câu kèm 8 phương án SQL ứng viên mô phỏng để test Agent 3
```

### 3.3. Chi Tiết Kiểm Thử 4 Phase

#### Phase 1: Data Pipeline & Schema Grounding
- **Test 1.1 (Graph Construction)**:
  - Khởi tạo đồ thị `SchemaGraphBuilder` từ SQLite DDL.
  - Kiểm tra số đỉnh Bảng, đỉnh Cột và cạnh nối `FOREIGN KEY`.
  - Khẳng định không có cạnh nối bị gãy (broken links).
- **Test 1.2 (Steiner Tree & Bridge Table Recovery)**:
  - Cho tập terminal nodes gồm 2 cột thuộc 2 bảng không nối trực tiếp (ví dụ: `schools` và `satscores` qua bảng trung gian `frpm`).
  - Chạy thuật toán `NodeWeightedSteinerTree`.
  - Xác minh 100% bảng cầu nối trung gian được bảo toàn trong đồ thị con $S'$.
- **Test 1.3 (Column Packaging & HN-SupCon Format)**:
  - Đóng gói thẻ cột 6 trường thông tin.
  - Xác minh cấu trúc JSON và nhãn $y \in \{0, 1\}$.
- **Test 1.4 (SynSQL Tagging & Data Cleaner)**:
  - Kiểm tra regex trích xuất thẻ `<think>` và `<answer>`.
  - Kiểm tra bộ lọc loại bỏ comment `--`, câu thiếu `SELECT` và độ dài vượt ngưỡng.

#### Phase 2: SLM Training Scripts (Dry Run & Tensor Verification)
- **Test 2.1 (Completion-Only Loss Masking)**:
  - Tạo tensor giả lập prompt + `<think>` + `<answer>`.
  - Chạy hàm `mask_labels_for_completion_only()`.
  - Khẳng định toàn bộ token thuộc prompt nhận nhãn `-100`, chỉ token trong completion có nhãn gốc.
- **Test 2.2 (LLM-JEPA 2-Block Causal Mask & Latent Loss)**:
  - Tạo mô-đun JEPA giả lập.
  - Xác nhận ma trận 2-block causal mask: Text block không nhìn trộm Code block.
  - Kiểm tra tính toán hàm mất mát cosine distance và cờ JEPA Dropout ($LD=0.5$).
- **Test 2.3 (Dense GRPO 4 Reward Heads)**:
  - Kiểm tra 4 hàm tính thưởng:
    * `format_reward`: +1.0 nếu có đủ `<think>` và `<answer>`, 0.0 nếu thiếu.
    * `exec_reward`: Chạy thực thi câu SQL trên `financial_mini.sqlite`, trả về +1.0 nếu kết quả khớp Gold SQL.
    * `memory_reward`: Phạt nếu câu SQL dài dòng hoặc thừa JOIN không cần thiết.
    * `atomic_reward`: Kiểm tra tính hợp lệ của toán tử SQL.
- **Test 2.4 (DriveCheckpointCallback)**:
  - Khởi tạo callback với thư mục local tạm và Drive tạm.
  - Kích hoạt sự kiện `on_save` và xác minh file checkpoint được sao lưu chính xác.
- **Test 2.5 (1-Step Training Dry Run)**:
  - Dùng `DummyModel` (kích thước siêu nhỏ ~100KB) chạy 1 bước `forward()` và `backward()` để đảm bảo pipeline huấn luyện không bị crash gradient.

#### Phase 3: Colab Notebook Structure Verification
- **Test 3.1 (Notebook JSON & Syntax Integrity)**:
  - Duyệt toàn bộ 5 notebook trong `notebooks/`:
    * `01_colab_data_prep.ipynb`
    * `02_colab_train_grounding.ipynb`
    * `03_colab_train_generator_3b.ipynb`
    * `03b_colab_train_generator_7b.ipynb`
    * `04_colab_train_validator.ipynb`
  - Đọc dưới dạng JSON, kiểm tra tính hợp lệ của cấu trúc `cells`, `metadata`, `nbformat >= 4`.
- **Test 3.2 (Mandatory Cells & Workflow Assertions)**:
  - Kiểm tra sự hiện diện của lệnh kết nối Drive: `google.colab import drive`.
  - Kiểm tra lệnh cài đặt `requirements.txt`.
  - Kiểm tra kiểm tra phần cứng `!nvidia-smi`.
  - Kiểm tra cờ phục hồi checkpoint `--resume_from_checkpoint`.

#### Phase 4: Evaluation, Benchmark & Constrained Decoding
- **Test 4.1 (XGrammar PDA Engine & Safe Fallback)**:
  - Thử nghiệm khởi tạo grammar từ sub-schema $S'$.
  - Giả lập trường hợp không có thư viện `xgrammar` $\rightarrow$ hệ thống chuyển sang `RegexASTFallbackEngine` an toàn.
  - Xác nhận tỷ lệ crash = 0%.
- **Test 4.2 (Candidate Rollout Generator)**:
  - Kiểm tra logic sinh $K=8$ và $K=16$ quỹ đạo ứng viên.
  - Xác minh định dạng đầu ra danh sách ứng viên JSONL.
- **Test 4.3 (PV-SQL 10 AST Deterministic Rules)**:
  - Kiểm tra độc lập 10 quy tắc cú pháp bằng `sqlglot` đối chiếu với từ khóa câu hỏi:
    1. Distinct Rule (`unique`, `distinct`)
    2. Top-K Rule (`top`, `highest`, `lowest`)
    3. Average Rule (`average`, `mean`)
    4. Count Rule (`how many`, `number of`)
    5. Sum Rule (`sum`, `total`)
    6. Extreme Rule (`maximum`, `minimum`)
    7. Comparison Rule (`greater`, `less`, `more than`)
    8. Null Handling Rule (`is not null`)
    9. Join Validity Rule (hợp lệ theo PK-FK)
    10. Group By Completeness Rule (cột không tổng hợp trong SELECT phải có trong GROUP BY)
  - Viết test case cho cả mẫu SQL vi phạm và mẫu SQL tuân thủ.
- **Test 4.4 (Metric Calculation Harness)**:
  - Kiểm tra hàm tính Execution Accuracy (EX) trên SQLite thật.
  - Kiểm tra hàm tính Valid Efficiency Score (VES).
  - Kiểm tra hàm tính Schema Linking Recall (TPR).
  - Kiểm tra hàm tính Expected Calibration Error (ECE) với các bin xác suất.
  - Kiểm tra kịch bản cứu 48 câu mất bảng cầu nối.

### 3.4. Cấu Trúc Thư Mục `tests/` Đề Xuất
```
tests/
├── __init__.py
├── conftest.py                    # Mock fixtures, đường dẫn dữ liệu mini, dummy models
├── test_phase1_data_pipeline.py    # Test đồ thị PK-FK, Cây Steiner, thẻ cột, SynSQL cleaner
├── test_phase2_training_logic.py  # Test Completion loss, JEPA, GRPO reward, Drive callback
├── test_phase3_notebooks.py       # Test cấu trúc JSON và cells của các Colab notebooks
├── test_phase4_evaluation.py      # Test 10 quy tắc PV-SQL, XGrammar fallback, bộ đo EX/VES/ECE
└── test_end_to_end_pipeline.py    # Smoke test tích hợp toàn tuyến từ Schema -> SQL -> Eval
```

### 3.5. Kịch Bản Chạy Test Nhanh (`scripts/run_smoke_tests.py`)
Cung cấp một file script Python độc lập tại thư mục `scripts/` giúp người dùng hoặc hệ thống CI chạy toàn bộ smoke test chỉ bằng 1 câu lệnh:
```bash
python scripts/run_smoke_tests.py
```
Script này tự động:
1. Nạp test suite từ `unittest`.
2. Chạy lần lượt các nhóm test Phase 1 $\rightarrow$ Phase 4.
3. Xuất bảng tổng kết trực quan (bằng văn bản console) hiển thị trạng thái `PASSED` / `FAILED` cho từng phase.
4. Trả về mã thoát (exit code) `0` nếu toàn bộ passed, giúp xác minh nghiệm thu tự động theo tiêu chí C2.

---

## 4. Chiến Lược Tích Hợp XGrammar & Chế Độ Dự Phòng (Safe XGrammar Fallback)

### 4.1. Vấn Đề Kỹ Thuật Của XGrammar
`xgrammar` là thư viện sử dụng Pushdown Automaton (PDA) biên dịch ngữ pháp văn cảnh độc lập (Context-Free Grammar - CFG) trực tiếp thành mặt nạ bitmask trên từ điển token của LLM.
- **Ưu điểm**: Chặn 100% lỗi cú pháp SQL và lỗi ảo giác tên bảng, thời gian biên dịch chỉ mất 4–8 ms.
- **Thách thức**: Đòi hỏi pre-built binary tương thích với phiên bản PyTorch, CUDA và Python. Khi chạy cục bộ trên môi trường CPU, hoặc các nền tảng chưa được hỗ trợ, lệnh import hoặc khởi tạo PDA engine sẽ thất bại.

### 4.2. Kiến Trúc Lớp Trừu Tượng `GrammarEngine`
Để đảm bảo mã nguồn vận hành trơn tru ở mọi môi trường, áp dụng mẫu thiết kế **Strategy Pattern**:

```
                         ┌───────────────────────┐
                         │   BaseGrammarEngine   │ (Trừu tượng)
                         └───────────┬───────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
    ┌─────────────────────────┐             ┌─────────────────────────┐
    │    XGrammarPDAEngine    │             │ RegexASTFallbackEngine  │
    │  (Được chọn khi GPU     │             │  (Tự động kích hoạt khi  │
    │   & xgrammar sẵn sàng)  │             │   không có xgrammar)    │
    └─────────────────────────┘             └─────────────────────────┘
```

### 4.3. Thiết Kế Mã Nguồn Fallback Chuẩn
Trong module `src/agents/generation/grammar_engine.py`:
```python
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

logger = logging.getLogger(__name__)

class BaseGrammarEngine(ABC):
    @abstractmethod
    def build_grammar(self, allowed_tables: List[str], allowed_columns: List[str]):
        """Xây dựng bộ ngữ pháp ràng buộc theo lược đồ rút gọn S'."""
        pass

    @abstractmethod
    def get_logits_processor(self, tokenizer):
        """Trả về logits processor để truyền vào mô hình sinh của HuggingFace."""
        pass

class XGrammarPDAEngine(BaseGrammarEngine):
    def __init__(self):
        import xgrammar as xg
        self.xg = xg
        self.compiled_grammar = None

    def build_grammar(self, allowed_tables: List[str], allowed_columns: List[str]):
        # Tạo BNF Grammar cho phép chỉ truy vấn trên allowed_tables và allowed_columns
        bnf_grammar = self._generate_sql_bnf(allowed_tables, allowed_columns)
        self.compiled_grammar = self.xg.Grammar(bnf_grammar)
        return self.compiled_grammar

    def get_logits_processor(self, tokenizer):
        # Trả về XGrammar LogitsProcessor
        from xgrammar import GrammarLogitsProcessor
        return GrammarLogitsProcessor(self.compiled_grammar, tokenizer)

class RegexASTFallbackEngine(BaseGrammarEngine):
    def __init__(self):
        logger.warning(
            "⚠️ [FALLBACK KÍCH HOẠT]: XGrammar không khả dụng trên môi trường hiện tại. "
            "Chuyển sang chế độ Native Decoding kết hợp thẩm định AST sau sinh."
        )
        self.allowed_tables = set()
        self.allowed_columns = set()

    def build_grammar(self, allowed_tables: List[str], allowed_columns: List[str]):
        self.allowed_tables = set(allowed_tables)
        self.allowed_columns = set(allowed_columns)
        return self

    def get_logits_processor(self, tokenizer):
        # Chế độ fallback không can thiệp logits token-by-token mà để mô hình sinh tự do,
        # sau đó lọc kết quả qua sqlglot ở bước hậu xử lý
        return None

def create_grammar_engine() -> BaseGrammarEngine:
    try:
        import xgrammar
        return XGrammarPDAEngine()
    except Exception as e:
        logger.info(f"Không thể tải XGrammar ({e}). Sử dụng Fallback Engine.")
        return RegexASTFallbackEngine()
```

### 4.4. Cách Thức Kiểm Thử Fallback An Toàn
Trong file `tests/test_phase4_evaluation.py`:
1. **Test Normal Path**: Nếu có `xgrammar`, test khởi tạo grammar với bảng/cột mini.
2. **Test Fallback Path**: Ép buộc `create_grammar_engine(force_fallback=True)` để kiểm tra engine fallback:
   - Đảm bảo `get_logits_processor()` trả về `None` an toàn.
   - Đảm bảo hàm hậu xử lý `validate_generated_sql()` của `RegexASTFallbackEngine` phát hiện được câu SQL chứa bảng nằm ngoài `allowed_tables`.
   - Khẳng định không phát sinh Exception làm dừng chương trình.

---

## 5. Chiến Lược Lập Tài Liệu Bàn Giao (`HANDOVER.md`)

Theo yêu cầu **R3** và **Acceptance Criteria**, file `HANDOVER.md` đặt tại thư mục gốc của dự án là tài liệu kỹ thuật tối quan trọng để chuyển giao toàn bộ mã nguồn từ **Đội ngũ Lập trình (Implementation Squad)** sang **Đội ngũ Nghiên cứu & Phân tích Số liệu (Research & Analytics Team)**.

Dưới đây là cấu trúc chi tiết, mạch lạc và tiêu chuẩn cần triển khai cho file `HANDOVER.md`:

```
HANDOVER.md
├── 1. TỔNG QUAN DỰ ÁN & MỤC TIÊU BÀN GIAO
│   ├── Bối cảnh khoa học: Thay thế MARS-SQL 21B bằng hệ thống SLM 2.5B–4.0B
│   ├── Khung kiến trúc 3 Tác tử: Grounding -> Generation -> Validation
│   └── Phạm vi chuyển giao kỹ thuật
│
├── 2. BẢN ĐỒ VAI TRÒ TÁC TỬ THEO AGENTS_RESEARCH.md
│   ├── [AGENT-DIRECTOR-LEADER]: Điều phối trung tâm, Task Contracts, DoD Gatekeeper
│   ├── [AGENT-DATA-G1]: Kỹ sư Data Pipeline & Schema Grounding (Đồ thị PK-FK, Cây Steiner)
│   ├── [AGENT-GEN-G2]: Kỹ sư SLM Training & Constrained Decoding (Completion Loss, JEPA, GRPO, XGrammar)
│   └── [AGENT-VAL-BENCH]: Kỹ sư System-1 Validation & Benchmark (PV-SQL 10 AST Rules, 0-Token Laya-421M)
│
├── 3. DANH MỤC TỆP TIN & MÃ NGUỒN (MASTER FILE INVENTORY)
│   ├── Bảng danh mục chi tiết từng file (Đường dẫn, Tác tử chịu trách nhiệm, Mục đích, I/O)
│   ├── Cấu trúc module src/ (data/, training/, agents/, evaluation/)
│   ├── Cấu trúc thư mục notebooks/ (01, 02, 03, 03b, 04)
│   ├── Cấu trúc thư mục tests/ và scripts/
│   └── Cấu trúc gói mẫu dữ liệu data/mini/
│
├── 4. HƯỚNG DẪN THIẾT LẬP MÔI TRƯỜNG (ENVIRONMENT SETUP)
│   ├── Môi trường Cục bộ (Local / CPU / WSL2): Tạo venv, cài requirements, chạy smoke tests
│   └── Môi trường Google Colab: Lựa chọn loại GPU (T4, L4, A100), cài đặt CUDA, kiểm tra XGrammar
│
├── 5. QUY TRÌNH CHUẨN BỊ DỮ LIỆU (DATA PREPARATION WORKFLOW)
│   ├── Kiểm thử nhanh với data/mini/ (< 1 phút)
│   ├── Quy trình tải dữ liệu thô (BIRD 33.4GB, Spider, SynSQL-2.5M) qua gdown/Google Drive
│   └── Chạy các script tiền xử lý Phase 1 tạo 4 gói dữ liệu sạch
│
├── 6. QUY TRÌNH HUẤN LUYỆN TRÊN GOOGLE COLAB (COLAB TRAINING WORKFLOW)
│   ├── Cơ chế mount và cấu trúc thư mục Google Drive (MyDrive/Text2SQL_Colab/)
│   ├── Hướng dẫn từng bước chạy 5 Colab Notebooks
│   ├── Cấu hình huấn luyện mô hình 3B (FINER-SQL-3B) vs 7B (Qwen2.5-Coder-7B)
│   └── Cơ chế đồng bộ Checkpoint tự động & Phục hồi khi bị ngắt kết nối (Session Recovery)
│
├── 7. QUY TRÌNH ĐÁNH GIÁ, ĐO ĐẠC & TÁI LẬP KẾT QUẢ (BENCHMARK REPRODUCTION)
│   ├── Sinh 8–16 quỹ đạo ứng viên (Inference Rollouts)
│   ├── Chạy kiểm định PV-SQL 10 AST Rules và chấm điểm 0-token
│   ├── Tái lập số liệu BIRD-dev, Spider, Spider-DK (EX, VES, ECE)
│   ├── Kịch bản tái lập bài test 48 câu cứu bảng cầu nối (100% EX)
│   └── Xuất bảng số liệu Markdown và đồ thị khoa học cho bài báo
│
├── 8. HỆ THỐNG KIỂM THỬ TỰ ĐỘNG & NGHIỆM THU (SMOKE TESTING)
│   ├── Lệnh chạy kiểm thử nhanh: python scripts/run_smoke_tests.py
│   ├── Bảng đối chiếu tiêu chí nghiệm thu C1, C2, C3
│   └── Hướng dẫn xử lý sự cố thường gặp (Troubleshooting Guide)
│
└── 9. LIÊN HỆ & THÔNG TIN BÀN GIAO
    └── Ngày hoàn tất bàn giao, trạng thái kiểm định và chữ ký nghiệm thu
```

### 5.1. Bảng Mẫu Danh Mục Tệp Tin (File Inventory Template)
Trong Phần 3 của `HANDOVER.md`, danh mục file sẽ được trình bày dưới dạng bảng rõ ràng:

| Đường Dẫn Tệp Tin | Vai Trò Tác Tử | Mô Tả Chức Năng | Đầu Vào (Input) | Đầu Ra (Output) |
| :--- | :--- | :--- | :--- | :--- |
| `src/data/schema_graph.py` | `Data-G1` | Dựng đồ thị PK-FK và giải thuật Cây Steiner cứu bảng cầu nối | SQLite DDL, Terminal Columns | Sub-schema $S'$ rút gọn |
| `src/data/embed_packer.py` | `Data-G1` | Đóng gói thẻ cột 6 trường và tạo tập mẫu âm khó HN-SupCon | Lược đồ DB, Gold SQL | `DATA-G1-EMB.jsonl` |
| `src/data/synsql_cleaner.py`| `Data-G1` | Làm sạch SynSQL-2.5M thành 916K mẫu bọc thẻ `<think>` | SynSQL raw JSON | `DATA-G2-SFT.jsonl` |
| `scripts/make_mini_dataset.py`| `Data-G1` | Tạo bộ mẫu thử nghiệm thu nhỏ trong `data/mini/` | Raw data hoặc mock data | Thư mục `data/mini/` |
| `src/training/train_embedding.py`| `Gen-G2` | Huấn luyện bộ nhúng Qwen3-Embedding-0.6B (HN-SupCon) | `DATA-G1-EMB.jsonl` | Vector weights, `index.npy` |
| `src/training/train_sft_generator.py`| `Gen-G2` | Huấn luyện SFT Completion-Only trên mô hình 3B / 7B | `DATA-G2-SFT.jsonl` | LoRA / Full Checkpoints |
| `src/training/jepa_module.py`| `Gen-G2` | Cài đặt 2-block causal mask và hàm mục tiêu JEPA | Text & Code hidden states | JEPA alignment loss |
| `src/training/train_grpo_generator.py`| `Gen-G2` | Huấn luyện tăng cường GRPO với 4 đầu thưởng dày | FINER-SQL-3B, SQLite DBs | Model checkpoint tối ưu |
| `src/training/drive_callback.py`| `Gen-G2` | Callback tự động sao lưu checkpoint lên Google Drive | HuggingFace Trainer state | Sync vào Google Drive |
| `notebooks/01_colab_data_prep.ipynb`| `Gen-G2` / `Data-G1` | Notebook chuẩn bị dữ liệu Phase 1 trên Colab | Google Drive raw data | Dữ liệu sạch trên Drive |
| `notebooks/03_colab_train_generator_3b.ipynb`| `Gen-G2` | Notebook huấn luyện FINER-SQL-3B trên Colab GPU | SFT data, Drive path | Best 3B Checkpoint |
| `notebooks/03b_colab_train_generator_7b.ipynb`| `Gen-G2` | Notebook huấn luyện Qwen2.5-Coder-7B trên A100 | SFT data, Drive path | Best 7B Checkpoint |
| `src/agents/generation/grammar_engine.py`| `Gen-G2` | Bộ khóa XGrammar PDA kèm fallback an toàn | Sub-schema $S'$, Tokenizer | LogitsProcessor / Validator |
| `src/evaluation/pv_sql_rules.py`| `Val-Bench` | Kiểm tra 10 quy tắc cú pháp AST tất định bằng `sqlglot` | Câu hỏi tự nhiên, SQL query | Báo cáo vi phạm AST (0-10) |
| `src/evaluation/system1_validator.py`| `Val-Bench` | Chấm điểm song song 8–16 ứng viên 0-token (Laya-421M) | Candidates SQL, AST report | Xếp hạng & Best SQL |
| `src/evaluation/metrics.py`| `Val-Bench` | Tính toán các chỉ số học thuật: EX, VES, TPR, ECE | Predictions, Gold SQL, DBs | Bảng kết quả tổng hợp |
| `src/evaluation/evaluate_end_to_end.py`| `Val-Bench` | Kịch bản đánh giá toàn tuyến phối hợp cả 3 tác tử | Bộ dữ liệu BIRD/Spider | `evaluation_results.jsonl` |
| `scripts/run_smoke_tests.py`| `Director-Leader`| Script kiểm thử nhanh toàn bộ hệ thống từ Phase 1–4 | `data/mini/` | Báo cáo kiểm định console |

---

## 6. Khuyến Nghị Cho Nhóm Triển Khai (Actionable Recommendations)

1. **Khởi tạo Virtualenv chuẩn trong hướng dẫn**:
   Quy định rõ lệnh khởi tạo môi trường Python 3.10–3.12 sử dụng virtualenv để tránh lỗi PEP 668.
2. **Triển khai `data/mini/` độc lập ngay từ đầu**:
   Tạo sẵn thư mục `data/mini/` với 2 database SQLite thực tế kích thước nhỏ (`financial_mini.sqlite`, `california_schools.sqlite`) cùng các file JSONL mẫu. Điều này cho phép chạy test mọi lúc mọi nơi mà không phụ thuộc vào việc tải dataset BIRD 33GB.
3. **Mã nguồn phải bọc Try/Except đối với `xgrammar`**:
   Sử dụng đúng cấu trúc `GrammarEngine` với fallback sang `RegexASTFallbackEngine` đã trình bày tại Mục 4, giúp bài test chạy mượt mà ngay cả khi môi trường cục bộ chưa cài đặt được XGrammar GPU binary.
4. **Viết test cases đa tương thích (`unittest` + `pytest`)**:
   Xây dựng toàn bộ các test trong `tests/` kế thừa từ `unittest.TestCase` để người dùng có thể chạy trực tiếp bằng `python3 -m unittest discover tests` mà không cần cài đặt thêm bất kỳ test runner nào.
5. **Đảm bảo tính trọn vẹn của `HANDOVER.md`**:
   Sau khi hoàn tất cài đặt mã nguồn, file `HANDOVER.md` phải được điền đầy đủ mọi thông tin theo cấu trúc đã đề ra ở Mục 5, hỗ trợ đội ngũ nghiên cứu vận hành và tái lập kết quả một cách độc lập.
