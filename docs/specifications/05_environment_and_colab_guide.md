# HƯỚNG DẪN MÔI TRƯỜNG & HUẤN LUYỆN TRÊN GOOGLE COLAB (ENVIRONMENT & COLAB GUIDE)

## 1. Yêu Cầu Phần Cứng & Môi Trường Google Colab

Dự án được tối ưu hóa đặc biệt để có thể huấn luyện và đánh giá trên các gói dịch vụ điện toán đám mây **Google Colab** phổ biến:

| Phân Loại Colab | Loại GPU | VRAM Khả Dụng | Tương Thích Mô Hình | Ghi Chú Sử Dụng |
| :--- | :--- | :---: | :--- | :--- |
| **Colab Miễn Phí (Free)** | NVIDIA Tesla T4 | ~15 GB | `Qwen3-Embedding-0.6B`, `Laya-421M`, `FINER-SQL-3B` (QLoRA 4-bit) | Thích hợp chạy Phase 1, Phase 2 (Smoke test) và suy luận đánh giá. |
| **Colab Pro (Tiêu chuẩn)** | NVIDIA L4 / V100 | ~22.5 – 16 GB | `FINER-SQL-3B` (LoRA BF16/FP16), `Combo C` toàn tuyến | Môi trường chuẩn khuyến nghị để hoàn thành toàn bộ Phase 2 & 3. |
| **Colab Pro+ (Cao cấp)** | NVIDIA A100-SXM4 | ~40 GB | `Qwen2.5-Coder-7B` (Full SFT / GRPO), `Combo D` | Dành cho việc huấn luyện mô hình 7B nhằm xô đổ kỷ lục 21B của MARS-SQL. |

---

## 2. Quy Trình Đồng Bộ & Bảo Lưu Trọng Số Với Google Drive

Trên Google Colab, toàn bộ ổ cứng tạm tại thư mục `/content/` sẽ bị xóa sạch nếu phiên làm việc kết thúc hoặc bị ngắt kết nối quá thời gian chờ. Do đó, hệ thống áp dụng cơ chế **đồng bộ tự động 2 chiều với Google Drive**:

```
[ Google Colab Runtime ]                                [ Google Drive (Vĩnh Viễn) ]
/content/optimize/                                      MyDrive/Text2SQL_Colab/
├── data/mini/                   ◀── [Sync Khi Khởi Động] ──├── data/
├── checkpoints/ (tốc độ cao SSD) ──► [Auto-Sync Checkpoint] ──►└── checkpoints/
│   ├── step_1000/                                             ├── step_1000/
│   └── best_model/                                            └── best_model/
```

### 2.1. Lệnh Mount Google Drive Chuẩn
Trong mọi notebook Colab, bước đầu tiên luôn là kết nối với Drive:
```python
from google.colab import drive
import os

drive.mount('/content/drive')

DRIVE_DIR = "/content/drive/MyDrive/Text2SQL_Colab"
os.makedirs(f"{DRIVE_DIR}/checkpoints", exist_ok=True)
os.makedirs(f"{DRIVE_DIR}/logs", exist_ok=True)
print(f"✅ Đã kết nối Google Drive thành công tại: {DRIVE_DIR}")
```

### 2.2. Callback Tự Động Lưu Checkpoint An Toàn (DriveSyncCallback)
Mã nguồn huấn luyện trong `src/training/` được tích hợp sẵn class `DriveCheckpointCallback`:
```python
from transformers import TrainerCallback
import shutil
import os

class DriveCheckpointCallback(TrainerCallback):
    def __init__(self, drive_checkpoint_dir):
        self.drive_dir = drive_checkpoint_dir

    def on_save(self, args, state, control, **kwargs):
        checkpoint_folder = f"checkpoint-{state.global_step}"
        local_path = os.path.join(args.output_dir, checkpoint_folder)
        drive_path = os.path.join(self.drive_dir, checkpoint_folder)
        
        if os.path.exists(local_path):
            print(f"\n[DriveSync] Đang sao lưu {checkpoint_folder} sang Google Drive...")
            shutil.copytree(local_path, drive_path, dirs_exist_ok=True)
            print(f"[DriveSync] ✅ Đã lưu an toàn vào Google Drive: {drive_path}")
```

---

## 3. Khởi Tạo Môi Trường Trên Colab (Setup Commands)

Mỗi notebook trong thư mục `notebooks/` đều được chuẩn bị sẵn khối lệnh cài đặt môi trường chỉ với 1 thao tác bấm:

```bash
# 1. Kích hoạt môi trường GPU CUDA 12.x
!nvidia-smi

# 2. Clone mã nguồn dự án vào thư mục tạm Colab
!git clone https://github.com/your-username/optimize.git /content/optimize
%cd /content/optimize

# 3. Cài đặt các gói phụ thuộc tương thích cao
!pip install --upgrade pip
!pip install -r docs/specifications/requirements.txt

# 4. Kiểm tra khả năng tương thích của XGrammar
python -c "
try:
    import xgrammar
    print('✅ XGrammar GPU Engine hoạt động xuất sắc!')
except Exception as e:
    print('⚠️ XGrammar cần fallback sang Native Decoding:', e)
"
```

---

## 4. Xử Lý Gián Đoạn & Phục Hồi Huấn Luyện (Session Recovery)

Nếu phiên Colab bị ngắt giữa chừng do mất kết nối mạng hoặc hết thời gian cấp GPU:
1. Mở lại notebook và chọn lại Runtime GPU.
2. Chạy lại ô kết nối Google Drive: `drive.mount('/content/drive')`.
3. Script huấn luyện tự động phát hiện checkpoint gần nhất được lưu trong Drive thông qua cờ `--resume_from_checkpoint`:
   ```bash
   python src/training/train_sft_generator.py \
       --model_type 3b \
       --data_path /content/drive/MyDrive/Text2SQL_Colab/data/DATA-G2-SFT.jsonl \
       --output_dir /content/checkpoints/finer_sql_3b \
       --drive_sync_dir /content/drive/MyDrive/Text2SQL_Colab/checkpoints/finer_sql_3b \
       --resume_from_checkpoint latest
   ```
4. Quá trình huấn luyện sẽ tiếp tục ngay từ bước lặp (step) trước đó mà không phải train lại từ đầu.
