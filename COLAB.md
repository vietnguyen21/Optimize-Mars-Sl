# Chạy trên Google Colab

Mọi thành viên chạy các stage bằng cùng một script, `scripts/colab/run_stage.py`, thông qua
`notebooks/colab/launcher.ipynb`. Không chạy bằng cell tự viết.

## Thư mục trên Drive

```
MyDrive/Text2SQL_Colab/
  data/bird/dev.zip, train.zip     BIRD gốc (tải một lần)
  data/*.jsonl, manifest.json      do stage prepare tạo
  runs/<RUN_ID>/                   checkpoint, notebook đã chạy, launch-*.json của từng lần chạy
```

Database SQLite (khoảng 32 GB) được giải nén vào `/content/bird_db` ở đĩa local mỗi phiên, không để trên Drive.

## Các bước

1. Mở `notebooks/colab/launcher.ipynb` trên Colab, chọn GPU.
2. Chạy cell mount Drive và clone code.
3. Lần đầu trên một Drive: chạy cell `prepare`.
4. Đặt `STAGE`, `RUN_ID`, chạy cell khởi động. Xem tiến độ bằng cell trạng thái.
5. Mất runtime: mở lại, chạy lại bước 2 và 4 với cùng `RUN_ID`. Script tự resume từ checkpoint mới nhất.

| STAGE | Notebook | Ghi chú |
|---|---|---|
| `grounding` | 01 | Chạy được trên Python 3.13 hiện tại của Colab |
| `sft` | 02 | Cần Python 3.10–3.12 và CUDA 12 (xem bên dưới). `--set LAMBDA_JEPA=0` cho bản đối chứng không JEPA |
| `grpo` | 03 | Cần `--set SFT_CHECKPOINT=<đường dẫn checkpoint SFT trên Drive>` |
| `system1` | 04 | Cần dữ liệu `decision_*.jsonl` sinh từ generator đã train |
| `evaluation` | 05 | Cần các checkpoint đã chấp nhận |

## Quyết định đã chốt

- Generator mặc định: `Qwen/Qwen2.5-Coder-3B-Instruct` (không gated). Đổi bằng `--set GENERATOR_MODEL=...`.
- Dữ liệu: tách theo `db_id`, seed 42. Train 62 DB; cal-A (4 DB) cho đánh giá retrieval và hiệu chuẩn System-1;
  cal-B (3 DB) cho test System-1; BIRD dev chỉ dùng cho Notebook 05. Chưa áp danh sách lọc 8.036 câu của MARS-SQL.
- EX theo quy tắc của BIRD: `set(pred) == set(gold)`, timeout 30 giây mỗi truy vấn.
- Checkpoint lưu trên Drive; HF Hub sync tắt mặc định (`CHECKPOINT_SYNC_ENABLED=0`).

## Giới hạn hiện tại

- Colab mặc định dùng Python 3.13 và torch `cu130`. Notebook 02–05 bị chặn có chủ đích ở
  `src/training/environment.py`, vì `bitsandbytes` (NF4) và các bản pin chưa được kiểm chứng trên cấu hình này.
  Cần runtime Colab Python 3.12 / CUDA 12 trước khi chạy 02–05. Không nới điều kiện chặn khi chưa có pilot GPU qua kiểm tra NF4.
- Early exit ở Notebook 05 vẫn sinh đủ K candidate trước khi dừng, nên chưa được báo cáo là giảm latency.
