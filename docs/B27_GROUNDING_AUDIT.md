# B2.7 — SQL Grounding labels và chuẩn bị BIRD Train

## Phạm vi và bằng chứng

Không chạy full training, upload checkpoint, sửa gold SQL hoặc đổi checkpoint format. Qwen3-Embedding-0.6B, HN-SupCon, optimizer/RNG/resume và HF backend giữ nguyên. Pilot Kaggle step 2 → step 3 là bằng chứng do người dùng cung cấp, không phải lần chạy lại trong workspace này.

Full BIRD 9.428 records/69 DB không có trong repository local. Số liệu người dùng cung cấp là **baseline code cũ**: labels/parser PASS 9.340, FAIL 88; SQLite PASS 9.078, FAIL 350; cả hai PASS 9.042; 12 duplicate groups/12 records dư/7 conflicting groups và 9.030 unique groups trong tập 9.042. Chưa xác nhận lại các số này sau B2.7. Không kết luận đã phục hồi 36 câu. Tests chỉ chứng minh fixtures A/B/C và các dạng SQL được liệt kê bên dưới.

## Nguyên nhân và resolver

Resolver cũ duyệt `scope.columns`, có thể bao gồm correlated/external columns của scope con, rồi tra theo aliases scope ngoài. Đây là nguồn ambiguous giả ở nested SELECT. Nó bỏ qua source là CTE/derived scope, bỏ qua qualified columns không khớp và lọc bỏ explicit annotations không tồn tại mà không báo lỗi. Split guard cũ chỉ so chuỗi question nguyên bản.

`src/training/sql_grounding.py` dùng SQLGlot pinned 26.16.4:

1. Lấy schema bằng SQLite read-only, dùng đủ schema để tránh cột thiếu card làm sai unique matching.
2. Xác nhận physical tables thuộc DB. SQLGlot qualify trên **bản copy AST**, tắt schema inference/partial qualification; source SQL không đổi.
3. Qualify unqualified columns theo selected sources của từng scope. CTE/derived outputs được kiểm tra, không biến alias derived thành bảng thật.
4. Duyệt `scope.walk()` chỉ trong scope hiện tại. Qualified correlated references chỉ tìm parent khi scope cho phép correlation; local alias luôn ưu tiên.
5. Thu thập physical `table.column` và kiểm tên thật. ORDER BY output aliases không tự thành cột mới. Nếu bất kỳ tham chiếu nào không chắc chắn, reject cả label record để review; không chấp nhận một tập positives thiếu một phần rồi dùng phần còn lại làm negatives.
6. Expand `SELECT *`/`table.*` theo schema. `COUNT(*)` không expand thành mọi cột. Query chỉ COUNT(*) hoặc literal không có physical column evidence → `no_positive_columns`, manual review.
7. Annotation explicit phải tồn tại toàn bộ trong cards của đúng DB; strict mode còn yêu cầu khớp chính xác tập SQL evidence. Missing positive cards/stale cards/duplicate card identities bị chặn.

Ví dụ Disney: `'voice-actor'` vẫn là literal; `COUNT(movie)` cung cấp positive `voice-actors.movie`. Không sửa dấu quote hoặc dự đoán `voice-actors.voice-actor`.

Các trường hợp còn review: NATURAL JOIN implicit keys (chưa suy diễn), table-valued/unsupported sources, schema-qualified attached DB, missing cards, implicit rowid không có card, recursive/SQL dialect constructs mà qualifier không chứng minh được, COUNT(*)-only, và SQL compile lỗi. SQLite build thiếu custom function/collation/extension cũng bị quarantine; không cài hoặc chạy extension để né lỗi.

## API và tương thích pilot

```python
build_embedding_examples(cards, records, db_dir, label_mode='scope_v2')  # public default mới
build_embedding_examples(cards, records, db_dir, label_mode='strict')   # compile bắt buộc
build_embedding_examples(cards, records, db_dir, label_mode='legacy')   # implementation cũ nguyên trạng
```

`run_embedding_training()` **vẫn mặc định legacy**, không thêm default argument vào checkpoint sync fingerprint. Chọn mới bằng `grounding_label_mode='strict'` qua kwargs; contract strict có thêm label mode, từ chối lẫn với checkpoint pilot cũ. Format/state không đổi. Không chuyển pilot đang resume sang data/split/resolver mới. Muốn train dataset mới phải dùng output và experiment ID mới; không xóa checkpoint cũ.

Strict yêu cầu cards gắn `db_id`; không đoán membership từ table name giống nhau giữa DB. Untagged mini/pilot cũ vẫn dùng legacy. `scope_v2` chỉ cho phép untagged fallback có schema thật và input một DB; multi-DB cần cards có tag. Trong API không có DB, cards là schema đầu vào; full training phải dùng strict với SQLite thật.

## Audit read-only

CLI độc lập không cần GPU hoặc HF Secret và không khởi tạo checkpoint backend:

```bash
python scripts/audit_grounding_labels.py \
  --data /path/to/train.json \
  --db-root /path/to/train_databases \
  --cards /path/to/full_column_cards.jsonl \
  --output /kaggle/working/b27-audit-new \
  --split --seed 42 --validation-fraction 0.1
```

Input nhận JSON array hoặc JSONL qua `load_records`. DB layout: `<root>/<db_id>.sqlite` hoặc `<root>/<db_id>/<db_id>.sqlite`. SQLGlot version phải khớp `--sqlglot-version` (default đúng pin); không đổi version để so baseline thiếu kiểm soát.

SQLite dùng URI `mode=ro`, connection `query_only`, `trusted_schema=OFF`, deny-write authorizer và chỉ `EXPLAIN QUERY PLAN` cho SELECT/WITH. Không chạy query lấy result rows, không sửa DB/SQL. Schema đọc từ sqlite_master/PRAGMA. Error groups: missing_table, missing_column, ambiguous_column, syntax_error, other_sqlite_error; thiếu DB/unsupported statement tách riêng. Parser-valid nhưng SQLite-invalid luôn nằm trong quarantine, không vào strict-valid.

Output directory phải mới. Không overwrite output cũ. Exit 0: audit xong; không tự nghĩa là mọi record PASS. Exit 2: có yêu cầu split nhưng split không thể tạo (ví dụ chỉ còn một non-conflicting group); audit/report vẫn được giữ. Runtime/input/version errors có exit khác 0. `audit_summary.json` được ghi cuối; không có file này nghĩa là job chưa hoàn tất.

| File | Nội dung |
|---|---|
| `audit_summary.json` | Totals, compile/label PASS/FAIL, errors, DB breakdown, versions, input/cards/DB/source hashes, recovery definition, duplicate counts, split status |
| `valid_candidates.jsonl` | SQLite PASS + labels chắc chắn + có cả positives/negatives; conflicting groups vẫn được đánh dấu để review |
| `recoverable_candidates.jsonl` | Tập trên nhưng legacy builder hiện được giữ lại FAIL; không phải tuyên bố phục hồi 36 câu baseline |
| `quarantined_records.jsonl` | SQLite FAIL, bất kể label resolver PASS/FAIL |
| `manual_review.jsonl` | Label unresolved, no negatives, stale/missing cards, hoặc conflicting SQL duplicate groups; có thể giao với quarantine |
| `duplicate_groups.jsonl` | Nhóm duplicate toàn input, source indices và toàn bộ SQL variants, không chọn gold |
| `split_manifest.json` | Khi split PASS: normalization/seed/fingerprints, group assignment và indices, excluded conflicting IDs |
| `train.jsonl`, `internal_validation.jsonl` | Split đã loại toàn bộ conflicting groups; tất cả member của mỗi non-conflicting group cùng một split |

`source_index` là **zero-based vị trí trong input gốc**, không phải question_id hoặc index sau lọc. `source_record` giữ nguyên record gốc (kể cả nếu input đã có field source_index). Record xuất còn giữ db_id/question/SQL/evidence và positive_columns xác minh được, nên runner load trực tiếp. Fingerprints ràng buộc indices với input/cards cụ thể.

`duplicate_groups`/`unique_question_groups` mặc định tính toàn input. Để so với 12/9.030 baseline trong tập strict-valid, dùng các field `valid_candidate_duplicate_*`, `valid_candidate_conflicting_sql_groups`, `valid_candidate_unique_question_groups`. SQLite-invalid duplicate có gold khác vẫn khiến cả nhóm bị loại khỏi split; không ngầm chọn gold của member còn valid.

Recovery mới = legacy builder FAIL + scope_v2 usable labels PASS + SQLite PASS. `legacy_label_sqlite_matrix` cho so chéo code cũ trên cùng inputs/runtime. Nếu muốn đối chiếu đúng 36 records cũ, phải join source_index + db_id từ log baseline và xác nhận fingerprint dataset; summary tổng không đủ chứng minh từng câu đã phục hồi.

## Duplicate guard và internal validation

Normalization v1 cố định trong code/test: Unicode NFKC → casefold → collapse/strip Unicode whitespace; **giữ punctuation**. Key `(db_id, normalized_question)`. SQL variants so exact SQL sau strip ở hai đầu, không normalize literal/nội dung SQL. Đây là nhận diện conflict bảo thủ, có thể đưa hai SQL tương đương vào review.

Split sort group IDs bằng SHA-256(seed, group_id), chọn khoảng validation_fraction số nhóm, giữ ít nhất một nhóm mỗi bên. Không dùng Python hash ngẫu nhiên. Conflict groups loại toàn bộ khỏi split, giữ nguyên ở reports. Không tùy tiện deduplicate/chọn một gold. Non-conflicting duplicates giữ mọi record trong cùng split; nếu muốn giảm trọng số duplicate cần quyết định nghiên cứu riêng. Validation là **internal split từ BIRD Train, không phải official BIRD dev/test**. Runner còn kiểm normalized question overlap giữa train và eval trước load model.

## Notebook 01

Mặc định mini/pilot không đổi. Để chỉ chuẩn bị full dataset:

```python
os.environ.update({
    'GROUNDING_FULL_DATASET': '1',
    'GROUNDING_ALLOW_FULL_TRAINING': '0',
    'TRAIN_DATA': '/kaggle/input/YOUR_BIRD/train.json',
    'DB_DIR': '/kaggle/input/YOUR_BIRD/train_databases',
    'CARDS_PATH': '/kaggle/working/full_column_cards.jsonl',  # có db_id cho mọi card
    'GROUNDING_AUDIT_DIR': '/kaggle/working/b27-audit-new',
    'KAGGLE_OUTPUT_DIR': '/kaggle/working/b27-preparation-new',
    'CHECKPOINT_RESUME_SOURCE': 'none',
})
```

Full preparation không cần retrieval_dev.jsonl bên ngoài: tạo internal split. Không khởi tạo private checkpoint writer khi chỉ prepare, không gọi training runner. Notebook vẫn có environment/GPU preflight và có thể trích card sample values theo luồng cũ nếu không cung cấp CARDS_PATH; muốn audit thuần compile/schema, dùng CLI trên cards đã có (không trích sample). Dùng output mới, không ghi đè cards/graphs cũ. Chỉ review xong mới chủ động bật training; B2.7 không bật cờ này.

Khi cần resume strict experiment đã tạo sau review, không chạy audit lại/đổi split: đặt `GROUNDING_FULL_DATASET=0`, `GROUNDING_LABEL_MODE=strict`, TRAIN_DATA/RETRIEVAL_EVAL_DATA/CARDS_PATH trỏ đúng files đã chuẩn bị, output/experiment/seed/config đúng checkpoint strict. Pilot cũ giữ label mode legacy và files/IDs cũ. Đừng dùng checkpoint pilot để khởi động strict dataset với giả định đó là resume tương thích.

## Cập nhật repository Dataset và cell Kaggle

1. Đóng gói snapshot working tree mới, gồm src/scripts/tests/docs và Notebook 01; không đóng gói `.git`, tokens, caches, checkpoints hoặc research artifacts. Không cần commit/push.
2. Tạo version mới cho Kaggle Dataset chứa source, attach version đó vào notebook. Giữ BIRD input gốc và checkpoint Dataset cũ nguyên trạng.
3. Restart kernel, đặt KAGGLE_REPO_DIR đúng thư mục có src/, xác nhận file `src/training/sql_grounding.py` mới tồn tại. Không dùng module đã import từ snapshot cũ.
4. Dùng SQLGlot pin 26.16.4 đã bootstrap; chạy audit cell sau. Không cần Hugging Face token hoặc bật GPU cho CLI.

```python
import json, os, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone
REPO_DIR = Path(os.environ['KAGGLE_REPO_DIR'])
DATA = Path('/kaggle/input/YOUR_BIRD/train.json')
DB_ROOT = Path('/kaggle/input/YOUR_BIRD/train_databases')
CARDS = Path('/kaggle/working/full_column_cards.jsonl')
OUT = Path('/kaggle/working') / ('b27-audit-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
assert (REPO_DIR/'scripts/audit_grounding_labels.py').is_file()
assert DATA.is_file() and DB_ROOT.is_dir() and CARDS.is_file()
subprocess.run([sys.executable, '-B', str(REPO_DIR/'scripts/audit_grounding_labels.py'),
    '--data', str(DATA), '--db-root', str(DB_ROOT), '--cards', str(CARDS),
    '--output', str(OUT), '--split', '--seed', '42', '--validation-fraction', '0.1'],
    cwd=REPO_DIR, check=True)
summary = json.loads((OUT/'audit_summary.json').read_text())
print(json.dumps(summary, indent=2, ensure_ascii=False))
# Số dưới đây do người dùng cung cấp, chỉ so nếu cùng raw BIRD Train.
baseline = {'total_records': 9428, 'sqlite_pass': 9078, 'sqlite_fail': 350,
            'valid_candidates': 9042, 'valid_candidate_duplicate_groups': 12,
            'valid_candidate_conflicting_sql_groups': 7,
            'valid_candidate_unique_question_groups': 9030}
print('DIFF vs reported old baseline:',
      {k: {'old': v, 'new': summary[k], 'delta': summary[k]-v} for k,v in baseline.items()})
assert summary['split_status'] == 'PASS'
train = [json.loads(x) for x in (OUT/'train.jsonl').read_text().splitlines()]
val = [json.loads(x) for x in (OUT/'internal_validation.jsonl').read_text().splitlines()]
sys.path.insert(0,str(REPO_DIR))
from src.training.grounding_audit import question_key
assert not {question_key(r) for r in train} & {question_key(r) for r in val}
assert all(r['_grounding_audit']['sqlite']['status']=='PASS' and
           r['_grounding_audit']['grounding']['status']=='PASS' and
           not r['_grounding_audit'].get('conflicting_sql',False) for r in train+val)
print('Prepared artifacts:', OUT, 'NO TRAINING EXECUTED')
```

## PASS/FAIL trước full training

PASS khi report hoàn chỉnh và fingerprints/SQLGlot/input đúng; audit chạy đủ input và DB dự kiến; không có SQLite-invalid/unresolved/annotation mismatch trong train/validation; zero normalized group overlap; zero conflicting group trong split; positive labels truy về đúng cards/DB; regression/pilot tests pass; recovered records đối chiếu được với baseline indices. Quarantine/manual có thể còn records: phải xác nhận chúng đã bị loại hoặc được review rõ ràng, không ép mọi gold thành valid.

FAIL nếu job không có summary, version/input lệch, thiếu DB/cards, split FAIL, còn overlap/conflicting gold/SQLite-invalid trong split, hoặc regression fail. Số valid/recovered tăng không đủ làm tiêu chí duy nhất. Chưa chạy full BIRD sau sửa thì trạng thái full-data acceptance vẫn **NOT EXECUTED**. GPU/memory và model training là acceptance khác, không được suy luận từ audit labels.

## Kết quả local của lượt B2.7

Native CPU full suite: **295 passed**, 0 skipped/failures/errors (7.562s). System Python: **237 passed, 58 dependency skips**, 0 failures/errors (1.360s). Có 34 tests B2.7 mới. Smoke 6/6 passed (0.952s); 59 Python files và 5 notebooks syntax/schema passed; git diff --check passed. AST legacy label builder và HN-SupCon không đổi; checkpoint runtime/backend và requirements pins không đổi.

CLI trên mini dataset thật 24 records/2 DB, với tagged cards tạo từ schema ở thư mục /tmp: compile/labels/valid **24**, recovered **0**, quarantine/manual **0**, split nội bộ **21/3**, load lại strict thành công. Tests A/B chứng minh phục hồi hai fixtures so với legacy, C chứng minh literal không thành identifier; không ngoại suy thành 36 records full BIRD. Full BIRD audit và acceptance full training vẫn **NOT EXECUTED**.
