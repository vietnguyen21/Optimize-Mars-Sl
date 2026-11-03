# Forensic Audit Report — Milestone M1 (Data Pipeline & Schema Grounding)

**Auditor**: `auditor_m1` (`[teamwork_auditor]`)  
**Target Milestone**: M1 (Phase 1: Data Pipeline & Schema Grounding)  
**Evaluated Artifacts**: `src/grounding/`, `src/common/`, `data/mini/`, `scripts/make_mini_dataset.py`, `tests/test_m1_stress.py`  
**Integrity Mode**: `development` (per `ORIGINAL_REQUEST.md`, also audited against `demo` and `benchmark` standards)  
**Overall Verdict**: **CLEAN** (Integrity Mandate Fully Upheld)

---

## 1. Observation (Empirical Findings & Verbatim Traces)

### A. Static Code Analysis of Core Modules

1. **`src/grounding/steiner_tree.py`**:
   - **Genuineness Check**: Implements an authentic Disjoint Set Union (`DisjointSet`), Kou-Markowsky-Berman (KMB) metric closure, Kruskal's Minimum Spanning Tree algorithm, iterative non-terminal leaf pruning, and foreign key edge extraction.
   - **Hardcoding Check**: Searched for references to sample tables (`clients`, `cards`, `accounts`, `schools`, `frpm`, `satscores`, `districts`, `transactions`).
     * `grep -i "clients" src/grounding/steiner_tree.py` -> **0 occurrences**.
     * `grep -i "cards" src/grounding/steiner_tree.py` -> **0 occurrences**.
     * `grep -i "accounts" src/grounding/steiner_tree.py` -> **0 occurrences**.
     * The algorithm contains zero hardcoded table names or shortcut conditions.
   - **Facade Check**: No `NotImplementedError`, no dummy constant returns (`return True`, `return ["accounts"]`). All methods perform genuine graph traversal and pruning.

2. **`src/grounding/graph_builder.py`**:
   - **Genuineness Check**: Connects directly to SQLite via `sqlite3.connect(..., uri=True)`. Queries `sqlite_master` for user tables (`SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'`).
   - Executes `PRAGMA table_info("{tbl}")` to extract column definitions, types, and primary keys.
   - Executes `PRAGMA foreign_key_list("{tbl}")` to extract explicit foreign key relations.
   - Implements `PurePythonGraph` with Dijkstra's shortest path algorithm (`heapq` priority queue) and breadth-first connected components extraction as an offline fallback when `networkx` is not present.
   - Implements `_infer_implicit_foreign_keys` for legacy schemas lacking explicit FK declarations.

3. **`src/grounding/column_card.py`**:
   - **Genuineness Check**: Connects to SQLite and dynamically queries distinct sample values from real tables:
     `SELECT DISTINCT "{col_name}" FROM "{table_name}" WHERE "{col_name}" IS NOT NULL LIMIT 3;`
   - Formats complete 6-field `ColumnCard` objects (`table_name`, `column_name`, `data_type`, `constraints`, `column_description`, `sample_values`).
   - Supports serialization and deserialization via JSONL (`save_cards_to_jsonl`, `load_cards_from_jsonl`).

4. **`src/grounding/embedding_retriever.py`**:
   - **Genuineness Check**: Dual-mode schema retrieval engine.
     * Dense mode: Cosine similarity over normalized embeddings using `SentenceTransformer` and `np.dot`.
     * Offline mode: Fully implemented BM25 scoring with Robertson-Spärck Jones IDF ($IDF = \ln(1 + \frac{N - DF + 0.5}{DF + 0.5})$) with $k_1 = 1.5, b = 0.75$, weighted field tokenization (`column_name`: 4.0, `table_name`: 3.0, `column_description`: 2.5, `sample_values`: 2.0, `constraints`: 1.0, `data_type`: 0.5), exact phrase matching bonuses, and PK/FK priority boosts.
   - Not a stub; tested against unseen adversarial queries with correct BM25 score ranking.

### B. Runtime Integrity Tracing (`data/mini/`)

1. **Binary SQLite Inspection**:
   - `data/mini/financial_mini.sqlite`:
     * File header: `b'SQLite format 3\x00'` (Authentic SQLite 3 binary).
     * Tables: 5 (`districts`: 4 rows, `clients`: 8 rows, `accounts`: 8 rows, `cards`: 6 rows, `transactions`: 12 rows).
     * Total rows: 38 rows. Foreign key constraints validated.
   - `data/mini/california_schools.sqlite`:
     * File header: `b'SQLite format 3\x00'` (Authentic SQLite 3 binary).
     * Tables: 3 (`schools`: 10 rows, `frpm`: 10 rows, `satscores`: 10 rows).
     * Indexes: 3 auto-indexes created for primary keys.
     * Total rows: 30 rows. Foreign key constraints validated.

2. **Mini BIRD Benchmark Queries (`data/mini/mini_bird.jsonl`)**:
   - Exactly 24 queries (12 for `financial_mini`, 12 for `california_schools`).
   - Categorized by difficulty: `simple` (8), `moderate` (8), `challenging` (8).
   - **Empirical Execution Test**: All 24 SQL queries were executed against their respective SQLite databases using Python `sqlite3`.
   - Result: **24/24 queries executed successfully and produced non-empty record sets (100% pass rate)**.

3. **Synthetic CoT Trajectories (`data/mini/mini_synsql.jsonl`)**:
   - Exactly 20 records.
   - Every record contains valid `<prompt>`, `<think>` (step-by-step reasoning), and `<answer>` SQL tags conforming to Completion-Only loss training format.

4. **Workspace Cleanliness & Pre-Populated Artifact Detection**:
   - Executed `find . -name '*.log' -o -name '*result*' -o -name '*output*'`.
   - Result: No pre-populated logs, fake result attestation files, or cached outputs were found in the project tree.

---

## 2. Logic Chain (Audit Reasoning)

1. **Hypothesis: Did `worker_m1` cheat by hardcoding table names or expected bridge outputs in `steiner_tree.py`?**
   - *Observation*: Source code search reveals 0 mentions of test tables.
   - *Adversarial Test*: Tested on an unseen synthetic hospital schema (`patients -> admissions -> doctor_assignments -> doctors -> departments`).
   - *Result*: `find_steiner_tree(['patients', 'departments'])` returned `bridges = ['admissions', 'doctor_assignments', 'doctors']`, achieving 100% bridge recovery on an unseen database.
   - *Inference*: Cheating hypothesis is conclusively disproven. The algorithm is an authentic implementation of Kou-Markowsky-Berman (KMB) Steiner Tree approximation.

2. **Hypothesis: Are the database files in `data/mini/` dummy text or broken mocks?**
   - *Observation*: Magic bytes match standard SQLite 3 binary. All 8 tables contain relational data with foreign key integrity.
   - *Empirical Test*: All 24 queries from `mini_bird.jsonl` were executed directly and returned non-empty row sets matching domain logic.
   - *Inference*: The dataset is authentic and suitable for independent smoke testing.

3. **Hypothesis: Is `embedding_retriever.py` a placeholder or dummy facade?**
   - *Observation*: The file contains a complete BM25 implementation with custom tokenizer, term frequency calculation, document frequency indexing, and field weighting.
   - *Adversarial Test*: Queried on novel products/customers cards (`Which item is an iPhone?` -> correctly returned `products.name` with score 1.0; `Send electronic mail` -> correctly returned `customers.email` with score 1.0).
   - *Inference*: The retriever is functional and mathematically authentic.

---

## 3. Caveats & Boundary Observations

1. **Implicit Foreign Key Heuristic**:
   In `graph_builder.py`, `_infer_implicit_foreign_keys` is invoked when `infer_implicit_fks=True` and `len(edges) < len(tables) - 1`. If two tables both have columns ending in `_id` (e.g., both have `dept_id`), an implicit edge with weight 2.0 is inferred. For databases with complete explicit foreign keys, setting `infer_implicit_fks=False` ensures only explicit constraints are mapped.
2. **Offline Dense Vector Fallback**:
   Because the local test environment operates offline without downloaded model checkpoints from Hugging Face, the retriever seamlessly and reliably defaults to the built-in field-aware BM25 engine.

---

## 4. Conclusion & Final Verdict

**VERDICT: CLEAN**

The work products delivered by `worker_m1` for Milestone M1 (Data Pipeline & Schema Grounding) strictly adhere to all integrity mandates:
- **No hardcoded outputs or shortcut tables**.
- **No facade or dummy classes**.
- **No fabricated verification logs**.
- **All SQLite databases, column cards, and benchmark queries are authentic, fully functional, and independently verified**.

The codebase is genuine, robust, and approved for subsequent milestones (M2: Training Scripts & M3: Colab Notebooks).

---

## 5. Verification Method (Independent Replication)

Anyone can independently reproduce the forensic verification findings using the commands below:

### Command 1: Run Full 23-Test Adversarial Stress Suite
```bash
python3 -m unittest -v tests/test_m1_stress.py
```
*Expected Result*: `Ran 23 tests ... OK` (Exit code 0).

### Command 2: Execute All 24 BIRD Queries Against SQLite
```bash
python3 -c "
import json, sqlite3
conn_fin = sqlite3.connect('data/mini/financial_mini.sqlite')
conn_sch = sqlite3.connect('data/mini/california_schools.sqlite')
with open('data/mini/mini_bird.jsonl') as f:
    queries = [json.loads(l) for l in f if l.strip()]
for q in queries:
    conn = conn_fin if q['db_id'] == 'financial_mini' else conn_sch
    rows = conn.cursor().execute(q['SQL']).fetchall()
    assert len(rows) > 0, f'Query {q[\"question_id\"]} returned empty!'
conn_fin.close()
conn_sch.close()
print('All 24 queries executed successfully!')
"
```
*Expected Result*: `All 24 queries executed successfully!` (Exit code 0).

### Command 3: Test Steiner Tree on Synthetic Unseen Schema
```bash
python3 -c "
import tempfile, os, sqlite3, sys
sys.path.insert(0, '.')
from src.grounding.graph_builder import SchemaGraphBuilder
from src.grounding.steiner_tree import NodeWeightedSteinerTree

temp_dir = tempfile.mkdtemp()
db_path = os.path.join(temp_dir, 'synthetic.sqlite')
conn = sqlite3.connect(db_path)
conn.cursor().executescript('''
CREATE TABLE t1 (id1 INT PRIMARY KEY);
CREATE TABLE t2 (id2 INT PRIMARY KEY, id1 INT, FOREIGN KEY (id1) REFERENCES t1(id1));
CREATE TABLE t3 (id3 INT PRIMARY KEY, id2 INT, FOREIGN KEY (id2) REFERENCES t2(id2));
''')
conn.commit(); conn.close()

graph = SchemaGraphBuilder.build_from_sqlite(db_path, infer_implicit_fks=False)
sel, bridges, _, ver = NodeWeightedSteinerTree(graph).find_steiner_tree(['t1', 't3'])
assert 't2' in bridges, 'Bridge table t2 was not recovered!'
assert ver is True
print('Synthetic Steiner Tree recovery passed!')
"
```
*Expected Result*: `Synthetic Steiner Tree recovery passed!` (Exit code 0).
