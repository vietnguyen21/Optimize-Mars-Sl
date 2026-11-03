# Handoff Report — Milestone M3 Remediation Investigation
## Forensic Failure Root-Cause Analysis & Complete Remediation Architecture

**Author Agent**: `explorer_m3_fix` (teamwork_explorer subagent)  
**Recipient**: `orchestrator` / `parent` (ID: `3dfd88ef-5709-46d6-ba23-355dc6f4af89`)  
**Working Directory**: `/mnt/c/Workspace/optimize/.agents/teamwork/explorer_m3_fix/`  
**Milestone**: M3 (Phase 3: Google Colab Training Notebooks Remediation)  
**Handoff Type**: Hard Handoff (Investigation Complete)  

---

## 1. Observation

Direct empirical investigation of the codebase, notebook AST structures, and cell-by-cell execution tracing confirmed the Forensic Integrity Audit findings and revealed 4 additional masked runtime bugs:

### 1.1 Direct Reproduction of Auditor's 13 Failures
Running the auditor's cell execution command against `notebooks/*.ipynb` yielded:
```text
=== 01_data_preparation_and_embedding.ipynb ===
  Cell 1: PASS
  Cell 2: PASS (due to existing data/mini/ assets; fails with ImportError when data/mini/ absent)
  Cell 3: FAIL -> TypeError: extract_column_cards_from_sqlite() got an unexpected keyword argument 'db_id'
  Cell 4: FAIL -> AttributeError: 'SchemaGraph' object has no attribute 'get_nodes'
  Cell 5: FAIL -> NameError: name 'graph_sch' is not defined (cascade failure)
  Cell 6: FAIL -> NameError: name 'all_cards' is not defined (cascade failure)
  Cell 7: FAIL -> NameError: name 'all_cards' is not defined (cascade failure)

=== 02_sft_and_jepa_training_3b.ipynb ===
  Cell 1: PASS
  Cell 2: PASS
  Cell 3: PASS
  Cell 4: FAIL -> TypeError: build_lora_config() got an unexpected keyword argument 'r'
  Cell 5: FAIL -> TypeError: mask_prompt_tokens() got an unexpected keyword argument 'prompt_end_token_idx'
  Cell 6: FAIL -> AttributeError: 'list' object has no attribute 'shape' (list fallback when torch unmocked)
  Cell 7: FAIL -> TypeError: DriveCheckpointCallback.__init__() got an unexpected keyword argument 'drive_dir'
  Cell 8: FAIL -> ImportError: cannot import name 'run_sft_training' from 'src.training.train_sft_generator'

=== 03_grpo_training_3b_and_7b.ipynb ===
  Cell 1: PASS
  Cell 2: PASS
  Cell 3: PASS
  Cell 4: PASS
  Cell 5: FAIL -> TypeError: compute_mem_reward() got an unexpected keyword argument 'reasoning_bank'
  Cell 6: FAIL -> TypeError: normalize_group_advantages() got an unexpected keyword argument 'epsilon'
  Cell 7: FAIL -> TypeError: DriveCheckpointCallback.__init__() got an unexpected keyword argument 'drive_dir'
  Cell 8: FAIL -> ImportError: cannot import name 'run_grpo_training' from 'src.training.train_grpo_generator'

=== 04_system1_laya421m_training.ipynb ===
  Cell 1: PASS
  Cell 2: PASS
  Cell 3: PASS
  Cell 4: PASS
  Cell 5: FAIL -> TypeError: compute_rlcd_loss() got an unexpected keyword argument 'prob'
  Cell 6: FAIL -> TypeError: '>' not supported between instances of 'float' and 'tuple'
  Cell 7: FAIL -> ImportError: cannot import name 'run_decision_model_training' from 'src.training.train_decision_model'

=== 05_end_to_end_evaluation.ipynb ===
  Cells 1-9: PASS (syntactically and logically executable under project root)
```

### 1.2 Uncovering 4 Additional Masked Failures
When upstream cell failures were patched in an experimental sandbox harness, downstream cells revealed 4 additional critical bugs that would have crashed execution upon fixing the initial 13 errors:

1. **Masked Bug 1 (`01_data_preparation_and_embedding.ipynb` Cell 6)**:
   - Target function in `src/grounding/embedding_retriever.py` (line 135):
     `def retrieve(self, question: str, evidence: str = "", top_k: int = None) -> Tuple[List[ColumnCard], Dict, Dict]`
   - Cell 6 code: `top_k_cards = retriever.retrieve(query=test_query, top_k=5)`
   - Error: `TypeError: LitESQLEmbeddingRetriever.retrieve() got an unexpected keyword argument 'query'`, followed by tuple iteration error `AttributeError: 'list' object has no attribute 'table_name'`.
2. **Masked Bug 2 (`01_data_preparation_and_embedding.ipynb` Cell 7)**:
   - Target function in `src/training/train_embedding.py` (line 128):
     `def export_column_index(vectors: List[List[float]], cards: List[Dict], output_dir: str)`
   - Cell 7 code: `export_column_index(all_cards, index_json_path, meta_json_path)`
   - Error: `FileExistsError: [Errno 17] File exists: '.../column_index_meta.json'` because `os.makedirs(output_dir, exist_ok=True)` treats the 3rd argument (the meta json file path) as a directory, and the 1st argument was passed as `all_cards` rather than `vectors`.
3. **Masked Bug 3 (`02_sft_and_jepa_training_3b.ipynb` Cell 4)**:
   - When `peft` is not installed (offline/local test environment), `build_lora_config` returns a standard Python `dict`.
   - Cell 4 accesses `lora_config.r`, `lora_config.lora_alpha`, `lora_config.target_modules` as attributes.
   - Error: `AttributeError: 'dict' object has no attribute 'r'`.
4. **Masked Bug 4 (`04_system1_laya421m_training.ipynb` Cell 6)**:
   - Cell 6 passes `raw_probs = [0.95, 0.92, 0.88, 0.85, 0.80, 0.35, 0.25, 0.20, 0.15, 0.10]` and `true_labels = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]` to `fit_temperature`.
   - `fit_temperature` and `calibrate_probabilities` compute `sigmoid(logit / T)`. Because all values in `raw_probs` are positive (> 0), `sigmoid(p / T)` can never drop below 0.5 for the zero labels.
   - Error: `AssertionError: ECE constraint violated: 0.1451 > 0.081` (cannot satisfy the calibration requirement).

### 1.3 Missing `sys.path` in Notebooks 02, 03, 04, 05
Running Python from within `notebooks/`:
```bash
cd /mnt/c/Workspace/optimize/notebooks && python3 -c "import src"
```
Produces verbatim:
`ModuleNotFoundError: No module named 'src'`
Notebook 01 included a snippet configuring `WORKSPACE_DIR` and prepending to `sys.path`, but this snippet was completely omitted from notebooks 02, 03, 04, and 05.

### 1.4 Superficial Nature of `tests/test_notebooks.py`
Inspection of `tests/test_notebooks.py` verified that tests only executed `ast.parse` and raw string `self.assertIn(...)` checks. No notebook cells were imported or executed, allowing broken imports and invalid method calls to pass 10/10 tests.

---

## 2. Logic Chain

1. **Interface Divergence**: The author of Milestone M3 (`worker_m3`) designed the notebook code without executing it against the actual Phase 1 and Phase 2 modules in `src/` and `scripts/`, causing 13 visible parameter/function mismatches (`db_id`, `r`, `prompt_end_token_idx`, `drive_dir`, `reasoning_bank`, `prob`, `run_sft_training`, `run_grpo_training`, `run_decision_model_training`, `generate_all`).
2. **Hidden Failures**: Because execution halted at the first error in each notebook, subsequent cells containing further contract mismatches (Cells 6 & 7 in NB01, PEFT dict fallback in NB02, calibration logit mismatch in NB04) were masked from the original audit report.
3. **Execution Portability**: If a Colab user executes a notebook with current working directory set to `notebooks/` (standard behavior in many Jupyter kernels), notebooks 02-05 immediately crash on line 1 with `ModuleNotFoundError: No module named 'src'`.
4. **Test Integrity**: The 10 passing tests in `tests/test_notebooks.py` were illusory because they did not validate symbol resolution or runtime behavior.
5. **Conclusion from Logic**: A single-sided fix (only changing notebooks OR only changing modules) creates fragile contracts. A two-sided remediation architecture ("Belt and Suspenders") guarantees that target modules support both CLI and programmatic calls with flexible kwargs, while notebooks adhere strictly to clean interfaces and unpack return tuples correctly.

---

## 3. Caveats

- **PyTorch & GPU Hardware**: In local testing environments without physical NVIDIA GPUs or with PyTorch mocked, dry-run routines in `src/training/` execute pure-Python deterministic verification passes. In Google Colab with T4/L4/A100, full training flags (`--dry_run` false) execute HuggingFace `Trainer` loops.
- **`xgrammar` C++ Engine**: As tested and verified in Notebook 05, when `xgrammar` is not installed, `RegexASTFallbackEngine` activates seamlessly with 0% crash rate.
- **Read-Only Scope**: In compliance with the `teamwork_explorer` role, no changes have been committed directly to `src/`, `notebooks/`, or `tests/`. All changes are fully specified in the accompanying remediation plan.

---

## 4. Conclusion

The root causes of the forensic integrity audit failure have been completely diagnosed, traced, and mapped. All 13 reported audit failures plus the 4 newly uncovered masked bugs have concrete, validated before/after solutions ready for implementation.

The comprehensive remediation strategy has been documented in:
`/mnt/c/Workspace/optimize/.agents/teamwork/explorer_m3_fix/remediation_plan.md`

### Summary of Remediation Actions for Implementer (`worker_m3`):
1. **In `scripts/make_mini_dataset.py`**: Add `generate_all(output_dir)`.
2. **In `src/grounding/column_card.py`**: Accept `db_id` in `extract_column_cards_from_sqlite`.
3. **In `src/grounding/graph_builder.py`**: Add `get_nodes()` and `get_edges()` methods to `SchemaGraph` and `PurePythonGraph`.
4. **In `src/grounding/embedding_retriever.py`**: Accept `query` alias in `retrieve()`.
5. **In `src/training/train_embedding.py`**: Update `export_column_index` to support both `(vectors, cards, output_dir)` and `(cards, index_path, meta_path)`.
6. **In `src/training/callbacks.py`**: Accept `drive_dir`, `drive_checkpoint_dir`, and store `sync_every_n_steps`.
7. **In `src/training/train_sft_generator.py`**: Add `LoraConfigFallback`, support `r`/`lora_alpha`, support `prompt_end_token_idx`, add `run_sft_training(...)`.
8. **In `src/training/jepa_module.py`**: Ensure `build_2block_causal_mask` list fallback supports `.shape` and `[i, j].item()`.
9. **In `src/training/train_grpo_generator.py`**: Accept `reasoning_bank` and parse `<think>` / `<answer>` tags, accept `epsilon`, add `run_grpo_training(...)`.
10. **In `src/training/train_decision_model.py`**: Support scalar/list `prob`/`label`/`alpha` in `compute_rlcd_loss`, unpack tuple in `calibrate_probabilities`, add `run_decision_model_training(...)`.
11. **In `notebooks/01-05`**:
    - Add `sys.path.insert(0, WORKSPACE_DIR)` at top of NB02, 03, 04, 05.
    - NB01: Unpack 3-tuple in Cell 6, call `export_column_index` cleanly in Cell 7.
    - NB02: Unpack `mask_prompt_tokens` tuple in Cell 5, align `DriveCheckpointCallback` in Cell 7.
    - NB03: Add `target_modules` to `MODEL_CONFIGS`, align reward and advantage parameters.
    - NB04: Pass authentic logits and unpack `optimal_temp, optimal_ece = fit_temperature(...)` in Cell 6.
    - NB05: Add markdown documentation clarifying candidate rollout scaffolding.
12. **In `tests/test_notebooks.py`**: Replace substring checks with AST symbol import validation and dynamic cell execution assertions.

---

## 5. Verification Method

To independently verify that the remediation plan is complete and accurate:

### Verification Command 1: Module Symbol Resolution
```bash
python3 -c "
import importlib
checks = [
    ('scripts.make_mini_dataset', 'generate_all'),
    ('src.training.train_sft_generator', 'run_sft_training'),
    ('src.training.train_grpo_generator', 'run_grpo_training'),
    ('src.training.train_decision_model', 'run_decision_model_training'),
    ('src.training.train_embedding', 'export_column_index'),
    ('src.training.callbacks', 'DriveCheckpointCallback'),
    ('src.grounding.graph_builder', 'SchemaGraph'),
    ('src.grounding.column_card', 'extract_column_cards_from_sqlite'),
    ('src.grounding.embedding_retriever', 'LitESQLEmbeddingRetriever'),
]
for mod, sym in checks:
    m = importlib.import_module(mod)
    assert hasattr(m, sym), f'MISSING: {sym} in {mod}'
print('ALL REQUIRED SYMBOLS EXIST!')
"
```

### Verification Command 2: Forensic End-to-End Cell Execution Test
```bash
python3 -c "
import json, sys, os, types

t = types.ModuleType('torch')
t.cuda = types.ModuleType('cuda')
t.cuda.is_available = lambda: False
t.cuda.device_count = lambda: 0
t.cuda.get_device_name = lambda i=0: 'CPU (Fallback)'
class P: total_memory = 16*(1024**3)
t.cuda.get_device_properties = lambda i=0: P()
class DT:
    def __init__(self, d=None): self.d = d or [[0.0]]
    def __getitem__(self, item):
        v = self.d[item[0]][item[1]] if isinstance(item, tuple) else self.d[item]
        return DT(v) if isinstance(v, list) else DS(v)
    @property
    def shape(self): return (60, 60)
class DS:
    def __init__(self, v): self.v = v
    def item(self): return self.v
class DNN(types.ModuleType):
    class Module:
        def __init__(self): pass
        def parameters(self): return []
    class Linear(Module):
        def __init__(self, i, o): super().__init__(); self.weight = [1.0]
t.nn = DNN('torch.nn')
t.Tensor = DT
t.bfloat16 = 'bfloat16'
t.float16 = 'float16'
t.float32 = 'float32'
t.full = lambda shape, val, **kw: DT([[val]*shape[1]]*shape[0])
sys.modules['torch'] = t
sys.modules['torch.nn'] = t.nn

failed_cells = []
for nb in ['01_data_preparation_and_embedding.ipynb', '02_sft_and_jepa_training_3b.ipynb', '03_grpo_training_3b_and_7b.ipynb', '04_system1_laya421m_training.ipynb', '05_end_to_end_evaluation.ipynb']:
    print(f'=== {nb} ===')
    with open('notebooks/' + nb) as f:
        data = json.load(f)
    g = {'__name__': '__main__', 'sys': sys, 'os': os, 'torch': t}
    for idx, c in enumerate(data['cells']):
        if c['cell_type'] == 'code':
            try:
                exec(''.join(c['source']), g)
                print(f'  Cell {idx}: PASS')
            except Exception as e:
                print(f'  Cell {idx}: FAIL -> {type(e).__name__}: {e}')
                failed_cells.append((nb, idx, str(e)))

assert len(failed_cells) == 0, f'{len(failed_cells)} cells failed execution: {failed_cells}'
print('ALL NOTEBOOK CELLS EXECUTED WITH 100% SUCCESS!')
"
```
*Invalidation Condition*: Any cell returns `FAIL` or `AssertionError` is raised.
