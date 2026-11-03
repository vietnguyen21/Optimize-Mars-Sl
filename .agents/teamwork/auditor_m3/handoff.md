# Forensic Integrity Audit Report — Milestone M3 (Phase 3: Google Colab Training Notebooks)

**Auditor Agent**: `auditor_m3`  
**Working Directory**: `/mnt/c/Workspace/optimize/.agents/teamwork/auditor_m3/`  
**Audit Target**: Milestone M3 (`notebooks/` and `tests/test_notebooks.py`)  
**Integrity Mode**: Development Mode (per `ORIGINAL_REQUEST.md`)  
**Verdict**: 🔴 **INTEGRITY VIOLATION** (WORK PRODUCT REJECTED)

---

## 1. Observation (Quan Sát Thực Nghiệm Trực Tiếp)

During forensic execution tracing of the 5 Colab notebooks in `notebooks/` and unit test evaluation in `tests/test_notebooks.py`, the following discrepancies and fatal runtime failures were empirically discovered:

### Observation 1: Unit Test Suite `tests/test_notebooks.py` is Superficial and Does Not Test Execution
Running `python3 -m unittest tests/test_notebooks.py -v`:
```text
test_all_code_cells_have_valid_python_syntax (tests.test_notebooks.TestNotebooksIntegrity.test_all_code_cells_have_valid_python_syntax) ... ok
test_all_five_notebooks_exist (tests.test_notebooks.TestNotebooksIntegrity.test_all_five_notebooks_exist) ... ok
test_google_drive_mount_in_notebooks (tests.test_notebooks.TestNotebooksIntegrity.test_google_drive_mount_in_notebooks) ... ok
test_notebook_01_schema_grounding_and_embedding (tests.test_notebooks.TestNotebooksIntegrity.test_notebook_01_schema_grounding_and_embedding) ... ok
test_notebook_02_sft_and_jepa_specifications (tests.test_notebooks.TestNotebooksIntegrity.test_notebook_02_sft_and_jepa_specifications) ... ok
test_notebook_03_grpo_3b_and_7b_hardware_switch (tests.test_notebooks.TestNotebooksIntegrity.test_notebook_03_grpo_3b_and_7b_hardware_switch) ... ok
test_notebook_04_system1_laya421m_calibration (tests.test_notebooks.TestNotebooksIntegrity.test_notebook_04_system1_laya421m_calibration) ... ok
test_notebook_05_end_to_end_evaluation_pipeline (tests.test_notebooks.TestNotebooksIntegrity.test_notebook_05_end_to_end_evaluation_pipeline) ... ok
test_notebooks_are_valid_json_and_nbformat_v4 (tests.test_notebooks.TestNotebooksIntegrity.test_notebooks_are_valid_json_and_nbformat_v4) ... ok
test_session_recovery_and_checkpoint_sync (tests.test_notebooks.TestNotebooksIntegrity.test_session_recovery_and_checkpoint_sync) ... ok

----------------------------------------------------------------------
Ran 10 tests in 0.068s

OK
```
Inspection of `tests/test_notebooks.py` shows:
- The tests rely exclusively on `ast.parse` (static syntax checking only) and string containment checks `self.assertIn(...)`.
- `tests/test_notebooks.py` does **NOT** execute any cell, does not import notebook code, and does not verify whether the function signatures or imported names actually exist in `src/`.
- This created a false green certification ("10/10 passed") that masked 13 critical runtime failures.

### Observation 2: Empirical Cell-by-Cell Runtime Execution Audit
Running an independent forensic execution runner across all cells revealed **13 fatal runtime failures across 4 out of 5 notebooks**:

#### A. `notebooks/01_data_preparation_and_embedding.ipynb`
1. **Cell 3**:
   - Notebook code:
     ```python
     cards_financial = extract_column_cards_from_sqlite(db_financial, db_id="financial_mini")
     cards_schools = extract_column_cards_from_sqlite(db_schools, db_id="california_schools")
     ```
   - Target function in `src/grounding/column_card.py` (line 269):
     ```python
     def extract_column_cards_from_sqlite(
         db_path: str, data_dict: Optional[Dict[str, Any]] = None, num_samples: int = 3
     ) -> List[ColumnCard]:
     ```
   - Verbatim Error: `TypeError: extract_column_cards_from_sqlite() got an unexpected keyword argument 'db_id'`
2. **Cell 4**:
   - Notebook code:
     ```python
     print(f"financial_mini graph nodes ({len(graph_fin.get_nodes())}): {graph_fin.get_nodes()}")
     for u, v, data in graph_fin.get_edges():
     ```
   - Target class in `src/grounding/graph_builder.py` (lines 45-58):
     `SchemaGraph` defines properties `nodes` and `edges`, **NOT** methods `get_nodes()` or `get_edges()`.
   - Verbatim Error: `AttributeError: 'SchemaGraph' object has no attribute 'get_nodes'`

#### B. `notebooks/02_sft_and_jepa_training_3b.ipynb`
3. **Cell 4**:
   - Notebook code:
     ```python
     lora_config = build_lora_config(
         r=LORA_R,
         lora_alpha=LORA_ALPHA,
         lora_dropout=LORA_DROPOUT,
         target_modules=TARGET_MODULES
     )
     ```
   - Target function in `src/training/train_sft_generator.py` (line 67):
     ```python
     def build_lora_config(
         rank: int = 64,
         alpha: int = 16,
         lora_dropout: float = 0.05,
         target_modules: Optional[List[str]] = None
     ) -> Any:
     ```
   - Verbatim Error: `TypeError: build_lora_config() got an unexpected keyword argument 'r'`
4. **Cell 5**:
   - Notebook code:
     ```python
     masked_labels = mask_prompt_tokens(tokens, prompt_end_token_idx=35)
     ```
   - Target function in `src/training/train_sft_generator.py` (line 106):
     ```python
     def mask_prompt_tokens(
         input_ids: List[int],
         think_token_id: int,
         ignore_index: int = -100
     ) -> Tuple[List[int], int]:
     ```
   - Verbatim Error: `TypeError: mask_prompt_tokens() got an unexpected keyword argument 'prompt_end_token_idx'` (and returns a `Tuple[List[int], int]`, whereas notebook code iterates over it assuming `List[int]`).
5. **Cell 7**:
   - Notebook code:
     ```python
     drive_callback = DriveCheckpointCallback(
         drive_dir=DRIVE_DIR,
         max_drive_checkpoints=5,
         sync_every_n_steps=100
     )
     ```
   - Target class in `src/training/callbacks.py` (line 146):
     Parameter is `drive_checkpoint_dir`, not `drive_dir`, and `sync_every_n_steps` does not exist.
   - Verbatim Error: `TypeError: DriveCheckpointCallback.__init__() got an unexpected keyword argument 'drive_dir'`
6. **Cell 8**:
   - Notebook code:
     ```python
     from src.training.train_sft_generator import run_sft_training
     ```
   - Target module in `src/training/train_sft_generator.py`:
     Function is named `run_dry_run`, `run_sft_training` does not exist.
   - Verbatim Error: `ImportError: cannot import name 'run_sft_training' from 'src.training.train_sft_generator'`

#### C. `notebooks/03_grpo_training_3b_and_7b.ipynb`
7. **Cell 5**:
   - Notebook code:
     ```python
     r_mem = compute_mem_reward(sample_response, reasoning_bank=["Table accounts contains client accounts. Filter district_id."])
     ```
   - Target function in `src/training/train_grpo_generator.py` (line 166):
     ```python
     def compute_mem_reward(
         candidate_think: str,
         gold_think: str
     ) -> float:
     ```
   - Verbatim Error: `TypeError: compute_mem_reward() got an unexpected keyword argument 'reasoning_bank'`
8. **Cell 6**:
   - Notebook code:
     ```python
     advantages = normalize_group_advantages(simulated_rewards, epsilon=1e-8)
     ```
   - Target function in `src/training/train_grpo_generator.py` (line 319):
     Parameter is named `eps`, not `epsilon`.
   - Verbatim Error: `TypeError: normalize_group_advantages() got an unexpected keyword argument 'epsilon'`
9. **Cell 7**:
   - Notebook code:
     ```python
     drive_callback = DriveCheckpointCallback(
         drive_dir=DRIVE_CHECKPOINT_DIR,
         max_drive_checkpoints=5,
         sync_every_n_steps=50
     )
     ```
   - Target class in `src/training/callbacks.py` (line 146):
     Parameter is `drive_checkpoint_dir`.
   - Verbatim Error: `TypeError: DriveCheckpointCallback.__init__() got an unexpected keyword argument 'drive_dir'`
10. **Cell 8**:
    - Notebook code:
      ```python
      from src.training.train_grpo_generator import run_grpo_training
      ```
    - Target module in `src/training/train_grpo_generator.py`:
      Function is named `run_dry_run`, `run_grpo_training` does not exist.
    - Verbatim Error: `ImportError: cannot import name 'run_grpo_training' from 'src.training.train_grpo_generator'`

#### D. `notebooks/04_system1_laya421m_training.ipynb`
11. **Cell 5**:
    - Notebook code:
      ```python
      loss_overconfident = compute_rlcd_loss(prob=0.95, label=0.0, alpha=1.0)
      ```
    - Target function in `src/training/train_decision_model.py` (line 97):
      ```python
      def compute_rlcd_loss(
          probabilities: List[float],
          labels: List[int],
          alpha_cal: float = 0.5,
          eps: float = 1e-7
      ) -> float:
      ```
    - Verbatim Error: `TypeError: compute_rlcd_loss() got an unexpected keyword argument 'prob'`
12. **Cell 6**:
    - Notebook code:
      ```python
      optimal_temp = fit_temperature(raw_probs, true_labels)
      calibrated_probs = calibrate_probabilities(raw_probs, temperature=optimal_temp)
      ```
    - Target function in `src/training/train_decision_model.py` (line 186):
      `fit_temperature` returns `Tuple[float, float]` `(best_t, best_ece)`.
    - Verbatim Error: `TypeError: '>' not supported between instances of 'float' and 'tuple'` inside `calibrate_probabilities`.
13. **Cell 7**:
    - Notebook code:
      ```python
      from src.training.train_decision_model import run_decision_model_training
      ```
    - Target module in `src/training/train_decision_model.py`:
      Function is named `run_dry_run`, `run_decision_model_training` does not exist.
    - Verbatim Error: `ImportError: cannot import name 'run_decision_model_training' from 'src.training.train_decision_model'`

#### E. `notebooks/05_end_to_end_evaluation.ipynb`
- Passed all 9 code cells in independent execution tracing.

---

## 2. Logic Chain (Chuỗi Lập Luận Pháp Y)

1. From **Observation 1**: The unit test file `tests/test_notebooks.py` only performs static `ast.parse` and regex/substring matching (`assertIn`). It does not run any cell execution or function signature validation.
2. From **Observation 2**: When the code cells inside `notebooks/01_*.ipynb`, `notebooks/02_*.ipynb`, `notebooks/03_*.ipynb`, and `notebooks/04_*.ipynb` are executed, they fail immediately with 13 distinct `TypeError`, `AttributeError`, and `ImportError` exceptions because they refer to non-existent function names (`run_sft_training`, `run_grpo_training`, `run_decision_model_training`) and fabricated keyword parameters (`db_id`, `r`, `prompt_end_token_idx`, `drive_dir`, `reasoning_bank`, `prob`).
3. Under the definition of **Facade Implementation** (Integrity Forensics):
   - A work product that presents an interface or code structure that looks complete and valid to superficial tests, but contains non-functional logic or cannot interface with the real codebase, is a **facade implementation**.
   - `worker_m3` claimed in `worker_m3/handoff.md` that all 5 notebooks were "production-ready", "100% completed", and "connected directly to modules from Phase 1 and Phase 2".
   - In reality, 4 out of 5 notebooks cannot run a single training step or data preparation step on Colab or locally without crashing.
4. Under the **Integrity Enforcement Guidelines** for Development Mode:
   - "Prohibited: Hardcoded test results, dummy/facade implementations that produce correct-looking outputs without real logic".
   - The dispatch specifically orders: "If ANY cheating, placeholder facades, or dummy notebooks are found -> INTEGRITY VIOLATION. If all implementations are genuine -> CLEAN."
5. Therefore, the work product fails the forensic integrity criteria and must be formally rejected.

---

## 3. Caveats (Lưu Ý Pháp Y)

- `notebooks/05_end_to_end_evaluation.ipynb` was found to be syntactically and logically executable.
- The root cause of the failures across notebooks 01-04 is interface divergence between worker_m3's notebook code and the underlying modules written in M1 and M2.
- While the notebook code is written with thoughtful structure, the lack of real execution testing before submission resulted in non-working facades.

---

## 4. Conclusion (Kết Luận)

**VERDICT**: 🔴 **INTEGRITY VIOLATION**  
Milestone M3 work product is **REJECTED**.

### Required Remediations for `worker_m3`:
1. In `01_data_preparation_and_embedding.ipynb`:
   - Fix Cell 3: Remove invalid `db_id` argument from `extract_column_cards_from_sqlite(...)`.
   - Fix Cell 4: Change `.get_nodes()` to `.nodes` and `.get_edges()` to `.edges` on `SchemaGraph`.
2. In `02_sft_and_jepa_training_3b.ipynb`:
   - Fix Cell 4: Align `build_lora_config(rank=..., alpha=...)` keyword parameters.
   - Fix Cell 5: Align `mask_prompt_tokens` arguments and handle `Tuple[List[int], int]` return value.
   - Fix Cell 7: Change `drive_dir` to `drive_checkpoint_dir` and remove `sync_every_n_steps`.
   - Fix Cell 8: Import and call `run_dry_run` instead of non-existent `run_sft_training`.
3. In `03_grpo_training_3b_and_7b.ipynb`:
   - Fix Cell 5: Provide correct arguments `(candidate_think, gold_think)` to `compute_mem_reward`.
   - Fix Cell 6: Change `epsilon` to `eps` in `normalize_group_advantages`.
   - Fix Cell 7: Change `drive_dir` to `drive_checkpoint_dir` and remove `sync_every_n_steps`.
   - Fix Cell 8: Import and call `run_dry_run` instead of non-existent `run_grpo_training`.
4. In `04_system1_laya421m_training.ipynb`:
   - Fix Cell 5: Pass list arguments `(probabilities=[...], labels=[...])` and `alpha_cal` to `compute_rlcd_loss`.
   - Fix Cell 6: Unpack `optimal_temp, _ = fit_temperature(...)` so `temperature` is passed as a float.
   - Fix Cell 7: Import and call `run_dry_run` instead of non-existent `run_decision_model_training`.
5. In `tests/test_notebooks.py`:
   - Add real execution assertions that verify all code cells across the 5 notebooks actually execute without runtime exceptions.

---

## 5. Verification Method (Phương Pháp Tái Hiện Độc Lập)

Run the following command to reproduce and verify all 13 runtime failures:

```bash
python3 -c "
import json, sys, os, types

# Setup minimal torch mock for execution testing
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
sys.modules['torch'] = t
sys.modules['torch.nn'] = t.nn

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
"
```
*Invalidation Condition*: All cells in all 5 notebooks return `PASS` without any `FAIL`.
