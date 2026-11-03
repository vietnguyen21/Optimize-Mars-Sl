## 2026-11-02T05:18:49Z
You are worker_m3, a teamwork_worker subagent fulfilling the role of [AGENT-GEN-G2] (Kỹ sư SLM Training & Constrained Decoding) for Milestone M3 (Phase 3: Google Colab Training Notebooks).

Your working directory is: /mnt/c/Workspace/optimize/.agents/teamwork/worker_m3/
You must first read the original user request at:
/mnt/c/Workspace/optimize/.agents/teamwork/ORIGINAL_REQUEST.md
Also read PROJECT.md at:
/mnt/c/Workspace/optimize/PROJECT.md
And reference survey reports and previous milestone handoffs at:
/mnt/c/Workspace/optimize/.agents/teamwork/spec_miner_survey_1/survey_report.md
/mnt/c/Workspace/optimize/.agents/teamwork/spec_miner_survey_2/survey_report.md
/mnt/c/Workspace/optimize/.agents/teamwork/explorer_survey_3/survey_report.md
/mnt/c/Workspace/optimize/.agents/teamwork/worker_m1/handoff.md
/mnt/c/Workspace/optimize/.agents/teamwork/worker_m2/handoff.md

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Your exclusive file ownership:
- notebooks/01_data_preparation_and_embedding.ipynb
- notebooks/02_sft_and_jepa_training_3b.ipynb
- notebooks/03_grpo_training_3b_and_7b.ipynb
- notebooks/04_system1_laya421m_training.ipynb
- notebooks/05_end_to_end_evaluation.ipynb
- tests/test_notebooks.py

Scope and Requirements for Milestone M3 (Phase 3):
1. Construct 5 production-ready, fully formed Jupyter Notebooks in valid `.ipynb` JSON format (nbformat v4):
   - `notebooks/01_data_preparation_and_embedding.ipynb`:
     * Clone / setup workspace, install dependencies, prepare BIRD / Spider / mini data.
     * Extract 6-field column cards, build PK-FK schema graph with bridge table preservation.
     * Run `LitESQLEmbeddingRetriever` and export offline vector indices.
   - `notebooks/02_sft_and_jepa_training_3b.ipynb`:
     * Google Colab environment setup (detect GPU: T4 / L4 / A100).
     * Google Drive automated mounting (`from google.colab import drive; drive.mount('/content/drive')`).
     * Checkpoint directory configuration (`/content/drive/MyDrive/Text2SQL_Colab/checkpoints/sft_jepa_3b/`).
     * Session recovery: `--resume_from_checkpoint latest`.
     * Model loading: FINER-SQL-3B (Unsloth 4-bit / 16-bit or HuggingFace PEFT LoRA r=64 to 512).
     * Training loop using `CompletionOnlyDataCollator` (masking prompt tokens with -100) and `compute_jepa_loss` with 2-block causal mask and `[PRED]` token.
     * `DriveCheckpointCallback` integration.
   - `notebooks/03_grpo_training_3b_and_7b.ipynb`:
     * Supports both 3B (FINER-SQL-3B) and 7B (Qwen2.5-Coder-7B).
     * Hardware configuration switch: Free T4 (QLoRA 4-bit, micro-batch 1, 8 rollouts), Pro L4 (BF16, micro-batch 2), Pro+ A100 (BF16, 16 rollouts).
     * Google Drive mounting and checkpoint sync.
     * Dense GRPO implementation with 4 reward heads ($R_{\text{format}}, R_{\text{exec}}, R_{\text{mem}}, R_{\text{atomic}}$) and group advantage normalization.
     * Training and evaluation logging.
   - `notebooks/04_system1_laya421m_training.ipynb`:
     * ModernBERT-large (Laya-421M) RLCD training with dual learning rate ($10^{-5}$ backbone, $10^{-4}$ head).
     * Temperature scaling calibration achieving $ECE \le 0.081$.
     * Export calibrated model weights to Google Drive.
   - `notebooks/05_end_to_end_evaluation.ipynb`:
     * Full evaluation pipeline on Google Colab: candidate generation (8-16 rollouts), XGrammar PDA constrained decoding & fallback, PV-SQL 10 AST rules, System-1 0-token early exit gate, and benchmark metrics calculation (EX, VSR, latency).
2. Verification test `tests/test_notebooks.py`:
   - Validate that all 5 `.ipynb` files are valid JSON.
   - Verify that all code cells have valid Python syntax (`ast.parse`).
   - Verify that Google Drive mount code is present in notebooks 02, 03, 04, 05.
   - Verify that 3B and 7B configurations are present in notebook 03.
   - Verify that session recovery (`--resume_from_checkpoint` / `find_latest_checkpoint`) is present.
3. Write your completion report in `/mnt/c/Workspace/optimize/.agents/teamwork/worker_m3/handoff.md` and update `progress.md`. Send a message to orchestrator when done.
