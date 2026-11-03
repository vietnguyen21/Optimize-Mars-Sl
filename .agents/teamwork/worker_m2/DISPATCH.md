## 2026-11-02T04:49:20Z
You are worker_m2, a teamwork_worker subagent fulfilling the role of [AGENT-GEN-G2] (Kỹ sư SLM Training & Constrained Decoding) according to AGENTS_RESEARCH.md and docs/specifications/.

Your working directory is: /mnt/c/Workspace/optimize/.agents/teamwork/worker_m2/
You must first read the original user request at:
/mnt/c/Workspace/optimize/.agents/teamwork/ORIGINAL_REQUEST.md
Also read the master project specification at:
/mnt/c/Workspace/optimize/PROJECT.md
And reference survey reports and previous milestone handoffs at:
/mnt/c/Workspace/optimize/.agents/teamwork/spec_miner_survey_1/survey_report.md
/mnt/c/Workspace/optimize/.agents/teamwork/spec_miner_survey_2/survey_report.md
/mnt/c/Workspace/optimize/.agents/teamwork/worker_m1/handoff.md

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Your exclusive file ownership:
- src/training/__init__.py
- src/training/callbacks.py
- src/training/jepa_module.py
- src/training/train_sft_generator.py
- src/training/train_grpo_generator.py
- src/training/train_embedding.py
- src/training/train_decision_model.py

Scope and Requirements for Milestone M2 (Phase 2):
1. `src/training/callbacks.py`:
   - Implement `DriveCheckpointCallback` (inheriting from `transformers.TrainerCallback` when available or generic PyTorch callback).
   - Automatically sync checkpoints from local `/content/checkpoints/` to Google Drive `/content/drive/MyDrive/Text2SQL_Colab/checkpoints/`.
   - Support session recovery flag `--resume_from_checkpoint latest`.
2. `src/training/jepa_module.py`:
   - Implement the LLM-JEPA (Joint-Embedding Predictive Architecture) module for Text-to-SQL.
   - Formulate 2-block causal mask with `[PRED]` prediction token.
   - Compute Cosine Distance Loss between predicted latent representations (question + schema) and target latent representations (reasoning thought + SQL), scaled by $\lambda=1.0$ with Random JEPA Loss Dropout ($LD=0.5$).
3. `src/training/train_sft_generator.py`:
   - Implement Completion-Only Cross-Entropy Loss SFT training script for FINER-SQL-3B / Qwen2.5-Coder-7B.
   - Mask prompt/schema tokens with label `-100` so loss is computed strictly over `<think>...</think><answer>...</answer>`.
   - Support LoRA/QLoRA configuration (rank=64 to 512, alpha=16, target modules q, k, v, o, gate, up, down).
   - Support dual execution: full training mode with HuggingFace/TRL/PEFT, and a clean `--dry_run` mode that verifies token masking, forward pass, and loss calculation on a small dummy tensor / `data/mini/mini_synsql.jsonl` without downloading heavy model weights.
4. `src/training/train_grpo_generator.py`:
   - Implement Dense GRPO (Group Relative Policy Optimization) trainer with 4 fine-grained reward heads:
     * $R_{\text{format}}$: binary 0/1 for `<think>...</think><answer>...</answer>` tags.
     * $R_{\text{exec}}$: 0/1 execution match on target SQLite database.
     * $R_{\text{mem}}$: soft reward matching against Reasoning Bank patterns.
     * $R_{\text{atomic}}$: AST clause matching reward (SELECT, FROM, WHERE, GROUP BY, ORDER BY).
   - Normalize group advantages to resolve Advantage Collapse.
   - Support `--dry_run` mode on `data/mini/`.
5. `src/training/train_embedding.py`:
   - Implement training for LitE-SQL Qwen3-Embedding-0.6B with Hard-Negative Supervised Contrastive Loss (HN-SupCon, margin=0.1).
6. `src/training/train_decision_model.py`:
   - Implement Laya-421M (ModernBERT) RLCD training with dual learning rates ($10^{-5}$ backbone, $10^{-4}$ head) and $ECE \le 0.081$ probability calibration.
7. Verification:
   - Run verification tests on all training modules (e.g. testing `DriveCheckpointCallback`, Completion-Only loss calculation, JEPA cosine distance computation, and Dense GRPO 4-head reward computation).
8. Write your completion report in `/mnt/c/Workspace/optimize/.agents/teamwork/worker_m2/handoff.md` and update `progress.md`. Send a message to the orchestrator when done.
