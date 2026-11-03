## 2026-11-02T04:06:32Z
You are spec_miner_survey_2, a teamwork_spec_miner subagent.
Your working directory is: /mnt/c/Workspace/optimize/.agents/teamwork/spec_miner_survey_2/
You must first read the original user request at:
/mnt/c/Workspace/optimize/.agents/teamwork/ORIGINAL_REQUEST.md

Your role is to extract detailed requirements, roadmaps, and phase-by-phase implementation criteria:
1. Read docs/specifications/03_project_requirements.md and docs/specifications/04_roadmap_and_phases.md in detail.
2. For Phase 1 (Data Pipeline & Schema Grounding):
   - What data models, formats, and schemas are required?
   - What algorithms for PK-FK Graph construction, Steiner Tree pruning / schema subgraphs, and vector embeddings?
   - What are the requirements for data/mini/ (schema, records, ground-truth queries for testing)?
3. For Phase 2 (SLM Training Scripts in src/training/):
   - What are the specifications for Completion-Only Cross Entropy Loss?
   - What is the architecture and loss for LLM-JEPA (Joint-Embedding Predictive Architecture)?
   - What is the architecture and reward/loss design for GRPO (Group Relative Policy Optimization)?
   - Trainer arguments, LoRA/QLoRA configuration, saving strategies.
4. For Phase 3 (Notebooks in notebooks/):
   - What are the specifications for Google Colab notebooks?
   - Requirements for 3B (FINER-SQL) and 7B (Qwen2.5-Coder-7B)?
   - Automated Google Drive mounting, checkpoint directory layout, and recovery mechanisms.
5. For Phase 4 (Evaluation & Benchmark in src/evaluation/):
   - Requirements for sampling 8–16 candidate trajectories per prompt for ablation studies.
   - 0-token decoding scoring methodology.
   - XGrammar PDA grammar-constrained decoding mechanics, BNF/EBNF grammars for SQL, and fallback mechanisms on syntax/compilation failure.
   - Benchmarking metrics: Execution Accuracy (EX), Valid SQL Rate (VSR), latency, token efficiency.
6. Document your findings in /mnt/c/Workspace/optimize/.agents/teamwork/spec_miner_survey_2/survey_report.md and write a standard handoff in /mnt/c/Workspace/optimize/.agents/teamwork/spec_miner_survey_2/handoff.md.
7. When complete, send a message to the orchestrator summarizing your key findings and citing the report path. Keep progress updated in /mnt/c/Workspace/optimize/.agents/teamwork/spec_miner_survey_2/progress.md.
