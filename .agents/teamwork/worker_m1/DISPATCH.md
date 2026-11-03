## 2026-11-02T04:17:38Z
You are worker_m1, a teamwork_worker subagent fulfilling the role of [AGENT-DATA-G1] (Kỹ sư Data Pipeline & Schema Grounding) according to AGENTS_RESEARCH.md and docs/specifications/.

Your working directory is: /mnt/c/Workspace/optimize/.agents/teamwork/worker_m1/
You must first read the original user request at:
/mnt/c/Workspace/optimize/.agents/teamwork/ORIGINAL_REQUEST.md
Also read the master project specification at:
/mnt/c/Workspace/optimize/PROJECT.md
And reference survey reports at:
/mnt/c/Workspace/optimize/.agents/teamwork/spec_miner_survey_1/survey_report.md
/mnt/c/Workspace/optimize/.agents/teamwork/spec_miner_survey_2/survey_report.md
/mnt/c/Workspace/optimize/.agents/teamwork/explorer_survey_3/survey_report.md

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Your exclusive file ownership:
- src/__init__.py
- src/common/__init__.py
- src/common/schemas.py
- src/grounding/__init__.py
- src/grounding/column_card.py
- src/grounding/graph_builder.py
- src/grounding/steiner_tree.py
- src/grounding/embedding_retriever.py
- scripts/make_mini_dataset.py
- data/mini/

Scope and Requirements for Milestone M1 (Phase 1):
1. `src/common/schemas.py`: Implement Pydantic data models for `ColumnCard`, `CompactSubSchema`, `TableGraphNode`, `FKEdge`, `GenerationInput`, `SQLCandidate`, and `ValidationResult` according to PROJECT.md interface contracts.
2. `src/grounding/column_card.py`: Implement 6-field column card generation (`table_name`, `column_name`, `data_type`, `constraints`, `column_description`, `sample_values`) extracted directly from SQLite database schemas or schema dicts.
3. `src/grounding/graph_builder.py`: Build directed/undirected schema graph $G=(V, E)$ from SQLite foreign keys or schema metadata using `networkx`. Table nodes, column attributes, and PK-FK edges.
4. `src/grounding/steiner_tree.py`: Implement the Node-Weighted Steiner Tree (NWST) approximation algorithm. Given query terminal nodes (tables referenced in question), find the minimal connecting subgraph that bridges them. Must guarantee bridge table preservation (100% bridge table recovery when 2 terminal tables connect only via a bridge table).
5. `src/grounding/embedding_retriever.py`: Implement LitE-SQL schema retrieval logic. Provide embedding extraction over column cards with cosine similarity. Include a robust fallback (e.g. TF-IDF / BM25 / token overlap) when embedding model weights are not loaded locally, so that it works seamlessly offline.
6. `scripts/make_mini_dataset.py`: Python script that creates the miniature evaluation dataset at `data/mini/`:
   - `data/mini/financial_mini.sqlite`: Real SQLite database with at least 4 related tables (e.g. `clients`, `accounts`, `transactions`, `cards`) with explicit PK-FK relations and multi-hop paths.
   - `data/mini/california_schools.sqlite`: Real SQLite database with at least 3 tables (e.g. `schools`, `satscores`, `frpm`) with PK-FK relations.
   - `data/mini/mini_bird.jsonl`: At least 20 sample BIRD queries with db_id, question, evidence, ground truth SQL, and difficulty levels.
   - `data/mini/mini_synsql.jsonl`: Sample synthetic reasoning trajectories with question, schema, think steps, and SQL answer.
7. Verification: Run `python3 scripts/make_mini_dataset.py` to populate `data/mini/`. Write a self-contained test verification in your report confirming that SQLite databases are created, tables and foreign keys are valid, and Steiner Tree finds bridge tables.
8. Write your completion report in `/mnt/c/Workspace/optimize/.agents/teamwork/worker_m1/handoff.md` and update `progress.md`. Send a message to the orchestrator when done.
