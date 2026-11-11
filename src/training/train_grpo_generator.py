"""
Dense GRPO (Group Relative Policy Optimization) Trainer for Text-to-SQL SLMs.

Implements FINER-SQL Dense GRPO with 4 fine-grained reward heads:
  1. R_format in {0, 1}: Binary reward for well-formed <think>...</think><answer>...</answer> tags.
  2. R_exec in {0, 1, 2}: Execution reward on target SQLite database (0: error, 1: valid run, 2: 100% gold match).
  3. R_mem in [0, 1]: Soft semantic thought-matching reward against Reasoning Bank patterns.
  4. R_atomic in [0, 1]: AST clause matching reward (SELECT, FROM, WHERE, GROUP BY, ORDER BY).

Normalizes informative groups; zero-variance groups have zero advantages and are skipped:
  A_j = (R_j - mean(R)) / (std(R) + eps).

Supports full TRL / custom GRPO execution and standalone --dry_run mode on data/mini/.
"""

import os
import re
import sys
import json
import math
import sqlite3
import argparse
import logging
from pathlib import Path
import random
from typing import Optional, List, Dict, Any, Tuple, Union

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.training.callbacks import DriveCheckpointCallback, get_resume_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Check for PyTorch availability
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


# ---------------------------------------------------------------------------
# 1. Reward Head 1: Formatting Reward (R_format)
# ---------------------------------------------------------------------------

def compute_format_reward(generation_text: str) -> float:
    """
    Evaluates formatting adherence.
    Requires:
      - Valid <think>...</think> pair with non-whitespace content.
      - Valid <answer>...</answer> pair with non-whitespace content.
      - <think> appears before <answer>.
    
    Returns:
        1.0 if all conditions are met, otherwise 0.0.
    """
    if not generation_text:
        return 0.0

    think_match = re.search(r"<think>(.*?)</think>", generation_text, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", generation_text, re.DOTALL)

    if not think_match or not answer_match:
        return 0.0

    think_content = think_match.group(1).strip()
    answer_content = answer_match.group(1).strip()

    if not think_content or not answer_content:
        return 0.0

    # Ensure <think> precedes <answer>
    if think_match.start() >= answer_match.start():
        return 0.0

    return 1.0


# ---------------------------------------------------------------------------
# 2. Reward Head 2: Execution Reward (R_exec)
# ---------------------------------------------------------------------------

def clean_sql_query(sql_text: str) -> str:
    """Strips markdown code fences, trailing semicolons, and excess whitespace."""
    s = sql_text.strip()
    s = re.sub(r"^```(?:sql)?", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"```$", "", s).strip()
    return s.rstrip(";")


def execute_sqlite_query(query, db_path, timeout=3.0):
    """Read-only execution with a SQLite VM deadline (not only a lock timeout)."""
    from src.evaluation.metrics import execute_query_safely
    ok, rows = execute_query_safely(clean_sql_query(query), db_path, timeout=timeout)
    return (True, rows, None) if ok else (False, None, str(rows))


def compute_exec_reward(
    candidate_sql: str,
    gold_sql: str,
    db_path: str,
    timeout: float = 3.0
) -> float:
    """
    Evaluates SQLite execution:
      - 0.0: Syntax or runtime execution error on SQLite.
      - 1.0: Executes cleanly without error, but results do not match gold query.
      - 2.0: Executes cleanly and result set matches gold query 100%.
    """
    cand_ok, cand_rows, _ = execute_sqlite_query(candidate_sql, db_path, timeout=timeout)
    if not cand_ok:
        return 0.0

    gold_ok, gold_rows, _ = execute_sqlite_query(gold_sql, db_path, timeout=timeout)
    if not gold_ok:
        # If gold query itself cannot run, treat clean candidate execution as valid SQL (1.0)
        return 1.0

    from src.evaluation.metrics import compare_sql_results
    if compare_sql_results(cand_rows, gold_rows,
            preserve_order=bool(re.search(r"\bORDER\s+BY\b", gold_sql, re.I))):
        return 2.0

    return 1.0


# ---------------------------------------------------------------------------
# 3. Reward Head 3: Memory / Thought Matching Reward (R_mem)
# ---------------------------------------------------------------------------

def compute_mem_reward(
    candidate_think: str,
    gold_think: Optional[str] = None,
    reasoning_bank: Optional[Union[List[str], str]] = None,
    **kwargs: Any
) -> float:
    """
    Soft reward matching candidate thought reasoning against Reasoning Bank reference.
    Computes token-level Jaccard similarity and schema-entity coverage.
    Supports candidate strings with or without <think> tags.
    
    Returns:
        Float in [0.0, 1.0].
    """
    if "<think>" in candidate_think:
        m = re.search(r"<think>(.*?)</think>", candidate_think, re.DOTALL)
        cand_text = m.group(1).strip() if m else candidate_think
    else:
        cand_text = candidate_think

    if gold_think is not None:
        gold_text = gold_think
    elif reasoning_bank is not None:
        gold_text = "\n".join(reasoning_bank) if isinstance(reasoning_bank, list) else str(reasoning_bank)
    else:
        gold_text = ""

    if not cand_text or not gold_text:
        return 0.0

    # Normalize words
    def tokenize(text: str) -> set:
        words = re.findall(r"\b[A-Za-z0-9_]+\b", text.lower())
        return set(words)

    c_words = tokenize(cand_text)
    g_words = tokenize(gold_text)

    if not c_words or not g_words:
        return 0.0

    intersection = len(c_words & g_words)
    union = len(c_words | g_words)
    jaccard = intersection / max(union, 1)

    # Key SQL reasoning concepts: table names, join, where, group, order, sum, avg, count
    sql_keywords = {"join", "where", "group", "order", "sum", "avg", "count", "max", "min", "filter", "select", "bridge"}
    g_keys = g_words & sql_keywords
    if g_keys:
        key_recall = len(c_words & g_keys) / len(g_keys)
    else:
        key_recall = jaccard

    # Combined soft score
    score = 0.5 * jaccard + 0.5 * key_recall
    return round(min(max(score, 0.0), 1.0), 4)


# ---------------------------------------------------------------------------
# 4. Reward Head 4: Atomic AST Clause Matching Reward (R_atomic)
# ---------------------------------------------------------------------------

def extract_sql_clauses(sql: str) -> Dict[str, set]:
    """
    Deconstructs SQL into atomic clauses: SELECT, FROM, WHERE, GROUP BY, ORDER BY.
    """
    # SQLGlot supplies actual clause boundaries, including nested queries.
    # The lexical fallback remains only for dependency-free smoke tests.
    try:
        import sqlglot
        tree = sqlglot.parse_one(clean_sql_query(sql), read="sqlite")
        if tree is None:
            return {key: set() for key in ["SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY"]}
        result = {}
        nodes = {"SELECT": tree.args.get("expressions", []),
                 "FROM": [tree.args.get("from_") or tree.args.get("from")] + tree.args.get("joins", []),
                 "WHERE": [tree.args.get("where")], "GROUP BY": [tree.args.get("group")],
                 "ORDER BY": [tree.args.get("order")]}
        for key, values in nodes.items():
            result[key] = set()
            for node in values:
                if node is not None:
                    result[key].update(re.findall(r"\b[A-Z0-9_\.]+\b", node.sql(dialect="sqlite", normalize=True).upper()))
        return result
    except ImportError:
        pass
    except Exception:
        return {key: set() for key in ["SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY"]}
    cleaned = clean_sql_query(sql).upper()
    clauses = {
        "SELECT": set(),
        "FROM": set(),
        "WHERE": set(),
        "GROUP BY": set(),
        "ORDER BY": set()
    }

    # Regex patterns for clause boundaries
    pattern = r"(SELECT|FROM|WHERE|GROUP BY|ORDER BY|LIMIT|HAVING)\b"
    splits = re.split(pattern, cleaned)
    
    current_clause = None
    for part in splits:
        p = part.strip()
        if not p:
            continue
        if p in clauses:
            current_clause = p
        elif current_clause and current_clause in clauses:
            # Tokenize clause elements
            tokens = set(re.findall(r"\b[A-Z0-9_\.]+\b", p))
            clauses[current_clause].update(tokens)

    return clauses


def compute_atomic_reward(candidate_sql: str, gold_sql: str) -> float:
    """
    Evaluates atomic AST clause matching between candidate SQL and gold SQL.
    Computes overlap across SELECT, FROM, WHERE, GROUP BY, and ORDER BY clauses.

    Returns:
        Float in [0.0, 1.0].
    """
    if not candidate_sql or not gold_sql:
        return 0.0

    try:
        import sqlglot
        if sqlglot.parse_one(clean_sql_query(candidate_sql), read="sqlite") is None:
            return 0.0
    except ImportError:
        pass  # Explicit smoke mode can operate without the AST dependency.
    except Exception:
        return 0.0
    c_clauses = extract_sql_clauses(candidate_sql)
    g_clauses = extract_sql_clauses(gold_sql)

    clause_scores = []
    for clause_name in ["SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY"]:
        c_set = c_clauses.get(clause_name, set())
        g_set = g_clauses.get(clause_name, set())

        if not c_set and not g_set:
            # Neither query uses this clause -> agreement
            clause_scores.append(1.0)
        elif not c_set or not g_set:
            # One query uses it, the other does not -> penalty
            clause_scores.append(0.0)
        else:
            intersection = len(c_set & g_set)
            union = len(c_set | g_set)
            jaccard = intersection / max(union, 1)
            clause_scores.append(jaccard)

    avg_score = sum(clause_scores) / len(clause_scores)
    return round(min(max(avg_score, 0.0), 1.0), 4)


# ---------------------------------------------------------------------------
# 5. Composite Reward & Group Advantage Normalization
# ---------------------------------------------------------------------------

def compute_dense_rewards(
    generation_text: str,
    gold_sql: str,
    gold_think: str,
    db_path: str
) -> Dict[str, float]:
    """
    Computes all 4 fine-grained reward heads for a single candidate generation.
    """
    r_format = compute_format_reward(generation_text)

    # Extract answer and think from generation
    cand_sql = ""
    cand_think = ""
    ans_match = re.search(r"<answer>(.*?)</answer>", generation_text, re.DOTALL)
    if ans_match:
        cand_sql = ans_match.group(1).strip()
    think_match = re.search(r"<think>(.*?)</think>", generation_text, re.DOTALL)
    if think_match:
        cand_think = think_match.group(1).strip()

    r_exec = compute_exec_reward(cand_sql, gold_sql, db_path) if cand_sql else 0.0
    r_mem = compute_mem_reward(cand_think, gold_think) if cand_think else 0.0
    r_atomic = compute_atomic_reward(cand_sql, gold_sql) if cand_sql else 0.0

    total_reward = r_format + r_exec + r_mem + r_atomic

    return {
        "r_format": r_format,
        "r_exec": r_exec,
        "r_mem": r_mem,
        "r_atomic": r_atomic,
        "total_reward": round(total_reward, 4)
    }


def normalize_group_advantages(
    rewards: List[float],
    eps: float = 1e-8,
    epsilon: Optional[float] = None,
    **kwargs: Any
) -> List[float]:
    """
    Normalizes rewards across a group of G sampled rollouts to resolve Advantage Collapse.
    
    Formula:
      mean_R = sum(rewards) / G
      std_R = sqrt(sum((r - mean_R)^2) / G) + eps
      A_j = (r_j - mean_R) / std_R
    """
    effective_eps = epsilon if epsilon is not None else eps
    g = len(rewards)
    if g == 0:
        return []
    if g == 1:
        return [0.0]

    mean_r = sum(rewards) / g
    var_r = sum((r - mean_r) ** 2 for r in rewards) / g
    if effective_eps <= 0 or not all(math.isfinite(r) for r in rewards):
        raise ValueError("Advantages require finite rewards and positive epsilon")
    if var_r <= effective_eps ** 2:
        return [0.0] * g
    std_r = math.sqrt(var_r) + effective_eps

    advantages = [(r - mean_r) / std_r for r in rewards]
    return advantages


def compute_grpo_loss(
    logprobs: List[float],
    ref_logprobs: List[float],
    advantages: List[float],
    clip_eps: float = 0.2,
    beta_kl: float = 0.04,
    old_logprobs: Optional[List[float]] = None
) -> float:
    """
    Computes GRPO clipped policy surrogate loss with KL penalty.
    
    L_GRPO = - 1/G * sum( min( ratio * A, clip(ratio, 1-eps, 1+eps) * A ) ) + beta * D_KL
    """
    g = len(logprobs)
    if g == 0:
        return 0.0

    if not (len(ref_logprobs) == len(advantages) == g) or (old_logprobs is not None and len(old_logprobs) != g):
        raise ValueError("Policy, old-policy, reference, and advantage counts must match")
    losses = []
    old = old_logprobs if old_logprobs is not None else list(logprobs)
    for lp, ref_lp, old_lp, adv in zip(logprobs, ref_logprobs, old, advantages):
        # PPO ratio uses the sampled OLD policy, never the frozen reference.
        ratio = math.exp(lp - old_lp)
        surr1 = ratio * adv
        surr2 = max(min(ratio, 1.0 + clip_eps), 1.0 - clip_eps) * adv
        policy_loss = -min(surr1, surr2)
        
        # KL divergence approx: ref_lp - lp or exp(ref_lp - lp) - (ref_lp - lp) - 1
        kl_div = math.exp(ref_lp - lp) - (ref_lp - lp) - 1.0
        step_loss = policy_loss + beta_kl * max(kl_div, 0.0)
        losses.append(step_loss)

    return sum(losses) / g


# ---------------------------------------------------------------------------
# 6. DenseGRPOTrainer Class & Dry Run Mode
# ---------------------------------------------------------------------------

class DenseGRPOTrainer:
    """
    Dense GRPO Trainer coordinating multi-candidate rollouts,
    4-head reward computation, advantage normalization, and policy update.
    """
    def __init__(
        self,
        model_name_or_path: str = "FINER-SQL-3B",
        group_size: int = 4,
        clip_eps: float = 0.2,
        beta_kl: float = 0.04,
        learning_rate: float = 1e-6
    ):
        self.model_name_or_path = model_name_or_path
        self.group_size = group_size
        self.clip_eps = clip_eps
        self.beta_kl = beta_kl
        self.learning_rate = learning_rate

    def evaluate_group(
        self,
        candidates: List[str],
        gold_sql: str,
        gold_think: str,
        db_path: str
    ) -> Dict[str, Any]:
        """
        Evaluates a group of G candidates, computes 4 rewards per candidate,
        normalizes group advantages, and computes policy loss.
        """
        evaluations = []
        raw_rewards = []

        for cand in candidates:
            res = compute_dense_rewards(cand, gold_sql, gold_think, db_path)
            evaluations.append(res)
            raw_rewards.append(res["total_reward"])

        advantages = normalize_group_advantages(raw_rewards)

        # Check if advantage collapse occurred
        variance = sum((r - (sum(raw_rewards) / len(raw_rewards))) ** 2 for r in raw_rewards)
        has_variance = variance > 1e-6

        return {
            "evaluations": evaluations,
            "raw_rewards": raw_rewards,
            "advantages": advantages,
            "advantage_collapse_prevented": has_variance
        }


def run_dry_run(
    mini_bird_path: str = "data/mini/mini_bird.jsonl",
    db_dir: str = "data/mini",
    group_size: int = 4
) -> Dict[str, Any]:
    """
    Executes dry run verification on mini dataset:
    1. Loads sample queries from data/mini/mini_bird.jsonl.
    2. Builds 4 candidate rollouts per query:
       - Cand 0: Gold SQL with perfect <think> and <answer> (HighScore).
       - Cand 1: Valid SQL with wrong WHERE condition (PartialScore).
       - Cand 2: SQL with syntax error (LowScore).
       - Cand 3: Corrupt formatting missing </answer> (ZeroFormat).
    3. Evaluates all 4 reward heads against actual SQLite databases.
    4. Normalizes group advantages and verifies Advantage Collapse is solved.
    5. Computes GRPO surrogate loss.
    """
    logger.info(f"[DryRun] Starting Dense GRPO verification using '{mini_bird_path}'...")
    if not os.path.exists(mini_bird_path):
        raise FileNotFoundError(f"Mini dataset not found: {mini_bird_path}")

    queries = []
    with open(mini_bird_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    logger.info(f"[DryRun] Loaded {len(queries)} queries for GRPO testing.")
    trainer = DenseGRPOTrainer(group_size=group_size)

    verified_queries = 0
    all_advantages = []

    for q_idx, item in enumerate(queries[:3]):
        db_id = item["db_id"]
        db_path = os.path.join(db_dir, f"{db_id}.sqlite")
        gold_sql = item["SQL"]
        gold_think = item.get("evidence", "Analyze schema and join matching foreign keys.")

        # Construct group of 4 candidate rollouts with diverse qualities
        cands = [
            # Candidate 0: Perfect gold format and execution
            f"<think>1. Target query: {gold_think}. 2. Execute joins carefully.</think>\n<answer>\n{gold_sql}\n</answer>",
            # Candidate 1: Syntactically valid query on same DB but different result
            f"<think>1. Filter data differently.</think>\n<answer>\nSELECT * FROM ( {gold_sql} ) LIMIT 0;\n</answer>",
            # Candidate 2: Syntax error
            f"<think>1. Missing syntax.</think>\n<answer>\nSELECT FROM NON_EXISTENT_TABLE;\n</answer>",
            # Candidate 3: Corrupted formatting
            f"Here is SQL: {gold_sql}"
        ]

        group_res = trainer.evaluate_group(cands, gold_sql, gold_think, db_path)
        advantages = group_res["advantages"]
        all_advantages.append(advantages)

        # Confirm that advantages are not all zero (Advantage Collapse prevented)
        assert group_res["advantage_collapse_prevented"], f"Advantage collapse occurred in query {q_idx}!"
        assert max(advantages) > 0.0 and min(advantages) < 0.0, "Advantage normalization failed to produce positive and negative signals!"

        # Simulated logprobs
        logprobs = [-0.1, -0.8, -1.5, -2.0]
        ref_logprobs = [-0.2, -0.7, -1.4, -1.9]
        loss = compute_grpo_loss(logprobs, ref_logprobs, advantages)

        verified_queries += 1
        logger.info(
            f"[DryRun] Query {q_idx+1} ({db_id}): Rewards={group_res['raw_rewards']}, "
            f"Advantages={advantages}, GRPO Loss={loss:.4f}"
        )

    logger.info(f"[DryRun] ✅ Verified {verified_queries} queries. Advantage collapse successfully prevented across all groups.")
    return {
        "status": "PASSED",
        "verified_queries": verified_queries,
        "group_size": group_size,
        "advantage_collapse_prevented": True,
        "reward_heads": ["r_format", "r_exec", "r_mem", "r_atomic"]
    }


def completion_token_logprobs(model, token_ids, prompt_length, temperature=1.0):
    """Return only sampled response-token log probabilities, retaining gradients."""
    outputs = model(input_ids=token_ids, attention_mask=torch.ones_like(token_ids), use_cache=False)
    logits = outputs.logits[:, prompt_length - 1:-1, :].float() / temperature
    targets = token_ids[:, prompt_length:]
    # logsumexp avoids retaining a full vocabulary-sized softmax result.
    selected = logits.gather(-1, targets[..., None]).squeeze(-1) - logits.logsumexp(-1)
    return selected.squeeze(0)


def grpo_token_objective(new_logprobs, old_logprobs, reference_logprobs, advantage,
                         clip_eps=0.2, beta_kl=0.04):
    """Differentiable clipped tokenwise GRPO with k3 reference KL estimator.

    Old-policy probabilities are sampled before ANY group optimizer update.
    The immutable SFT reference is used only by KL. EOS is included and padding
    excluded by passing one unpadded response at a time.
    """
    ratio = (new_logprobs - old_logprobs.detach()).exp()
    surrogate = torch.minimum(ratio * advantage,
        ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantage)
    delta = reference_logprobs.detach() - new_logprobs
    kl = delta.exp() - delta - 1
    return (-surrogate + beta_kl * kl).mean()


def _schema_for_record(record, db_path):
    if record.get("schema") or record.get("ddl_statements"):
        return record.get("schema", record.get("ddl_statements", ""))
    connection = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True)
    try:
        return ";\n".join(row[0] for row in connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL ORDER BY name"))
    finally:
        connection.close()


def _activate_adapter(model, name):
    model.set_adapter(name)
    for pname, parameter in model.named_parameters():
        if ".reference." in pname:
            parameter.requires_grad_(False)


from src.training.checkpoint_sync import persistent_training


@persistent_training('grpo_3b')
def run_grpo_training(data_path="data/mini/mini_bird.jsonl", db_dir="data/mini",
    output_dir=None, drive_dir=None, num_rollouts=4, epochs=2,
    learning_rate=5e-6, dry_run=False, **kwargs):
    """Sequential, memory-bounded policy optimization from a genuine SFT adapter.

    One shared frozen NF4 base holds trainable policy and immutable reference
    LoRA adapters. Responses are sampled from the model without gold SQL; labels
    are used exclusively for the four rewards. Zero-variance groups are skipped,
    not perturbed to manufacture advantages. Checkpoints include both adapters.
    """
    group_size = int(kwargs.get("group_size", num_rollouts))
    if dry_run:
        return dict(run_dry_run(mini_bird_path=data_path, db_dir=db_dir,
                                group_size=group_size), mode="smoke", trained=False)
    if not TORCH_AVAILABLE:
        raise RuntimeError("Real GRPO requires PyTorch/Transformers/PEFT/bitsandbytes; dry_run=True is smoke only")
    from src.training.runtime import (load_records, build_prompt, load_causal_model,
        seed_everything, resolve_db, ExperimentLogger, dataset_fingerprint,
        save_checkpoint, load_checkpoint)
    from src.training.train_sft_generator import _optimizer_update, _resolve_resume, _generation_extra
    import sqlglot  # Required for genuine atomic AST rewards, optional only in smoke mode.
    sft_checkpoint = kwargs.get("sft_checkpoint", kwargs.get("adapter_path"))
    initial_resume = kwargs.get("resume_from_checkpoint")
    if initial_resume and (Path(initial_resume) / "reference" / "adapter_config.json").is_file():
        sft_checkpoint = str(Path(initial_resume) / "reference")
    if not sft_checkpoint or not (Path(sft_checkpoint) / "adapter_config.json").is_file():
        raise ValueError("GRPO requires sft_checkpoint pointing to an exported SFT PEFT adapter")
    output_dir = output_dir or "/kaggle/working/grpo_3b"
    device = kwargs.get("device", "cuda:0")
    max_steps = int(kwargs.get("max_steps", 100))
    save_steps = int(kwargs.get("save_steps", 25))
    accum = int(kwargs.get("gradient_accumulation_steps", 1))
    max_length = int(kwargs.get("max_length", 1024))
    max_new_tokens = int(kwargs.get("max_new_tokens", 128))
    seed = int(kwargs.get("seed", 42))
    clip_eps = float(kwargs.get("clip_eps", 0.2))
    beta_kl = float(kwargs.get("beta_kl", 0.04))
    temperature = float(kwargs.get("temperature", 1.0))
    if min(max_steps, save_steps, accum, max_new_tokens, epochs) <= 0 or group_size < 2:
        raise ValueError("GRPO needs positive counts and at least two sampled responses per group")
    if max_length <= max_new_tokens or not math.isfinite(temperature) or temperature <= 0 or beta_kl < 0 or not 0 < clip_eps < 1:
        raise ValueError("Invalid GRPO length, temperature, KL, or clipping configuration")
    seed_everything(seed)
    rows = load_records(data_path)
    resume = _resolve_resume(kwargs.get("resume_from_checkpoint", kwargs.get("resume")), output_dir)
    model, tokenizer = load_causal_model(kwargs.get("model_name_or_path", "meta-llama/Llama-3.2-3B-Instruct"),
        adapter_path=resume or sft_checkpoint, training=True, device=device,
        rank=int(kwargs.get("lora_rank", 16)))
    reference_source = Path(resume) / "reference" if resume and (Path(resume) / "reference").exists() else Path(sft_checkpoint)
    model.load_adapter(str(reference_source), adapter_name="reference", is_trainable=False)
    _activate_adapter(model, "default")
    # Keep sampled and optimized policies identical even if a backbone has dropout.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"))
    fingerprint = dataset_fingerprint(data_path)
    reference_weights = next((reference_source / name for name in
        ("adapter_model.safetensors", "adapter_model.bin") if (reference_source / name).exists()), None)
    if reference_weights is None:
        raise FileNotFoundError("Missing reference SFT adapter weights")
    contract = dict(group_size=group_size, accumulation=accum, max_length=max_length,
        max_new_tokens=max_new_tokens, seed=seed, beta_kl=beta_kl, clip_eps=clip_eps,
        temperature=temperature, learning_rate=learning_rate,
        reference_sha256=dataset_fingerprint(reference_weights))
    logger = ExperimentLogger(output_dir, dict(stage="grpo", mode="train", **contract))
    step, first_epoch, first_index = 0, 0, 0
    if resume:
        state = load_checkpoint(resume, model, optimizer, scaler=scaler,
            expected_extra=dict(kind='grpo', dataset_fingerprint=fingerprint, contract=contract))
        extra = state["extra"]
        if extra.get("kind") != "grpo" or extra.get("dataset_fingerprint") != fingerprint or extra.get("contract") != contract:
            raise ValueError("Resume checkpoint belongs to another GRPO dataset/configuration")
        step, first_epoch, first_index = state["step"], extra["epoch"], extra["batch_index"]
    optimizer.zero_grad(set_to_none=True)
    checkpoint = resume
    last_saved_step = step if resume else None
    skipped_zero, skipped_long, pending = 0, 0, 0
    epoch, next_index = first_epoch, first_index
    for epoch in range(first_epoch, int(epochs)):
        order = list(range(len(rows)))
        random.Random(seed + epoch).shuffle(order)
        for record_index in range(first_index if epoch == first_epoch else 0, len(order)):
            if step >= max_steps:
                break
            item = rows[order[record_index]]
            db_path = resolve_db(db_dir, item["db_id"])
            schema = _schema_for_record(item, db_path)
            prompt = build_prompt(item["question"], schema)
            prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"].to(device)
            next_index = record_index + 1
            if prompt_ids.shape[1] > max_length - max_new_tokens:
                skipped_long += 1
                logger.log(event="skip_overlength", epoch=epoch, record_index=record_index)
                continue
            samples, rewards = [], []
            _activate_adapter(model, "default")
            model.eval()
            for _ in range(group_size):
                with torch.no_grad():
                    generated = model.generate(input_ids=prompt_ids,
                        attention_mask=torch.ones_like(prompt_ids), do_sample=True,
                        temperature=temperature, top_p=1.0, top_k=0,
                        max_new_tokens=max_new_tokens, use_cache=True,
                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                    # Slice prompt away before decoding; no question/schema in reward text.
                    response = tokenizer.decode(generated[0, prompt_ids.shape[1]:], skip_special_tokens=True)
                    old = completion_token_logprobs(model, generated, prompt_ids.shape[1], temperature)
                    _activate_adapter(model, "reference")
                    ref = completion_token_logprobs(model, generated, prompt_ids.shape[1], temperature)
                    _activate_adapter(model, "default")
                reward = compute_dense_rewards(response, item.get("SQL", item.get("answer", item.get("sql", ""))),
                    item.get("think_steps", item.get("reasoning", item.get("evidence", ""))), db_path)
                rewards.append(reward)
                samples.append((generated.detach().cpu(), old.cpu(), ref.cpu()))
                del generated, old, ref
            advantages = normalize_group_advantages([r["total_reward"] for r in rewards])
            if not any(advantages):
                skipped_zero += 1
                logger.log(event="skip_zero_variance", step=step, epoch=epoch,
                           rewards=[r["total_reward"] for r in rewards])
            else:
                model.train()
                group_loss = 0.0
                for (ids, old, ref), advantage in zip(samples, advantages):
                    with torch.autocast(device_type="cuda" if str(device).startswith("cuda") else "cpu",
                            dtype=torch.float16 if str(device).startswith("cuda") else torch.bfloat16,
                            enabled=str(device).startswith("cuda")):
                        new = completion_token_logprobs(model, ids.to(device), prompt_ids.shape[1], temperature)
                        loss = grpo_token_objective(new, old.to(device), ref.to(device), advantage,
                                                    clip_eps, beta_kl) / group_size
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite GRPO objective")
                    scaler.scale(loss).backward()
                    group_loss += float(loss.detach())
                    del new, loss
                pending += 1
                logger.log(step=step, epoch=epoch, group_loss=group_loss,
                    rewards=[r["total_reward"] for r in rewards], advantages=advantages,
                    **{head: sum(r[head] for r in rewards) / group_size for head in
                       ("r_format", "r_exec", "r_mem", "r_atomic")})
            del samples, prompt_ids
            if pending and (pending == accum or next_index == len(order)):
                updated = _optimizer_update(parameters, optimizer, scaler, pending)
                pending = 0
                if updated:
                    step += 1
                logger.log(event="optimizer_update", step=step, optimizer_updated=updated,
                    peak_vram_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else 0)
                if updated and step % save_steps == 0:
                    checkpoint = save_checkpoint(output_dir, model, optimizer, step, scaler=scaler,
                        tokenizer=tokenizer, extra=_generation_extra("grpo", fingerprint,
                            epoch + int(next_index == len(order)), 0 if next_index == len(order) else next_index,
                            contract=contract))
                    last_saved_step = step
        # Flush a partial window even when the last records were overlength/skipped.
        if pending:
            if _optimizer_update(parameters, optimizer, scaler, pending):
                step += 1
            pending = 0
        if step >= max_steps:
            break
        next_index = len(order)
    if step != last_saved_step:
        checkpoint = save_checkpoint(output_dir, model, optimizer, step, scaler=scaler,
            tokenizer=tokenizer, extra=_generation_extra("grpo", fingerprint,
                epoch + int(next_index == len(order)), 0 if next_index == len(order) else next_index,
                            contract=contract))
    result = dict(status="COMPLETED" if step > 0 else "NO_POLICY_UPDATES", mode="train",
        trained=step > 0, optimizer_steps=step, checkpoint_path=checkpoint,
        skipped_zero_variance=skipped_zero, skipped_overlength=skipped_long)
    logger.log(event="finished", **result)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Genuine sampled Dense GRPO")
    parser.add_argument("--model_name_or_path", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--sft_checkpoint")
    parser.add_argument("--data_path", default="data/mini/mini_bird.jsonl")
    parser.add_argument("--db_dir", default="data/mini")
    parser.add_argument("--output_dir", default="/kaggle/working/grpo_3b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--beta_kl", type=float, default=0.04)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--save_steps", type=int, default=25)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    print(json.dumps(run_grpo_training(**vars(parse_args())), indent=2))


if __name__ == "__main__":
    main()
