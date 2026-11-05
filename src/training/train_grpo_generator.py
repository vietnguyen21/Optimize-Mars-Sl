"""
Dense GRPO (Group Relative Policy Optimization) Trainer for Text-to-SQL SLMs.

Implements FINER-SQL Dense GRPO with 4 fine-grained reward heads:
  1. R_format in {0, 1}: Binary reward for well-formed <think>...</think><answer>...</answer> tags.
  2. R_exec in {0, 1, 2}: Execution reward on target SQLite database (0: error, 1: valid run, 2: 100% gold match).
  3. R_mem in [0, 1]: Soft semantic thought-matching reward against Reasoning Bank patterns.
  4. R_atomic in [0, 1]: AST clause matching reward (SELECT, FROM, WHERE, GROUP BY, ORDER BY).

Resolves Advantage Collapse by ensuring non-zero variance and normalized group advantages:
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


def execute_sqlite_query(query: str, db_path: str, timeout: float = 3.0) -> Tuple[bool, Any, Optional[str]]:
    """
    Executes an SQL query safely against a SQLite database.
    
    Returns:
        Tuple of (success_bool, result_set_or_empty, error_message_or_None)
    """
    if not os.path.exists(db_path):
        return False, None, f"Database file not found: {db_path}"

    cleaned = clean_sql_query(query)
    if not cleaned:
        return False, None, "Empty SQL query"

    # Enforce read-only execution
    forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "ATTACH", "DETACH"]
    first_word = cleaned.split()[0].upper() if cleaned.split() else ""
    if first_word in forbidden:
        return False, None, f"Forbidden non-query statement: {first_word}"

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=timeout)
        cursor = conn.cursor()
        cursor.execute(cleaned)
        rows = cursor.fetchall()
        conn.close()
        return True, rows, None
    except Exception as e:
        return False, None, str(e)


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

    # Compare result sets
    if cand_rows == gold_rows:
        return 2.0

    # Compare unordered multisets if ordering is not strictly required
    try:
        if sorted(cand_rows) == sorted(gold_rows):
            return 2.0
    except Exception:
        pass

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
    std_r = math.sqrt(var_r) + effective_eps

    advantages = [(r - mean_r) / std_r for r in rewards]
    return [round(a, 4) for a in advantages]


def compute_grpo_loss(
    logprobs: List[float],
    ref_logprobs: List[float],
    advantages: List[float],
    clip_eps: float = 0.2,
    beta_kl: float = 0.04
) -> float:
    """
    Computes GRPO clipped policy surrogate loss with KL penalty.
    
    L_GRPO = - 1/G * sum( min( ratio * A, clip(ratio, 1-eps, 1+eps) * A ) ) + beta * D_KL
    """
    g = len(logprobs)
    if g == 0:
        return 0.0

    losses = []
    for lp, ref_lp, adv in zip(logprobs, ref_logprobs, advantages):
        # Ratio w = exp(lp - ref_lp)
        ratio = math.exp(lp - ref_lp)
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


def grpo_token_loss(
    new_logprobs: Any,
    old_logprobs: Any,
    ref_logprobs: Any,
    advantage: float,
    mask: Any,
    clip_eps: float = 0.2,
    beta_kl: float = 0.04
) -> Any:
    """Per-token GRPO loss for one completion: clipped surrogate plus KL to the reference policy."""
    ratio = torch.exp(new_logprobs - old_logprobs)
    surrogate = torch.min(ratio * advantage, torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage)
    log_ref_ratio = ref_logprobs - new_logprobs
    kl = torch.exp(log_ref_ratio) - log_ref_ratio - 1.0
    per_token = -surrogate + beta_kl * kl
    return (per_token * mask).sum() / mask.sum().clamp(min=1.0)


def completion_logprobs(model: Any, input_ids: Any, prompt_len: int) -> Any:
    """Log-probability of each completion token; input_ids is one prompt + completion, shape (1, L)."""
    logits = model(input_ids=input_ids).logits[0, prompt_len - 1:-1]
    targets = input_ids[0, prompt_len:]
    return torch.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def sample_group(model: Any, tokenizer: Any, prompt: str, group_size: int, max_prompt_tokens: int,
                 max_new_tokens: int, temperature: float, top_p: float) -> Tuple[Any, List[Any], List[str]]:
    """Samples G completions. Returns prompt ids, completion id tensors cut at EOS, decoded texts."""
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"][-max_prompt_tokens:]
    prompt_tensor = torch.tensor([prompt_ids], device=next(model.parameters()).device)
    model.eval()
    with torch.no_grad():
        sequences = model.generate(
            input_ids=prompt_tensor,
            attention_mask=torch.ones_like(prompt_tensor),
            max_new_tokens=max_new_tokens,
            num_return_sequences=group_size,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            use_cache=True
        )
    eos_ids = model.generation_config.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    completions, texts = [], []
    for seq in sequences:
        ids = seq[len(prompt_ids):].tolist()
        for pos, token in enumerate(ids):
            if token in eos_ids:
                ids = ids[:pos + 1]
                break
        completions.append(torch.tensor(ids, device=prompt_tensor.device))
        texts.append("<think>" + tokenizer.decode(ids, skip_special_tokens=True))
    return prompt_tensor, completions, texts


def select_signal_records(
    model: Any,
    tokenizer: Any,
    records: List[Dict[str, Any]],
    db_dir: str,
    output_path: str,
    num_samples: int = 8,
    max_prompt_tokens: int = 2048,
    max_new_tokens: int = 256
) -> List[Dict[str, Any]]:
    """
    Keeps questions the model answers correctly only some of the time: all-right and all-wrong
    groups give almost no learning signal. Appends per question, so it resumes after a disconnect.
    """
    from src.evaluation.candidate_generator import build_prompt
    from src.evaluation.evaluate_end_to_end import load_schema_ddl

    done_path = output_path + ".done"
    start = int(open(done_path).read()) if os.path.exists(done_path) else 0
    for idx in range(start, len(records)):
        item = records[idx]
        db_path = os.path.join(db_dir, f"{item['db_id']}.sqlite")
        prompt = build_prompt(item["question"], item.get("evidence", ""), load_schema_ddl(db_path))
        _, _, texts = sample_group(model, tokenizer, prompt, num_samples, max_prompt_tokens, max_new_tokens, 0.8, 0.95)
        correct = sum(compute_dense_rewards(t, item["SQL"], "", db_path)["r_exec"] == 2.0 for t in texts)
        if 0 < correct < num_samples:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        with open(done_path, "w") as f:
            f.write(str(idx + 1))
    with open(output_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def train_grpo_loop(
    model: Any,
    tokenizer: Any,
    records: List[Dict[str, Any]],
    db_dir: str,
    state_dir: str,
    group_size: int = 4,
    epochs: int = 1,
    learning_rate: float = 5e-6,
    clip_eps: float = 0.2,
    beta_kl: float = 0.04,
    max_prompt_tokens: int = 2048,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_grad_norm: float = 1.0,
    save_every: int = 100
) -> Dict[str, Any]:
    """
    One optimizer step per question. `model` is a PeftModel over the frozen SFT policy, so
    disable_adapter() gives the reference policy for the KL term. Single-turn rollouts.
    shortcut: no ReAct turns (the upgrade plan allows up to 2); add them after single-turn works.
    shortcut: rollouts are unconstrained until XGrammar decoding exists; bad SQL scores 0 instead.
    """
    from src.evaluation.candidate_generator import build_prompt
    from src.evaluation.evaluate_end_to_end import load_schema_ddl

    os.makedirs(state_dir, exist_ok=True)
    device_type = next(model.parameters()).device.type
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate)

    position = 0
    opt_path = os.path.join(state_dir, "optimizer.pt")
    pos_path = os.path.join(state_dir, "position.json")
    if os.path.exists(pos_path):
        with open(pos_path) as f:
            position = json.load(f)["position"]
        optimizer.load_state_dict(torch.load(opt_path, map_location=device_type))
        logger.info(f"Resuming GRPO at position {position}")

    metrics_path = os.path.join(state_dir, "metrics.jsonl")
    total = len(records) * epochs
    skipped = updates = 0
    reward_sum = 0.0

    def save_state(next_position: int) -> None:
        model.save_pretrained(os.path.join(state_dir, "adapter"))
        torch.save(optimizer.state_dict(), opt_path)
        with open(pos_path, "w") as f:
            json.dump({"position": next_position}, f)

    for step in range(position, total):
        item = records[step % len(records)]
        db_path = os.path.join(db_dir, f"{item['db_id']}.sqlite")
        prompt = build_prompt(item["question"], item.get("evidence", ""), load_schema_ddl(db_path))

        with torch.autocast(device_type, dtype=amp_dtype):
            prompt_tensor, completions, texts = sample_group(
                model, tokenizer, prompt, group_size, max_prompt_tokens, max_new_tokens, temperature, top_p)

        rewards = [compute_dense_rewards(t, item["SQL"], item.get("evidence", ""), db_path) for t in texts]
        totals = [r["total_reward"] for r in rewards]
        advantages = normalize_group_advantages(totals)
        reward_sum += sum(totals) / len(totals)
        record = {"step": step, "mean_reward": sum(totals) / len(totals),
                  "exec_correct": sum(r["r_exec"] == 2.0 for r in rewards)}

        if max(totals) - min(totals) < 1e-6:  # identical rewards: zero advantage, nothing to learn
            skipped += 1
        else:
            prompt_len = prompt_tensor.shape[1]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for completion, advantage in zip(completions, advantages):
                input_ids = torch.cat([prompt_tensor[0], completion]).unsqueeze(0)
                with torch.autocast(device_type, dtype=amp_dtype):
                    with torch.no_grad():
                        old_lp = completion_logprobs(model, input_ids, prompt_len)
                        with model.disable_adapter():
                            ref_lp = completion_logprobs(model, input_ids, prompt_len)
                    new_lp = completion_logprobs(model, input_ids, prompt_len)
                    loss = grpo_token_loss(new_lp, old_lp, ref_lp, advantage, torch.ones_like(new_lp),
                                           clip_eps, beta_kl) / group_size
                loss.backward()
                loss_value += loss.item()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            if not math.isfinite(loss_value) or not torch.isfinite(grad_norm):
                raise RuntimeError(f"Non-finite GRPO loss or gradient at step {step}")
            optimizer.step()
            updates += 1
            record.update({"loss": loss_value, "grad_norm": float(grad_norm)})

        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        if (step + 1) % save_every == 0:
            save_state(step + 1)

    save_state(total)
    return {"steps": total - position, "updates": updates, "skipped_no_variance": skipped,
            "mean_reward": reward_sum / max(1, total - position)}


def run_grpo_training(
    data_path: str = "data/mini/mini_bird.jsonl",
    db_dir: str = "data/mini",
    output_dir: Optional[str] = None,
    drive_dir: Optional[str] = None,
    num_rollouts: int = 4,
    epochs: int = 1,
    learning_rate: float = 5e-6,
    dry_run: bool = True,
    model_name_or_path: Optional[str] = None,
    beta_kl: float = 0.04,
    clip_eps: float = 0.2,
    lora_rank: int = 64,
    signal_samples: int = 8,
    max_new_tokens: int = 256,
    save_every: int = 100,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Dense GRPO after JEPA-SFT. dry_run=True runs the offline verification only.
    model_name_or_path is the SFT checkpoint. The JEPA loss stays off here. signal_samples=0 trains
    on every question. State lives in drive_dir (or output_dir), so a rerun resumes. The result
    is a full merged model in <output_dir>/best_model, loadable by the evaluator.
    """
    group_size = kwargs.get("group_size", num_rollouts)
    if dry_run:
        return run_dry_run(mini_bird_path=data_path, db_dir=db_dir, group_size=group_size)
    if not model_name_or_path or not output_dir:
        raise ValueError("Full GRPO needs model_name_or_path (SFT checkpoint) and output_dir")

    from peft import PeftModel, get_peft_model
    from src.evaluation.candidate_generator import load_generator
    from src.training.train_sft_generator import build_lora_config

    base, tokenizer = load_generator(model_name_or_path)
    state_dir = drive_dir or output_dir
    os.makedirs(state_dir, exist_ok=True)
    with open(data_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if signal_samples:
        records = select_signal_records(base, tokenizer, records, db_dir,
                                        os.path.join(state_dir, "signal.jsonl"), signal_samples,
                                        max_new_tokens=max_new_tokens)
        logger.info(f"GRPO signal set: {len(records)} questions")
        if not records:
            raise RuntimeError("No question has partially correct samples; GRPO has nothing to learn from")

    base.config.use_cache = False
    base.enable_input_require_grads()
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    adapter_dir = os.path.join(state_dir, "adapter")
    if os.path.exists(adapter_dir):
        model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=True)
    else:
        # Fresh LoRA starts as a no-op, so the policy starts equal to the frozen reference.
        model = get_peft_model(base, build_lora_config(rank=lora_rank, lora_dropout=0.0))
    model.print_trainable_parameters()

    result = train_grpo_loop(model, tokenizer, records, db_dir, state_dir, group_size=group_size, epochs=epochs,
                             learning_rate=learning_rate, beta_kl=beta_kl, clip_eps=clip_eps,
                             max_new_tokens=max_new_tokens, save_every=save_every)

    best_dir = os.path.join(output_dir, "best_model")
    merged = model.merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)
    result["saved_to"] = best_dir
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Dense GRPO Trainer for Text-to-SQL SLMs")
    parser.add_argument("--model_name_or_path", type=str, default="FINER-SQL-3B")
    parser.add_argument("--data_path", type=str, default="data/mini/mini_bird.jsonl")
    parser.add_argument("--db_dir", type=str, default="data/mini")
    parser.add_argument("--group_size", type=int, default=4, help="Rollouts per question (4, 8, 16)")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--beta_kl", type=float, default=0.04)
    parser.add_argument("--dry_run", action="store_true", help="Run dry run verification on mini dataset")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--drive_dir", type=str, default=None, help="Persistent state directory for resume")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--signal_samples", type=int, default=8, help="0 disables the signal filter")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--save_every", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dry_run:
        result = run_dry_run(mini_bird_path=args.data_path, db_dir=args.db_dir, group_size=args.group_size)
        print(json.dumps(result, indent=2))
        sys.exit(0)

    result = run_grpo_training(
        data_path=args.data_path, db_dir=args.db_dir, output_dir=args.output_dir, drive_dir=args.drive_dir,
        num_rollouts=args.group_size, epochs=args.epochs, learning_rate=args.learning_rate, dry_run=False,
        model_name_or_path=args.model_name_or_path, beta_kl=args.beta_kl, clip_eps=args.clip_eps,
        signal_samples=args.signal_samples, max_new_tokens=args.max_new_tokens, save_every=args.save_every
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
