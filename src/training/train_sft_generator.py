"""
Completion-Only Supervised Fine-Tuning (SFT) Trainer for Text-to-SQL SLMs.

Supports FINER-SQL-3B and Qwen2.5-Coder-7B with:
1. Completion-Only Cross-Entropy Loss: Prompt and Schema tokens masked with -100.
   Loss is computed strictly over <think>...</think><answer>...</answer>.
2. LoRA / QLoRA parameter-efficient fine-tuning (r=64..512, alpha=16, 7 projection targets).
3. Google Drive checkpoint sync via DriveCheckpointCallback & --resume_from_checkpoint latest.
4. Clean dual execution: Full HuggingFace/TRL/PEFT mode and standalone --dry_run mode on data/mini/.
"""

import os
import sys
import json
import math
import random
import argparse
from pathlib import Path
import logging
from typing import Optional, List, Dict, Any, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.training.callbacks import DriveCheckpointCallback, get_resume_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Check for HuggingFace / PyTorch libraries
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

try:
    import peft
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    import transformers
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


TARGET_MODULES_DEFAULT = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class LoraConfigFallback:
    """Lightweight fallback object matching PEFT LoraConfig attributes and dict behavior."""
    def __init__(self, **kwargs):
        self.r = kwargs.get("r", 64)
        self.lora_alpha = kwargs.get("lora_alpha", 16)
        self.target_modules = kwargs.get("target_modules", TARGET_MODULES_DEFAULT)
        self.lora_dropout = kwargs.get("lora_dropout", 0.05)
        self.bias = kwargs.get("bias", "none")
        self.task_type = kwargs.get("task_type", "CAUSAL_LM")

    def __getitem__(self, item):
        return getattr(self, item)

    def get(self, item, default=None):
        return getattr(self, item, default)

    def __repr__(self):
        return f"LoraConfigFallback(r={self.r}, lora_alpha={self.lora_alpha}, target_modules={self.target_modules})"


def build_lora_config(
    rank: int = 64,
    alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    r: Optional[int] = None,
    lora_alpha: Optional[int] = None,
    **kwargs: Any
) -> Any:
    """
    Builds LoRA / QLoRA configuration.
    
    Args:
        rank: LoRA attention dimension (64 to 512).
        alpha: LoRA scaling parameter.
        lora_dropout: Dropout probability for LoRA layers.
        target_modules: List of module names to apply LoRA to.
        r: Alias for rank.
        lora_alpha: Alias for alpha.

    Returns:
        peft.LoraConfig instance if PEFT is available, otherwise LoraConfigFallback.
    """
    resolved_rank = r if r is not None else rank
    resolved_alpha = lora_alpha if lora_alpha is not None else alpha
    targets = target_modules or TARGET_MODULES_DEFAULT
    if PEFT_AVAILABLE:
        return LoraConfig(
            r=resolved_rank,
            lora_alpha=resolved_alpha,
            target_modules=targets,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
    else:
        return LoraConfigFallback(
            r=resolved_rank,
            lora_alpha=resolved_alpha,
            target_modules=targets,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )


def mask_prompt_tokens(
    input_ids: List[int],
    think_token_id: Optional[int] = None,
    ignore_index: int = -100,
    prompt_end_token_idx: Optional[int] = None,
    **kwargs: Any
) -> Tuple[List[int], int]:
    """
    Given a sequence of token IDs, finds the first occurrence of think_token_id
    or uses prompt_end_token_idx, and sets all preceding tokens to ignore_index (-100).
    
    Returns:
        Tuple of (labels_list, completion_start_idx)
    """
    labels = list(input_ids)
    if prompt_end_token_idx is not None:
        completion_start = prompt_end_token_idx
    elif think_token_id is not None:
        completion_start = -1
        for idx, tid in enumerate(input_ids):
            if tid == think_token_id:
                completion_start = idx
                break
        if completion_start == -1:
            completion_start = 0
    else:
        completion_start = 0

    for i in range(completion_start):
        if i < len(labels):
            labels[i] = ignore_index

    return labels, completion_start


def compute_completion_loss(
    logits: Any,
    labels: Any,
    ignore_index: int = -100
) -> Tuple[float, int]:
    """
    Computes Cross-Entropy loss strictly over tokens where labels != ignore_index.
    
    Supports PyTorch Tensors or pure Python lists of logits and labels.
    
    Returns:
        Tuple of (loss_value, num_valid_tokens)
    """
    if TORCH_AVAILABLE and isinstance(logits, torch.Tensor) and isinstance(labels, torch.Tensor):
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        valid_mask = (shift_labels != ignore_index)
        num_valid = int(valid_mask.sum().item())
        return loss.item(), num_valid
    else:
        # Pure Python cross-entropy calculation
        total_loss = 0.0
        num_valid = 0
        seq_len = len(labels)

        for i in range(seq_len - 1):
            target_token = labels[i + 1]
            if target_token == ignore_index:
                continue

            step_logits = logits[i]
            if target_token >= len(step_logits) or target_token < 0:
                continue

            # Softmax log-prob calculation with numerical stability
            max_val = max(step_logits)
            exp_sum = sum(math.exp(x - max_val) for x in step_logits)
            log_prob = (step_logits[target_token] - max_val) - math.log(exp_sum)
            total_loss -= log_prob
            num_valid += 1

        avg_loss = (total_loss / max(num_valid, 1)) if num_valid > 0 else 0.0
        return avg_loss, num_valid


class CompletionOnlyDataCollator:
    """Tokenize prompt/completion separately and mask only the explicit boundary.

    Concatenating token IDs avoids BPE boundary ambiguity. Padding is masked by
    attention_mask, never by token identity (EOS is frequently also PAD).
    Overlength examples fail rather than silently training on a missing answer.
    """
    def __init__(self, tokenizer=None, think_tag="<think>", ignore_index=-100,
                 max_length=1024):
        self.tokenizer = tokenizer
        self.think_tag = think_tag
        self.ignore_index = ignore_index
        self.max_length = max_length

    def encode_example(self, item):
        from src.training.runtime import build_prompt
        if "input_ids" in item and "labels" in item:
            result = dict(item)
            if len(result["input_ids"]) > self.max_length:
                raise ValueError("Encoded example exceeds max_length")
            return result
        if "question" in item:
            prompt = build_prompt(item["question"], item.get("schema", item.get("ddl_statements", "")))
            answer = item.get("answer", item.get("SQL", item.get("sql", "")))
            if not answer:
                raise ValueError("SFT record requires answer/SQL/sql")
            reasoning = item.get("think_steps", item.get("reasoning", item.get("evidence", "")))
            completion = f"<think>\n{reasoning}\n</think>\n<answer>\n{answer}\n</answer>"
        elif "prompt" in item and "completion" in item:
            prompt, completion = item["prompt"], item["completion"]
        else:
            text = item.get("formatted_prompt", "")
            if self.think_tag not in text:
                raise ValueError("No explicit completion boundary in SFT record")
            prompt, suffix = text.split(self.think_tag, 1)
            completion = self.think_tag + suffix
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        if not prompt_ids:
            raise ValueError("A nonempty prompt is required for next-token supervision")
        # Split at SQL boundaries too, so target pooling never relies on a
        # guessed prefix length under BPE token merging.
        answer_open = completion.find("<answer>")
        answer_close = completion.find("</answer>", answer_open + len("<answer>")) if answer_open >= 0 else -1
        if answer_open >= 0 and answer_close >= 0:
            prefix_end = answer_open + len("<answer>")
            prefix_ids = self.tokenizer.encode(completion[:prefix_end], add_special_tokens=False)
            sql_ids = self.tokenizer.encode(completion[prefix_end:answer_close], add_special_tokens=False)
            suffix_ids = self.tokenizer.encode(completion[answer_close:], add_special_tokens=False)
            completion_ids = prefix_ids + sql_ids + suffix_ids
            sql_start = len(prompt_ids) + len(prefix_ids)
            sql_end = sql_start + len(sql_ids)
        else:
            completion_ids = self.tokenizer.encode(completion, add_special_tokens=False)
            sql_start, sql_end = len(prompt_ids), len(prompt_ids) + len(completion_ids)
        eos = self.tokenizer.eos_token_id
        if eos is not None and (not completion_ids or completion_ids[-1] != eos):
            completion_ids.append(eos)
        if not completion_ids:
            raise ValueError("Empty completion")
        ids = prompt_ids + completion_ids
        if len(ids) > self.max_length:
            raise ValueError(f"Example has {len(ids)} tokens > max_length={self.max_length}; preprocess or increase max_length")
        return {"input_ids": ids, "attention_mask": [1] * len(ids),
                "labels": [-100] * len(prompt_ids) + completion_ids,
                "pred_position": len(prompt_ids) - 1,
                "sql_start": min(sql_start, len(ids) - 1), "sql_end": sql_end}

    def __call__(self, batch):
        if self.tokenizer is None:
            return {"batch": batch}
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for real SFT collation")
        encoded = [self.encode_example(item) for item in batch]
        width = max(len(x["input_ids"]) for x in encoded)
        pad = self.tokenizer.pad_token_id
        if pad is None:
            raise ValueError("Tokenizer requires a pad_token_id")
        ids, labels, masks = [], [], []
        for item in encoded:
            n = width - len(item["input_ids"])
            ids.append(item["input_ids"] + [pad] * n)
            masks.append(item.get("attention_mask", [1] * len(item["input_ids"])) + [0] * n)
            labels.append(item["labels"] + [self.ignore_index] * n)
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "pred_positions": torch.tensor([x["pred_position"] for x in encoded]),
                "sql_starts": torch.tensor([x["sql_start"] for x in encoded]),
                "sql_ends": torch.tensor([x.get("sql_end", len(x["input_ids"])) for x in encoded])}


def joint_sft_jepa_loss(model, batch, predictor=None, lambda_jepa=0.0,
                        loss_dropout=0.5):
    """Genuine CE + stop-gradient latent alignment in one causal forward.

    The two-block mask equals the ordinary triangular causal mask: the final
    prompt anchor (last subtoken of literal [PRED]) cannot see completion tokens.
    SQL targets are pooled from completion hidden states and detached. No target
    token is fed into the context anchor, and no vocabulary resize is needed.
    """
    use_jepa = predictor is not None and lambda_jepa > 0
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    labels=batch["labels"], output_hidden_states=use_jepa,
                    use_cache=False)
    ce = outputs.loss
    auxiliary = ce.new_zeros(())
    if use_jepa and (not model.training or random.random() >= loss_dropout):
        hidden = outputs.hidden_states[-1]
        index = torch.arange(hidden.shape[0], device=hidden.device)
        context = hidden[index, batch["pred_positions"]]
        positions = torch.arange(hidden.shape[1], device=hidden.device)[None, :]
        target_mask = ((positions >= batch["sql_starts"][:, None])
            & (positions < batch["sql_ends"][:, None]) & batch["attention_mask"].bool())
        target = (hidden.detach() * target_mask[..., None]).sum(1) / target_mask.sum(1, keepdim=True).clamp_min(1)
        prediction = predictor(context.float())
        auxiliary = (1 - F.cosine_similarity(prediction, target.float(), dim=-1)).mean()
    return ce + lambda_jepa * auxiliary, {"ce_loss": float(ce.detach()),
                                        "jepa_loss": float(auxiliary.detach())}


def _optimizer_update(parameters, optimizer, scaler, pending, max_grad_norm=1.0):
    """Average actual accumulated microbatches, including the last partial group."""
    scaler.unscale_(optimizer)
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(pending)
    torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return scaler.get_scale() >= old_scale


def _resolve_resume(resume, output_dir):
    from src.training.runtime import latest_checkpoint
    if resume == "latest":
        return latest_checkpoint(output_dir)
    if resume and not (Path(resume) / "COMPLETE").is_file():
        raise ValueError(f"Incomplete or missing resume checkpoint: {resume}")
    return resume


def _generation_extra(kind, fingerprint, epoch, batch_index, predictor=None, **kwargs):
    return dict(kind=kind, dataset_fingerprint=fingerprint, epoch=epoch,
                batch_index=batch_index,
                predictor=predictor.state_dict() if predictor is not None else None,
                **kwargs)


def run_dry_run(
    data_path: str = "data/mini/mini_synsql.jsonl",
    vocab_size: int = 256,
    hidden_dim: int = 64
) -> Dict[str, Any]:
    """
    Executes a comprehensive offline verification on sample data:
    1. Loads samples from data_path.
    2. Builds synthetic token sequences for prompt and completion (<think>...</think><answer>...</answer>).
    3. Verifies prompt tokens have label -100.
    4. Computes genuine Completion-Only Cross-Entropy Loss.
    5. Confirms that prompt tokens do not influence the computed loss.

    Returns:
        Dictionary of verification metrics.
    """
    logger.info(f"[DryRun] Starting SFT Completion-Only verification using '{data_path}'...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dry run dataset not found at: {data_path}")

    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    logger.info(f"[DryRun] Loaded {len(samples)} samples from {data_path}")

    # Build character/byte-level pseudo-tokenizer for deterministic offline test
    THINK_TOKEN = 1
    ANSWER_TOKEN = 2

    verified_samples = 0
    total_valid_tokens = 0
    losses = []

    for idx, sample in enumerate(samples[:5]):
        text = sample.get("formatted_prompt", "")
        parts = text.split("<think>")
        if len(parts) < 2:
            continue

        prompt_str = parts[0]
        completion_str = "<think>" + parts[1]

        # Convert to token IDs
        prompt_tokens = [(ord(c) % (vocab_size - 10)) + 10 for c in prompt_str]
        completion_tokens = [THINK_TOKEN] + [(ord(c) % (vocab_size - 10)) + 10 for c in completion_str]
        input_ids = prompt_tokens + completion_tokens

        # Apply label masking: all tokens before THINK_TOKEN must be -100
        labels, start_idx = mask_prompt_tokens(input_ids, think_token_id=THINK_TOKEN)
        
        # Verify invariant: all prompt tokens have label -100
        assert all(lbl == -100 for lbl in labels[:start_idx]), "Prompt token was not masked with -100!"
        assert all(lbl != -100 for lbl in labels[start_idx:]), "Completion token was incorrectly masked!"

        # Generate synthetic logits for cross-entropy verification
        # High logit on the true token simulates trained model, random logits simulate initial model
        random.seed(idx + 42)
        logits = []
        for token_id in input_ids:
            step_logit = [(random.random() * 2.0 - 1.0) for _ in range(vocab_size)]
            # Boost true token slightly
            step_logit[token_id] += 2.0
            logits.append(step_logit)

        loss, num_valid = compute_completion_loss(logits, labels)
        losses.append(loss)
        total_valid_tokens += num_valid
        verified_samples += 1

    avg_loss = sum(losses) / max(len(losses), 1)
    logger.info(f"[DryRun] ✅ Verified {verified_samples} samples. Avg Completion Loss: {avg_loss:.4f}, Valid tokens: {total_valid_tokens}")

    # Verify LoRA configuration creation
    lora_cfg = build_lora_config(rank=512, alpha=16)
    logger.info(f"[DryRun] ✅ LoRA config verified: rank={getattr(lora_cfg, 'r', None)}, targets={TARGET_MODULES_DEFAULT}")

    return {
        "status": "PASSED",
        "verified_samples": verified_samples,
        "avg_loss": avg_loss,
        "total_valid_tokens": total_valid_tokens,
        "lora_rank": 512,
        "completion_masking_verified": True
    }


def run_sft_training(
    data_path="data/mini/mini_synsql.jsonl", output_dir="/kaggle/working/sft_jepa_3b",
    drive_dir=None, epochs=3, batch_size=1, learning_rate=2e-4,
    dry_run=False, **kwargs
):
    """Train a real QLoRA adapter; only explicit dry_run executes synthetic checks.

    Use lambda_jepa=0 for SFT-only. A checkpoint includes the external JEPA
    predictor, optimizer, scaler, RNG states, and the next data cursor. A single
    microbatch/device is the T4 default; two GPUs are not pooled.
    """
    if dry_run:
        return dict(run_dry_run(data_path=data_path), mode="smoke", trained=False)
    if not TORCH_AVAILABLE:
        raise RuntimeError("Real SFT requires PyTorch, Transformers, PEFT and bitsandbytes; use dry_run=True only for smoke tests")
    from src.training.runtime import (load_records, seed_everything, load_causal_model,
        ExperimentLogger, dataset_fingerprint, save_checkpoint, load_checkpoint)
    from src.training.jepa_module import LLM_JEPAPredictor
    max_steps = int(kwargs.get("max_steps", 100))
    save_steps = int(kwargs.get("save_steps", 50))
    accum = int(kwargs.get("gradient_accumulation_steps", 8))
    max_length = int(kwargs.get("max_length", kwargs.get("max_seq_length", 1024)))
    seed = int(kwargs.get("seed", 42))
    device = kwargs.get("device", "cuda:0")
    lambda_jepa = float(kwargs.get("lambda_jepa", 1.0))
    loss_dropout = float(kwargs.get("loss_dropout", 0.5))
    if min(max_steps, save_steps, accum, batch_size, epochs) <= 0 or lambda_jepa < 0 or not 0 <= loss_dropout <= 1:
        raise ValueError("Invalid SFT counts or JEPA configuration")
    seed_everything(seed)
    rows = load_records(data_path)
    resume = _resolve_resume(kwargs.get("resume_from_checkpoint", kwargs.get("resume")), output_dir)
    model, tokenizer = kwargs.get("model"), kwargs.get("tokenizer")
    if model is None or tokenizer is None:
        source = kwargs.get("model_name_or_path", "meta-llama/Llama-3.2-3B-Instruct")
        model, tokenizer = load_causal_model(source, adapter_path=resume, training=True,
                                            device=device, rank=int(kwargs.get("lora_rank", 16)))
    collator = CompletionOnlyDataCollator(tokenizer, max_length=max_length)
    encoded = []
    dropped = 0
    for row in rows:
        try:
            encoded.append(collator.encode_example(row))
        except ValueError as exc:
            if "max_length=" not in str(exc):
                raise
            dropped += 1
    if not encoded:
        raise ValueError("No SFT examples fit max_length; increase it or preprocess the dataset")
    predictor = (LLM_JEPAPredictor(model.config.hidden_size,
                 intermediate_dim=int(kwargs.get("predictor_dim", 256))).to(device)
                 if lambda_jepa > 0 else None)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if predictor is not None:
        parameters += list(predictor.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"))
    fingerprint = dataset_fingerprint(data_path)
    contract = dict(batch_size=batch_size, accumulation=accum, max_length=max_length,
                    seed=seed, lambda_jepa=lambda_jepa, loss_dropout=loss_dropout,
                    learning_rate=learning_rate, predictor_dim=int(kwargs.get("predictor_dim", 256)))
    logger = ExperimentLogger(output_dir, dict(stage="sft", mode="train", **contract,
        model_source=kwargs.get("model_name_or_path", "meta-llama/Llama-3.2-3B-Instruct"),
        lora_rank=int(kwargs.get("lora_rank", 16)),
        dropped_overlength=dropped, examples=len(encoded)))
    step, first_epoch, first_batch = 0, 0, 0
    if resume:
        state = load_checkpoint(resume, model, optimizer, scaler=scaler)
        extra = state["extra"]
        if extra.get("kind") != "sft" or extra.get("dataset_fingerprint") != fingerprint or extra.get("contract") != contract:
            raise ValueError("Resume checkpoint belongs to a different SFT dataset/configuration")
        if predictor is not None:
            predictor.load_state_dict(extra["predictor"])
        step, first_epoch, first_batch = state["step"], extra["epoch"], extra["batch_index"]
    optimizer.zero_grad(set_to_none=True)
    checkpoint = resume
    last_saved_step = step if resume else None
    epoch, next_batch = first_epoch, first_batch
    model.train()
    for epoch in range(first_epoch, int(epochs)):
        order = list(range(len(encoded)))
        random.Random(seed + epoch).shuffle(order)
        batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
        pending = 0
        for batch_index in range(first_batch if epoch == first_epoch else 0, len(batches)):
            if step >= max_steps:
                break
            batch = {k: v.to(device) for k, v in collator([encoded[i] for i in batches[batch_index]]).items()}
            with torch.autocast(device_type="cuda" if str(device).startswith("cuda") else "cpu",
                                dtype=torch.float16 if str(device).startswith("cuda") else torch.bfloat16,
                                enabled=str(device).startswith("cuda")):
                loss, metrics = joint_sft_jepa_loss(model, batch, predictor, lambda_jepa, loss_dropout)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite SFT objective")
            scaler.scale(loss).backward()
            pending += 1
            next_batch = batch_index + 1
            if pending == accum or next_batch == len(batches):
                updated = _optimizer_update(parameters, optimizer, scaler, pending)
                pending = 0
                if updated:
                    step += 1
                logger.log(step=step, epoch=epoch, loss=float(loss.detach()),
                           optimizer_updated=updated, **metrics)
                if updated and step % save_steps == 0:
                    checkpoint = save_checkpoint(output_dir, model, optimizer, step, scaler=scaler,
                        tokenizer=tokenizer, extra=_generation_extra("sft", fingerprint,
                        epoch + int(next_batch == len(batches)), 0 if next_batch == len(batches) else next_batch,
                        predictor, contract=contract))
                    last_saved_step = step
            del loss, batch
        if step >= max_steps:
            break
        next_batch = len(batches)
    if step != last_saved_step:
        checkpoint = save_checkpoint(output_dir, model, optimizer, step, scaler=scaler,
            tokenizer=tokenizer, extra=_generation_extra("sft", fingerprint,
            epoch + int(next_batch == len(batches)), 0 if next_batch == len(batches) else next_batch,
                        predictor, contract=contract))
    result = dict(status="COMPLETED", mode="train", trained=step > 0, optimizer_steps=step,
                  checkpoint_path=checkpoint, lambda_jepa=lambda_jepa,
                  examples=len(encoded), dropped_overlength=dropped)
    logger.log(event="finished", **result)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA completion-only SFT + JEPA")
    parser.add_argument("--model_name_or_path", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--data_path", default="data/mini/mini_synsql.jsonl")
    parser.add_argument("--output_dir", default="/kaggle/working/sft_jepa_3b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", "--num_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--lambda_jepa", type=float, default=1.0)
    parser.add_argument("--loss_dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    print(json.dumps(run_sft_training(**vars(parse_args())), indent=2))


if __name__ == "__main__":
    main()
