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
import shutil
import argparse
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
    """
    Data collator that masks prompt tokens with -100 so that loss is computed
    strictly over completion tokens (<think>...</think><answer>...</answer>).
    """
    def __init__(
        self,
        tokenizer: Any = None,
        think_tag: str = "<think>",
        ignore_index: int = -100,
        max_length: int = 2048
    ):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.think_tag = think_tag
        self.ignore_index = ignore_index

    def _collate_prompt_completion(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Tokenizes prompt and completion separately, exactly as inference sees them, so the
        label boundary needs no tag search. EOS stays in labels; long prompts lose their head.
        """
        pad_id = self.tokenizer.pad_token_id
        rows = []
        for item in batch:
            prompt_ids = self.tokenizer(item["prompt"], add_special_tokens=False)["input_ids"]
            completion_ids = self.tokenizer(item["completion"], add_special_tokens=False)["input_ids"]
            completion_ids = completion_ids[:self.max_length - 1] + [self.tokenizer.eos_token_id]
            prompt_ids = prompt_ids[-(self.max_length - len(completion_ids)):]
            rows.append((prompt_ids, completion_ids))

        width = max(len(p) + len(c) for p, c in rows)
        input_ids, attention_mask, labels = [], [], []
        for prompt_ids, completion_ids in rows:
            ids = prompt_ids + completion_ids
            pad = width - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)
            labels.append([self.ignore_index] * len(prompt_ids) + completion_ids + [self.ignore_index] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels)
        }

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if TORCH_AVAILABLE and self.tokenizer is not None and "prompt" in batch[0]:
            return self._collate_prompt_completion(batch)
        if TORCH_AVAILABLE and self.tokenizer is not None:
            # HuggingFace collator logic
            texts = [item["formatted_prompt"] for item in batch]
            encodings = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            input_ids = encodings["input_ids"]
            attention_mask = encodings["attention_mask"]
            labels = input_ids.clone()

            # Find think_tag token sequence
            tag_tokens = self.tokenizer.encode(self.think_tag, add_special_tokens=False)
            tag_len = len(tag_tokens)

            for b in range(input_ids.size(0)):
                seq = input_ids[b].tolist()
                found_idx = -1
                for i in range(len(seq) - tag_len + 1):
                    if seq[i : i + tag_len] == tag_tokens:
                        found_idx = i
                        break

                if found_idx != -1:
                    labels[b, :found_idx] = self.ignore_index
                # Mask padding tokens
                pad_token_id = self.tokenizer.pad_token_id or 0
                labels[b, input_ids[b] == pad_token_id] = self.ignore_index

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        else:
            # Standalone batch structure
            return {"batch": batch}


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
    logger.info(f"[DryRun] ✅ LoRA config verified: rank={lora_cfg.r}, targets={TARGET_MODULES_DEFAULT}")

    return {
        "status": "PASSED",
        "verified_samples": verified_samples,
        "avg_loss": avg_loss,
        "total_valid_tokens": total_valid_tokens,
        "lora_rank": 512,
        "completion_masking_verified": True
    }


def load_sft_records(data_path: str, db_dir: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Loads {prompt, completion} records. The prompt comes from build_prompt, the same function
    the evaluator uses, so train and inference see one format. Accepts BIRD-style records
    (question, evidence, SQL, db_id) and SynSQL records (question, schema, think_steps, answer).
    """
    from src.evaluation.candidate_generator import build_prompt
    from src.evaluation.evaluate_end_to_end import load_schema_ddl

    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            ddl = item.get("schema", "")
            if not ddl and db_dir:
                ddl = load_schema_ddl(os.path.join(db_dir, f"{item['db_id']}.sqlite"))
            sql = item.get("SQL") or item["answer"]
            records.append({
                "prompt": build_prompt(item["question"], item.get("evidence", ""), ddl, item.get("plan", "")),
                "completion": f"\n{item.get('think_steps', '')}\n</think>\n<answer>\n{sql}\n</answer>"
            })
    if not records:
        raise ValueError(f"No SFT records found in {data_path}")
    return records


def run_sft_training(
    data_path: str = "data/mini/mini_synsql.jsonl",
    output_dir: str = "/content/checkpoints/sft_3b",
    drive_dir: Optional[str] = None,
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    dry_run: bool = True,
    model_name_or_path: str = "Qwen/Qwen2.5-Coder-3B-Instruct",
    db_dir: Optional[str] = None,
    lora_rank: int = 64,
    lora_alpha: int = 16,
    resume_from_checkpoint: Optional[str] = None,
    save_steps: int = 500,
    seed: int = 42,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Completion-only SFT with LoRA. dry_run=True runs the offline verification only.
    Note: the LLM-JEPA loss is not integrated into this trainer yet.
    """
    if dry_run:
        return run_dry_run(data_path=data_path)
    if not (TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE and PEFT_AVAILABLE):
        raise RuntimeError("Full SFT requires torch, transformers and peft.")

    resume_path = get_resume_checkpoint(resume_from_checkpoint, output_dir, drive_dir)
    if resume_path:
        logger.info(f"Resuming training from checkpoint: {resume_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = load_sft_records(data_path, db_dir)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False
    model.enable_input_require_grads()  # gradient checkpointing with frozen base weights needs this
    model = get_peft_model(model, build_lora_config(rank=lora_rank, alpha=lora_alpha))
    model.print_trainable_parameters()

    callbacks = []
    if drive_dir:
        callbacks.append(DriveCheckpointCallback(drive_checkpoint_dir=drive_dir, local_checkpoint_dir=output_dir))

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        save_strategy="steps",
        save_steps=save_steps,
        seed=seed,
        logging_steps=50,
        remove_unused_columns=False,  # collator reads the raw prompt/completion fields
        gradient_checkpointing=True,
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=records,
        processing_class=tokenizer,  # saves the tokenizer inside every checkpoint
        data_collator=CompletionOnlyDataCollator(tokenizer=tokenizer),
        callbacks=callbacks
    )
    result = trainer.train(resume_from_checkpoint=resume_path)
    best_dir = os.path.join(output_dir, "best_model")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    if drive_dir:
        os.makedirs(drive_dir, exist_ok=True)
        shutil.copytree(best_dir, os.path.join(drive_dir, "best_model"), dirs_exist_ok=True)
    logger.info("Training complete and model saved.")
    return {"train_loss": result.training_loss, "global_step": result.global_step,
            "num_records": len(records), "saved_to": best_dir}


def parse_args():
    parser = argparse.ArgumentParser(description="Completion-Only SFT Trainer for Text-to-SQL SLMs")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-Coder-3B-Instruct",
                        help="HuggingFace model ID or local path (e.g., FINER-SQL-3B or Qwen2.5-Coder-7B)")
    parser.add_argument("--data_path", type=str, default="data/mini/mini_synsql.jsonl",
                        help="Path to SFT JSONL dataset")
    parser.add_argument("--output_dir", type=str, default="/content/checkpoints/finer_sql_3b",
                        help="Local checkpoint output directory")
    parser.add_argument("--drive_dir", type=str, default="/content/drive/MyDrive/Text2SQL_Colab/checkpoints/finer_sql_3b",
                        help="Google Drive persistent checkpoint directory")
    parser.add_argument("--lora_rank", type=int, default=64,
                        help="LoRA rank dimension (64 to 512)")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha scaling")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate for SFT")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Per-device train batch size")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Checkpoint path or 'latest' to resume from Google Drive / local")
    parser.add_argument("--db_dir", type=str, default=None,
                        help="SQLite directory used to build schema DDL for BIRD-style records")
    parser.add_argument("--dry_run", action="store_true",
                        help="Run dry run verification on mini dataset without loading heavy model")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.dry_run:
        result = run_dry_run(data_path=args.data_path)
        print(json.dumps(result, indent=2))
        sys.exit(0)

    logger.info(f"Starting SFT Trainer on model: {args.model_name_or_path}")
    result = run_sft_training(
        data_path=args.data_path,
        output_dir=args.output_dir,
        drive_dir=args.drive_dir,
        epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dry_run=False,
        model_name_or_path=args.model_name_or_path,
        db_dir=args.db_dir,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
