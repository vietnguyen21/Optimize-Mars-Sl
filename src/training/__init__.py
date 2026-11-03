"""
Training Package for Text-to-SQL SLM Research Architecture (Phase 2).

Modules:
  - callbacks: Google Drive checkpoint synchronization and session recovery.
  - jepa_module: LLM-JEPA 2-block causal mask and cosine distance loss with dropout.
  - train_sft_generator: Completion-Only SFT trainer with -100 label masking and LoRA/QLoRA.
  - train_grpo_generator: Dense GRPO trainer with 4 reward heads and advantage normalization.
  - train_embedding: LitE-SQL Qwen3-Embedding HN-SupCon trainer and offline index export.
  - train_decision_model: Laya-421M RLCD decision trainer with dual LR and probability calibration.
"""

from src.training.callbacks import (
    DriveCheckpointCallback,
    find_latest_checkpoint,
    get_resume_checkpoint,
)

from src.training.jepa_module import (
    build_2block_causal_mask,
    compute_cosine_distance,
    LLM_JEPAModule,
    compute_jepa_loss,
)

from src.training.train_sft_generator import (
    CompletionOnlyDataCollator,
    mask_prompt_tokens,
    compute_completion_loss,
    build_lora_config,
)

from src.training.train_grpo_generator import (
    compute_format_reward,
    compute_exec_reward,
    compute_mem_reward,
    compute_atomic_reward,
    compute_dense_rewards,
    normalize_group_advantages,
    compute_grpo_loss,
    DenseGRPOTrainer,
)

from src.training.train_embedding import (
    compute_hn_supcon_loss,
    HNSupConLoss,
    format_column_card_text,
    export_column_index,
)

from src.training.train_decision_model import (
    configure_dual_lr_optimizer,
    compute_rlcd_loss,
    compute_ece,
    calibrate_probabilities,
    fit_temperature,
)

__all__ = [
    # Callbacks
    "DriveCheckpointCallback",
    "find_latest_checkpoint",
    "get_resume_checkpoint",
    # JEPA
    "build_2block_causal_mask",
    "compute_cosine_distance",
    "LLM_JEPAModule",
    "compute_jepa_loss",
    # SFT Generator
    "CompletionOnlyDataCollator",
    "mask_prompt_tokens",
    "compute_completion_loss",
    "build_lora_config",
    # GRPO Generator
    "compute_format_reward",
    "compute_exec_reward",
    "compute_mem_reward",
    "compute_atomic_reward",
    "compute_dense_rewards",
    "normalize_group_advantages",
    "compute_grpo_loss",
    "DenseGRPOTrainer",
    # Embedding
    "compute_hn_supcon_loss",
    "HNSupConLoss",
    "format_column_card_text",
    "export_column_index",
    # Decision Model
    "configure_dual_lr_optimizer",
    "compute_rlcd_loss",
    "compute_ece",
    "calibrate_probabilities",
    "fit_temperature",
]
