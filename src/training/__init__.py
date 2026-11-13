"""Training public API, loaded lazily so notebook bootstrap stays dependency-light.

Importing src.training.kaggle must never import Torch/Transformers before pip setup.
"""
from importlib import import_module

_EXPORTS = {'DriveCheckpointCallback': 'src.training.callbacks',
 'find_latest_checkpoint': 'src.training.callbacks',
 'get_resume_checkpoint': 'src.training.callbacks',
 'build_2block_causal_mask': 'src.training.jepa_module',
 'compute_cosine_distance': 'src.training.jepa_module',
 'LLM_JEPAModule': 'src.training.jepa_module',
 'compute_jepa_loss': 'src.training.jepa_module',
 'CompletionOnlyDataCollator': 'src.training.train_sft_generator',
 'mask_prompt_tokens': 'src.training.train_sft_generator',
 'compute_completion_loss': 'src.training.train_sft_generator',
 'build_lora_config': 'src.training.train_sft_generator',
 'compute_format_reward': 'src.training.train_grpo_generator',
 'compute_exec_reward': 'src.training.train_grpo_generator',
 'compute_mem_reward': 'src.training.train_grpo_generator',
 'compute_atomic_reward': 'src.training.train_grpo_generator',
 'compute_dense_rewards': 'src.training.train_grpo_generator',
 'normalize_group_advantages': 'src.training.train_grpo_generator',
 'compute_grpo_loss': 'src.training.train_grpo_generator',
 'DenseGRPOTrainer': 'src.training.train_grpo_generator',
 'compute_hn_supcon_loss': 'src.training.train_embedding',
 'HNSupConLoss': 'src.training.train_embedding',
 'format_column_card_text': 'src.training.train_embedding',
 'export_column_index': 'src.training.train_embedding',
 'configure_dual_lr_optimizer': 'src.training.train_decision_model',
 'compute_rlcd_loss': 'src.training.train_decision_model',
 'compute_ece': 'src.training.train_decision_model',
 'calibrate_probabilities': 'src.training.train_decision_model',
 'fit_temperature': 'src.training.train_decision_model'}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value
