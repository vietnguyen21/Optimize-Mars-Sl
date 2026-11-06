"""
LLM-JEPA (Joint-Embedding Predictive Architecture) Alignment Module for Text-to-SQL.

Aligns latent representations between Natural Language / Schema context (Block 1)
and Reasoning / Target SQL (Block 2) via:
1. 2-Block Causal Attention Mask with [PRED] prediction token.
2. Latent Predictor Head mapping [PRED] representation to Target representation.
3. Cosine Distance Objective with Random JEPA Loss Dropout (LD=0.5) and scaling lambda=1.0.
"""

import math
import random
import logging
from typing import Optional, Tuple, Dict, Any, Union, List

logger = logging.getLogger(__name__)

# Check for PyTorch availability
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


class ItemVal(float):
    """Float wrapper providing .item() method for tensor API compatibility."""
    def item(self) -> float:
        return float(self)


class MatrixMaskFallback(list):
    """List subclass providing .shape and tuple indexing [i, j].item() fallback."""
    @property
    def shape(self) -> Tuple[int, int]:
        return (len(self), len(self[0]) if self else 0)

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, tuple):
            row, col = item
            val = super().__getitem__(row)[col]
            return ItemVal(val)
        return super().__getitem__(item)


def build_2block_causal_mask(
    seq_len_text: int,
    seq_len_code: int,
    dtype: Optional[Any] = None,
    device: Optional[Any] = None
) -> Any:
    """
    Constructs a 2-Block Causal Attention Mask.
    
    Sequence is partitioned into:
      Block 1: Text Context (Question + SubSchema + [PRED]), length = seq_len_text (T)
      Block 2: Target Code/Reasoning (<think>...</think><answer>...</answer>), length = seq_len_code (C)
      Total length N = T + C
      
    Mask Properties:
      - Block 1 tokens can only attend to prior/current tokens in Block 1 (causal within Block 1).
      - Block 1 tokens CANNOT attend to any token in Block 2 (values set to -inf / False).
      - Block 2 tokens can attend to all tokens in Block 1 (unmasked cross-attention).
      - Block 2 tokens can attend to prior/current tokens in Block 2 (causal within Block 2).
      - Block 2 tokens CANNOT attend to future tokens in Block 2.

    Returns:
      If PyTorch is available: Tensor of shape (N, N) where masked positions are -inf and unmasked are 0.0.
      If PyTorch is not available: MatrixMaskFallback (N x N) of float values (0.0 or -1e9).
    """
    T = seq_len_text
    C = seq_len_code
    N = T + C

    if TORCH_AVAILABLE:
        mask_dtype = dtype or torch.float32
        # Initialize with -inf
        mask = torch.full((N, N), float("-inf"), dtype=mask_dtype, device=device)

        # 1. Block 1 causal attention to Block 1: positions [0..T-1, 0..T-1]
        for i in range(T):
            mask[i, : i + 1] = 0.0

        # 2. Block 2 attention to Block 1: full attention to all T tokens of Block 1
        mask[T:, :T] = 0.0

        # 3. Block 2 causal attention to Block 2: positions [T..N-1, T..N-1]
        for i in range(C):
            row = T + i
            mask[row, T : row + 1] = 0.0

        return mask
    else:
        # Pure-Python matrix implementation with tensor-compatible API
        raw_mask = [[-1e9 for _ in range(N)] for _ in range(N)]
        # Block 1 causal
        for i in range(T):
            for j in range(i + 1):
                raw_mask[i][j] = 0.0

        # Block 2 attending to all Block 1
        for i in range(T, N):
            for j in range(T):
                raw_mask[i][j] = 0.0

        # Block 2 causal attending to Block 2
        for i in range(T, N):
            for j in range(T, i + 1):
                raw_mask[i][j] = 0.0

        return MatrixMaskFallback(raw_mask)


def compute_cosine_distance(
    u: Any,
    v: Any,
    eps: float = 1e-8
) -> Any:
    """
    Computes Cosine Distance d(u, v) = 1.0 - CosineSimilarity(u, v).
    
    Supports both PyTorch Tensors and pure-Python vectors.
    """
    if TORCH_AVAILABLE and isinstance(u, torch.Tensor) and isinstance(v, torch.Tensor):
        # Normalize vectors along feature dimension
        u_norm = F.normalize(u, p=2, dim=-1, eps=eps)
        v_norm = F.normalize(v, p=2, dim=-1, eps=eps)
        cos_sim = torch.sum(u_norm * v_norm, dim=-1)
        cos_dist = 1.0 - cos_sim
        return cos_dist
    else:
        # Pure-Python calculation
        if isinstance(u[0], (list, tuple)):
            # Batch of vectors
            distances = []
            for u_vec, v_vec in zip(u, v):
                dot = sum(x * y for x, y in zip(u_vec, v_vec))
                norm_u = math.sqrt(sum(x * x for x in u_vec)) + eps
                norm_v = math.sqrt(sum(y * y for y in v_vec)) + eps
                sim = dot / (norm_u * norm_v)
                distances.append(1.0 - sim)
            return distances
        else:
            # Single vector pair
            dot = sum(x * y for x, y in zip(u, v))
            norm_u = math.sqrt(sum(x * x for x in u)) + eps
            norm_v = math.sqrt(sum(y * y for y in v)) + eps
            sim = dot / (norm_u * norm_v)
            return 1.0 - sim


if TORCH_AVAILABLE:
    class LLM_JEPAPredictor(nn.Module):
        """
        Lightweight MLP Predictor Head projecting the latent representation
        of the [PRED] token to the target latent representation space.
        """
        def __init__(self, hidden_dim: int, intermediate_dim: Optional[int] = None):
            super().__init__()
            inter_dim = intermediate_dim or hidden_dim
            self.net = nn.Sequential(
                nn.Linear(hidden_dim, inter_dim),
                nn.LayerNorm(inter_dim),
                nn.GELU(),
                nn.Linear(inter_dim, hidden_dim)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class LLM_JEPAModule(nn.Module):
        """
        Full LLM-JEPA Module for Joint-Embedding alignment during training.
        """
        def __init__(
            self,
            hidden_dim: int = 2048,
            lambda_weight: float = 1.0,
            loss_dropout: float = 0.5,
            pred_token_id: int = 151665,
            predictor_dim: Optional[int] = None,
            loss_dropout_prob: Optional[float] = None,
            **kwargs: Any
        ):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.lambda_weight = lambda_weight
            eff_dropout = loss_dropout_prob if loss_dropout_prob is not None else loss_dropout
            self.loss_dropout = eff_dropout
            self.loss_dropout_prob = eff_dropout
            self.pred_token_id = pred_token_id
            self.predictor = LLM_JEPAPredictor(hidden_dim, intermediate_dim=predictor_dim)

        def forward(
            self,
            pred_latent: torch.Tensor,
            target_latent: torch.Tensor,
            force_compute: bool = False
        ) -> Dict[str, torch.Tensor]:
            """
            Computes JEPA loss with Random Loss Dropout.

            Args:
                pred_latent: Latent representation at [PRED] token position, shape (batch, hidden_dim)
                target_latent: Target representation of SQL block, shape (batch, hidden_dim)
                force_compute: If True, bypasses Random Loss Dropout (e.g. for eval)

            Returns:
                Dictionary with 'jepa_loss', 'unscaled_distance', and 'dropped' flag.
            """
            # Apply Random JEPA Loss Dropout (LD=0.5) during training
            is_dropped = False
            if self.training and not force_compute:
                if random.random() < self.loss_dropout:
                    is_dropped = True

            if not isinstance(pred_latent, torch.Tensor):
                pred_latent = torch.as_tensor(pred_latent, dtype=next(self.predictor.parameters()).dtype,
                                               device=next(self.predictor.parameters()).device)
            if not isinstance(target_latent, torch.Tensor):
                target_latent = torch.as_tensor(target_latent, dtype=pred_latent.dtype, device=pred_latent.device)
            predicted_target = self.predictor(pred_latent)
            cos_dist = compute_cosine_distance(predicted_target, target_latent.detach())
            mean_dist = cos_dist.mean()

            if is_dropped:
                # Return zero loss detached or zero tensor with grad requirement
                jepa_loss = predicted_target.sum() * 0.0
            else:
                jepa_loss = self.lambda_weight * mean_dist

            return {
                "jepa_loss": jepa_loss,
                "unscaled_distance": mean_dist.detach(),
                "dropped": torch.tensor(1.0 if is_dropped else 0.0, device=pred_latent.device)
            }

else:
    # Standalone pure-Python fallback class for testing without PyTorch
    class LLM_JEPAPredictor:
        def __init__(self, hidden_dim: int, intermediate_dim: Optional[int] = None):
            self.hidden_dim = hidden_dim
            self.intermediate_dim = intermediate_dim or hidden_dim
            # Simple pseudo-weights for deterministic transformation
            random.seed(42)
            self.w1 = [[(random.random() - 0.5) * 0.1 for _ in range(hidden_dim)] for _ in range(hidden_dim)]

        def forward(self, x: List[float]) -> List[float]:
            # Simple linear mapping
            out = []
            for row in self.w1:
                val = sum(r * xi for r, xi in zip(row, x))
                out.append(val)
            return out

    class LLM_JEPAModule:
        def __init__(
            self,
            hidden_dim: int = 128,
            lambda_weight: float = 1.0,
            loss_dropout: float = 0.5,
            pred_token_id: int = 151665,
            predictor_dim: Optional[int] = None,
            loss_dropout_prob: Optional[float] = None,
            **kwargs: Any
        ):
            self.hidden_dim = hidden_dim
            self.lambda_weight = lambda_weight
            eff_dropout = loss_dropout_prob if loss_dropout_prob is not None else loss_dropout
            self.loss_dropout = eff_dropout
            self.loss_dropout_prob = eff_dropout
            self.pred_token_id = pred_token_id
            self.training = True
            self.predictor = LLM_JEPAPredictor(hidden_dim, intermediate_dim=predictor_dim)

        def forward(
            self,
            pred_latent: Union[List[float], List[List[float]]],
            target_latent: Union[List[float], List[List[float]]],
            force_compute: bool = False
        ) -> Dict[str, Any]:
            is_dropped = False
            if self.training and not force_compute:
                if random.random() < self.loss_dropout:
                    is_dropped = True

            if isinstance(pred_latent[0], list):
                pred_trans = [self.predictor.forward(p) for p in pred_latent]
                cos_dist = compute_cosine_distance(pred_trans, target_latent)
                mean_dist = sum(cos_dist) / len(cos_dist)
            else:
                pred_trans = self.predictor.forward(pred_latent)
                mean_dist = compute_cosine_distance(pred_trans, target_latent)

            if is_dropped:
                jepa_loss = 0.0
            else:
                jepa_loss = self.lambda_weight * mean_dist

            return {
                "jepa_loss": jepa_loss,
                "unscaled_distance": mean_dist,
                "dropped": 1.0 if is_dropped else 0.0
            }


def compute_jepa_loss(
    predicted_latent: Any,
    target_latent: Any,
    lambda_weight: float = 1.0,
    loss_dropout: float = 0.5,
    is_training: bool = True
) -> Tuple[Any, bool]:
    """
    Convenience function to compute Cosine Distance Loss with Random Dropout.
    
    Returns:
        (loss_value, is_dropped_bool)
    """
    is_dropped = False
    if is_training and random.random() < loss_dropout:
        is_dropped = True

    cos_dist = compute_cosine_distance(predicted_latent, target_latent)
    if TORCH_AVAILABLE and isinstance(cos_dist, torch.Tensor):
        mean_dist = cos_dist.mean()
        if is_dropped:
            return torch.tensor(0.0, device=predicted_latent.device), True
        return lambda_weight * mean_dist, False
    else:
        if isinstance(cos_dist, list):
            mean_dist = sum(cos_dist) / len(cos_dist)
        else:
            mean_dist = cos_dist
        if is_dropped:
            return 0.0, True
        return lambda_weight * mean_dist, False
