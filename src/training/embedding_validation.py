"""Isolated, explicitly labeled held-out retrieval validation."""
from contextlib import contextmanager
import random


EVALUATION_KINDS = ('internal_validation', 'external_held_out')


def validation_contract(eval_steps, eval_path, evaluation_kind, *, session_mode=False):
    if eval_steps < 0:
        raise ValueError('eval_steps must be nonnegative (0 disables periodic validation)')
    if evaluation_kind is not None and evaluation_kind not in EVALUATION_KINDS:
        raise ValueError('Unknown evaluation_kind')
    if (eval_steps or evaluation_kind is not None) and not eval_path:
        raise ValueError('Validation requires an explicit held-out eval_path')
    if eval_path and (eval_steps or session_mode) and evaluation_kind is None:
        raise ValueError('Set evaluation_kind to internal_validation or external_held_out')
    if eval_steps or (eval_path and (session_mode or evaluation_kind is not None)):
        from src.training.runtime import dataset_fingerprint
        return dict(version=1, eval_steps=eval_steps, evaluation_kind=evaluation_kind,
                    eval_sha256=dataset_fingerprint(eval_path), top_k=10)
    return None


@contextmanager
def isolated_validation(model):
    """Preserve all training RNGs and individual module modes, including on error."""
    import numpy as np
    import torch
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        for module, training in modes:
            module.training = training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
