"""Credentials stay in memory, never in model configuration or git credentials."""
import os
from pathlib import Path


def hf_token(*, model=False, required=False):
    variable = 'MODEL_HF_TOKEN_SECRET' if model else 'CHECKPOINT_HF_TOKEN_SECRET'
    secret = os.environ.get(variable, os.environ.get('CHECKPOINT_HF_TOKEN_SECRET', 'HF_TOKEN'))
    token = os.environ.get(secret)
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret(secret)
        except Exception:
            token = None
    if required and not token:
        raise RuntimeError('HF credentials unavailable; grant the configured Kaggle Secret or secure environment variable') from None
    return token


def model_hf_options(source):
    """Public models can be anonymous; explicit model secrets must be available."""
    if Path(str(source)).exists():
        return {'local_files_only': True}
    token = hf_token(model=True, required='MODEL_HF_TOKEN_SECRET' in os.environ)
    return {'token': token} if token else {}
