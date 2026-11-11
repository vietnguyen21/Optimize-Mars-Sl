"""Shared reproducible, single-device training utilities (no simulated fallback)."""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import os
import random
import time
import tempfile


@dataclass
class RunConfig:
    output_dir: str
    seed: int = 42
    device: str = 'cuda:0'
    max_steps: int = 100
    max_length: int = 512
    learning_rate: float = 2e-5
    save_steps: int = 50
    resume: str | None = None
    gradient_accumulation_steps: int = 8

    def __post_init__(self):
        if min(self.max_steps, self.max_length, self.save_steps, self.gradient_accumulation_steps) <= 0:
            raise ValueError('Training counts and lengths must be positive')


def load_records(path):
    p = Path(path)
    with p.open(encoding='utf-8') as f:
        rows = json.load(f) if p.suffix == '.json' else [json.loads(line) for line in f if line.strip()]
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'Expected a nonempty record list: {path}')
    return rows


def seed_everything(seed=42):
    random.seed(seed)
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_prompt(question, schema=''):
    return ('<system>You are an expert SQL analyst. Return reasoning in <think> and '
            'SQLite SQL in <answer> tags.</system>\n'
            f'<schema>{schema}</schema>\n<question>{question}</question>\n[PRED]')


def resolve_db(db_dir, db_id):
    root = Path(db_dir).resolve()
    for candidate in (root / f'{db_id}.sqlite', root / db_id / f'{db_id}.sqlite'):
        p = candidate.resolve()
        if p.is_relative_to(root) and p.is_file():
            return str(p)
    raise FileNotFoundError(f'Database {db_id!r} missing under {db_dir}')


def dataset_fingerprint(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class ExperimentLogger:
    def __init__(self, output_dir, config=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if config is not None:
            payload = asdict(config) if hasattr(config, '__dataclass_fields__') else config
            self.log(event='config', config=payload)

    def log(self, metrics=None, **kwargs):
        record = dict(metrics or {}, **kwargs)
        record['time_unix'] = time.time()
        with (self.output_dir / 'metrics.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, default=str, allow_nan=False) + '\n')


def latest_checkpoint(output_dir):
    candidates = []
    for p in Path(output_dir).glob('checkpoint-*'):
        if p.name.split('-')[-1].isdigit() and (p / 'COMPLETE').exists():
            candidates.append((int(p.name.split('-')[-1]), p))
    return str(max(candidates)[1]) if candidates else None


def save_checkpoint(output_dir, model, optimizer, step, *, scaler=None, scheduler=None, extra=None, tokenizer=None):
    """Save complete model/optimizer/RNG state; completion marker written last.

    Existing checkpoints are never replaced. Call only at optimizer boundaries.
    """
    import numpy as np
    import torch
    destination = Path(output_dir) / f'checkpoint-{step}'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f'Checkpoint already exists: {destination}')
    # Interrupted writes remain hidden for inspection and cannot block retrying
    # the same optimizer step after recovery from an earlier complete checkpoint.
    path = Path(tempfile.mkdtemp(prefix=f'.checkpoint-{step}-', dir=destination.parent))
    if hasattr(model, 'save_pretrained'):
        model.save_pretrained(str(path))
        model_state = None
    else:
        model_state = model.state_dict()
    if tokenizer is not None:
        tokenizer.save_pretrained(str(path))
    state = dict(step=step, model=model_state, optimizer=optimizer.state_dict(),
                 rng_python=random.getstate(), rng_numpy=np.random.get_state(),
                 rng_torch=torch.get_rng_state(),
                 rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                 scaler=scaler.state_dict() if scaler else None,
                 scheduler=scheduler.state_dict() if scheduler else None, extra=extra or {})
    torch.save(state, path / 'training_state.pt')
    (path / 'COMPLETE').write_text('complete\n')
    from src.training.checkpoint_sync import ACTIVE_SYNC
    sync = ACTIVE_SYNC.get()
    if sync is not None:
        sync.prepare_manifest(path, step, extra or {})
    # COMPLETE remains the last file written; staging is never discoverable.
    (path / 'COMPLETE').write_text('complete\n')
    if destination.exists():
        raise FileExistsError(f'Checkpoint already exists: {destination}')
    path.rename(destination)
    if sync is not None:
        sync.completed(destination, step, extra or {})
    return str(destination)


def load_checkpoint(path, model, optimizer=None, *, scaler=None, scheduler=None, expected_extra=None):
    """Load trusted local training checkpoints only; includes Python/NumPy RNG pickle."""
    import numpy as np
    import torch
    p = Path(path)
    if not (p / 'COMPLETE').exists():
        raise ValueError(f'Incomplete checkpoint: {path}')
    from src.training.checkpoint_sync import verify
    manifest = verify(p)
    state = torch.load(p / 'training_state.pt', map_location='cpu', weights_only=False)
    if manifest and state.get('step') != manifest['step']:
        raise ValueError('Training state step differs from manifest')
    if expected_extra and any(state.get('extra', {}).get(k) != v for k, v in expected_extra.items()):
        raise ValueError('Resume checkpoint dataset/configuration mismatch')
    if state['model'] is not None:
        model.load_state_dict(state['model'])
    elif (p / 'adapter_config.json').exists():
        from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
        set_peft_model_state_dict(model, load_peft_weights(str(p)))
    else:
        # HF/sentence-transformer weights saved via their native interface.
        restored = type(model).from_pretrained(str(p)) if hasattr(type(model), 'from_pretrained') else type(model)(str(p), device='cpu')
        model.load_state_dict(restored.state_dict())
        del restored
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    if scaler is not None and state['scaler'] is not None:
        scaler.load_state_dict(state['scaler'])
    if scheduler is not None and state['scheduler'] is not None:
        scheduler.load_state_dict(state['scheduler'])
    random.setstate(state['rng_python'])
    np.random.set_state(state['rng_numpy'])
    torch.set_rng_state(state['rng_torch'])
    if torch.cuda.is_available() and state['rng_cuda'] is not None:
        torch.cuda.set_rng_state_all(state['rng_cuda'])
    return state


def load_causal_model(model_name_or_path, adapter_path=None, training=False, device='cuda:0', rank=16):
    """Load NF4 QLoRA on one T4; never aggregate the two GPUs as shared VRAM."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftConfig, PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
    source = str(adapter_path or model_name_or_path)
    if (Path(source) / 'adapter_config.json').exists():
        adapter_path = source
        model_name_or_path = os.environ.get('BASE_MODEL_PATH') or PeftConfig.from_pretrained(source).base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right' if training else 'left'
    use_cuda = str(device).startswith('cuda')
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use smoke mode or a GPU runtime')
    options = dict(device_map={'': device}, attn_implementation='sdpa',
                   torch_dtype=torch.float16 if use_cuda else torch.float32)
    if use_cuda:
        options['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **options)
    if use_cuda and training:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={'use_reentrant': False})
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=training)
    elif training:
        model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=rank * 2, lora_dropout=0.0,
            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
            task_type='CAUSAL_LM', bias='none'))
    if training:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.train()
    else:
        model.eval()
    return model, tokenizer
