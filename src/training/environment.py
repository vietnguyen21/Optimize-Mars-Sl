"""Fail-fast environment checks. No pretrained downloads, installation or uploads."""
from pathlib import Path
import importlib
import importlib.metadata as metadata
import json
import os
import shutil
import sys
import tempfile

PROFILES = {'grounding': 'kaggle-grounding.txt', 'sft_jepa_3b': 'kaggle-generation.txt',
            'grpo_3b': 'kaggle-generation.txt', 'system1': 'kaggle-core.txt',
            'evaluation': 'kaggle-evaluation.txt'}
CORE = {'transformers': 'transformers', 'accelerate': 'accelerate',
        'huggingface-hub': 'huggingface_hub', 'tokenizers': 'tokenizers',
        'safetensors': 'safetensors', 'numpy': 'numpy', 'networkx': 'networkx',
        'sqlglot': 'sqlglot', 'pydantic': 'pydantic'}


def profile_path(repo, stage):
    if stage not in PROFILES:
        raise ValueError('Unknown training environment profile')
    return Path(repo) / 'requirements' / PROFILES[stage]


def requirements(path):
    """Read recursive pip -r files without executing them."""
    result = {}
    for line in Path(path).read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if line.startswith('-r '):
            result.update(requirements(Path(path).parent / line[3:].strip()))
        elif line:
            from packaging.requirements import Requirement
            req = Requirement(line)
            result[req.name] = req
    return result


def cuda_stack():
    result = {}
    for name in ('torch', 'torchvision', 'torchaudio', 'triton'):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return result


def validate_torch_build(torch, installed=None):
    from packaging.version import Version
    installed = installed or metadata.version('torch')
    disk, loaded = Version(installed), Version(str(torch.__version__))
    # PyPI metadata can omit the CUDA local suffix present in torch.__version__.
    if disk.base_version != loaded.base_version or (disk.local and disk.local != loaded.local):
        raise RuntimeError('Loaded/disk PyTorch build mismatch; restart the kernel before training')


def check_disk(path, required, *, reserve=None):
    path = Path(path)
    while not path.exists():
        path = path.parent
    reserve = int(os.environ.get('CHECKPOINT_DISK_RESERVE_BYTES', str(256 * 1024**2))) if reserve is None else reserve
    if required < 0 or reserve < 0:
        raise ValueError('Disk budget must be nonnegative')
    if shutil.disk_usage(path).free < required + reserve:
        raise OSError(f'Insufficient disk space at {path}: need {required + reserve} bytes including reserve; no checkpoints deleted')


def check_transfer_disk(output, cache, payload):
    """Allow both downloaded cache and copied staging payload, on either filesystem."""
    output, cache = Path(output), Path(cache)
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    if output.stat().st_dev == cache.stat().st_dev:
        check_disk(output, payload * 2)
    else:
        check_disk(output, payload)
        check_disk(cache, payload)


def validate_environment(repo, stage, output_dir, device='cuda:0', *, smoke=False, quantization=False):
    if smoke:
        return {'smoke': True, 'environment_validation': 'offline wiring only'}
    if not (3, 10) <= sys.version_info[:2] < (3, 13):
        raise RuntimeError('Kaggle profiles require Python 3.10–3.12; validate a separate profile for other versions')
    import torch
    from packaging.version import Version
    validate_torch_build(torch)
    if not Version('2.5') <= Version(torch.__version__.split('+')[0]) < Version('3'):
        raise RuntimeError('Require preinstalled torch >=2.5,<3; select a compatible image before installing application packages')
    for name, req in requirements(profile_path(repo, stage)).items():
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            raise RuntimeError(f'Missing required package: {name}; bootstrap the {stage} profile') from None
        if installed not in req.specifier:
            raise RuntimeError(f'Package mismatch: {name} {installed}; require {req.specifier}')
        module = CORE.get(name, name.replace('-', '_'))
        try:
            loaded = importlib.import_module(module)
        except Exception as exc:
            raise RuntimeError(f'Import failed: {name} ({type(exc).__name__}); check binary compatibility and restart kernel') from None
        loaded_version = getattr(loaded, '__version__', None)
        if loaded_version and Version(str(loaded_version)) != Version(installed):
            raise RuntimeError(f'Loaded/disk version mismatch for {name}; restart kernel before training')
    try:
        transformers = importlib.import_module('transformers')
        for name in ('PreTrainedModel', 'AutoTokenizer', 'AutoModelForCausalLM',
                     'AutoModelForSequenceClassification', 'BitsAndBytesConfig'):
            getattr(transformers, name)
        if stage in ('system1', 'evaluation'):
            getattr(transformers, 'ModernBertForSequenceClassification')
    except Exception as exc:
        raise RuntimeError(f'Transformers model API import failed ({type(exc).__name__}); check optional binary packages and restart kernel') from None
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=output) as handle:
        handle.write(b'writable')
    check_disk(output, 0)
    report = {'stage': stage, 'device': device, 'torch_build': str(torch.__version__), 'cuda_build': torch.version.cuda}
    if str(device).startswith('cuda'):
        if not torch.cuda.is_available() or torch.version.cuda is None:
            raise RuntimeError('CUDA-enabled PyTorch/GPU unavailable; select a GPU runtime')
        index = torch.device(device).index
        index = torch.cuda.current_device() if index is None else index
        props = torch.cuda.get_device_properties(index)
        report.update(gpu=props.name, capability=[props.major, props.minor],
                      gpu_memory_bytes=props.total_memory, torch_arch_list=torch.cuda.get_arch_list())
        # Enumeration alone cannot detect missing binary kernels for sm_60/sm_75.
        x = torch.ones(8, 8, device=device, requires_grad=True)
        (x @ x).sum().backward()
        torch.cuda.synchronize(device)
        del x
        if stage in ('sft_jepa_3b', 'grpo_3b', 'evaluation') and quantization:
            import bitsandbytes as bnb
            layer = bnb.nn.Linear4bit(32, 32, bias=False, compute_dtype=torch.float16,
                                     quant_type='nf4', compress_statistics=True).to(device)
            layer.requires_grad_(False)
            x = torch.ones(2, 32, device=device, dtype=torch.float16, requires_grad=True)
            layer(x).float().square().mean().backward()
            if not torch.isfinite(x.grad).all():
                raise RuntimeError('NF4 kernel gradient verification failed')
            del layer, x
            report['nf4_kernel'] = 'passed'
    print(json.dumps(report))
    return report
