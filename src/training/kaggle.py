"""Environment-configurable notebook paths. Never downloads or mounts implicitly."""
from dataclasses import dataclass
from pathlib import Path
import os
import sys
import subprocess
import json
import importlib.metadata


@dataclass
class KagglePaths:
    repo_dir: Path
    data_dir: Path
    output_root: Path
    smoke: bool
    device: str

    @classmethod
    def from_env(cls):
        repo = Path(__file__).resolve().parents[2]
        smoke = os.environ.get('PIPELINE_MODE', 'train').lower() == 'smoke'
        mode = os.environ.get('PIPELINE_MODE', 'train').lower()
        if mode not in ('train', 'smoke'):
            raise ValueError('PIPELINE_MODE must be train or smoke')
        data = Path(os.environ.get('KAGGLE_DATA_DIR', str(repo / 'data' / 'mini') if smoke else '/kaggle/input/text2sql-data'))
        output = Path(os.environ.get('KAGGLE_OUTPUT_DIR', '/kaggle/working/optimize-mars' if Path('/kaggle').exists() else str(repo / 'notebooks' / 'runs')))
        return cls(repo, data, output, smoke, os.environ.get('TRAIN_DEVICE', 'cuda:0'))

    def input(self, variable, relative):
        path = Path(os.environ.get(variable, str(self.data_dir / relative)))
        if not self.smoke and not path.exists():
            raise FileNotFoundError(f'Required input {variable} does not exist: {path}')
        return str(path)

    def output(self, phase):
        p = self.output_root / ('smoke' if self.smoke else 'train') / phase
        p.mkdir(parents=True, exist_ok=True)
        return str(p)


def install_dependencies(repo_dir, stage='evaluation'):
    """Constrain the image CUDA stack; never reinstall it implicitly."""
    if os.environ.get('PIPELINE_MODE', 'train').lower() == 'smoke':
        return
    if os.environ.get('INSTALL_DEPS', '0') != '1':
        return
    from src.training.environment import profile_path, requirements, cuda_stack, validate_torch_build
    import tempfile
    from packaging.version import Version
    if not (3, 10) <= sys.version_info[:2] < (3, 13):
        raise RuntimeError('Bootstrap requires Python 3.10–3.12; validate another profile separately')
    before = cuda_stack()
    if 'torch' not in before or not Version('2.5') <= Version(before['torch'].split('+')[0]) < Version('3'):
        raise RuntimeError('Select an image with CUDA torch >=2.5,<3 before application installation')
    import torch
    validate_torch_build(torch, before['torch'])
    if torch.version.cuda is None or not torch.cuda.is_available():
        raise RuntimeError('Bootstrap requires the image CUDA-enabled torch and a GPU runtime; use smoke mode for offline checks')
    snapshot_dir = Path(os.environ.get('KAGGLE_OUTPUT_DIR', '/kaggle/working'))
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=snapshot_dir) as handle:
        handle.write(b'writable')
    profile = profile_path(repo_dir, stage)
    # A warm notebook must not replace already imported packages underneath itself.
    for name, req in requirements(profile).items():
        module = name.replace('-', '_')
        if module in sys.modules:
            try:
                installed = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                raise RuntimeError(f'Loaded package {name} has no distribution metadata; restart kernel') from None
            loaded_version = getattr(sys.modules[module], '__version__', installed)
            if installed not in req.specifier or Version(str(loaded_version)) != Version(installed):
                raise RuntimeError(f'{name} is already imported with an incompatible version; restart with a clean kernel before bootstrap')
    with tempfile.TemporaryDirectory(prefix='kaggle-bootstrap-') as directory:
        constraints = Path(directory) / 'cuda-stack.txt'
        report_path = Path(directory) / 'resolver.json'
        constraints.write_text(''.join(f'{name}=={version}\n' for name, version in before.items()))
        command = [sys.executable, '-m', 'pip', 'install', '--only-binary=:all:',
                   '-c', str(constraints), '-r', str(profile)]
        subprocess.check_call(command + ['--dry-run', '--report', str(report_path)])
        plan = json.loads(report_path.read_text())
        planned = {item['metadata']['name'].lower().replace('_', '-') for item in plan.get('install', [])}
        if planned.intersection({'torch', 'torchvision', 'torchaudio', 'triton'}):
            raise RuntimeError('Resolver would replace the protected CUDA stack; use a compatible image/profile')
        warm = sorted(name for name in planned if name.replace('-', '_') in sys.modules)
        if warm:
            raise RuntimeError(f'Resolver would replace loaded packages {warm}; restart a clean kernel or select a compatible image')
        subprocess.check_call(command)
    if cuda_stack() != before:
        raise RuntimeError('CUDA stack changed unexpectedly; stop and restart with the original image')
    subprocess.check_call([sys.executable, '-m', 'pip', 'check'])
    snapshot = {'protected_cuda_stack': before, 'profile': profile.name,
                'resolved_install': [{'name': entry['metadata']['name'], 'version': entry['metadata']['version']} for entry in plan.get('install', [])],
                'installed_packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']}}
    (snapshot_dir / 'dependency_snapshot.json').write_text(json.dumps(snapshot, indent=2))


def initialize_environment(repo_dir, stage, output_dir, *, smoke=False, device='cuda:0'):
    from src.training.environment import validate_environment
    return validate_environment(repo_dir, stage, output_dir, device, smoke=smoke, quantization=True)


def environment_report(output_dir):
    packages = {}
    for name in ('torch', 'transformers', 'accelerate', 'peft', 'bitsandbytes', 'sentence-transformers', 'huggingface-hub', 'tokenizers', 'safetensors', 'pydantic', 'networkx', 'sqlglot', 'numpy'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {'python': sys.version, 'packages': packages, 'device': os.environ.get('TRAIN_DEVICE', 'cuda:0'),
              'installed_packages': dict(sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions() if d.metadata['Name']))}
    try:
        import torch
        report['cuda_version'] = getattr(getattr(torch, 'version', None), 'cuda', None)
        report['gpus'] = [dict(name=torch.cuda.get_device_name(i), total_memory_bytes=torch.cuda.get_device_properties(i).total_memory) for i in range(torch.cuda.device_count())]
    except ImportError:
        report['gpus'] = []
    path = Path(output_dir) / 'environment.json'
    path.write_text(json.dumps(report, indent=2))
    return report


def initialize_checkpoint_backend(output_dir, stage, smoke=False):
    """Notebook preflight: mandatory privacy/ownership check before model loading."""
    if smoke:
        print({'checkpoint_sync': 'disabled in smoke mode'})
        return None
    from src.training.checkpoint_sync import CheckpointSync, verify
    backend = CheckpointSync(output_dir, stage)
    local_step = None
    for path in Path(output_dir).glob('checkpoint-*'):
        if path.name.removeprefix('checkpoint-').isdigit():
            try:
                verify(path)
                local_step = max(local_step or 0, int(path.name.split('-')[-1]))
            except ValueError:
                pass
    remote_step = None
    if backend.enabled:
        try:
            for candidate in backend.remote_candidates(compatible_only=False):
                try:
                    backend._retry(lambda: backend._verify_remote(candidate[1], candidate[2], candidate[3]))
                    remote_step = candidate[0]
                    break
                except ValueError:
                    continue
        except Exception as exc:
            backend._failure(exc)
    print({'latest_completed_local_step': local_step, 'latest_completed_remote_step': remote_step,
           'checkpoint_sync_enabled': backend.enabled})
    return backend
