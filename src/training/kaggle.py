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
        return str(Path(os.environ.get(variable, str(self.data_dir / relative))))

    def output(self, phase):
        p = self.output_root / ('smoke' if self.smoke else 'train') / phase
        p.mkdir(parents=True, exist_ok=True)
        return str(p)


def install_dependencies(repo_dir):
    """Opt-in Kaggle bootstrap; installed torch/CUDA is left intact."""
    if os.environ.get('INSTALL_DEPS', '0') == '1':
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', str(Path(repo_dir) / 'requirements-kaggle.txt')])


def environment_report(output_dir):
    packages = {}
    for name in ('torch', 'transformers', 'peft', 'bitsandbytes', 'sentence-transformers', 'sqlglot', 'numpy'):
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
