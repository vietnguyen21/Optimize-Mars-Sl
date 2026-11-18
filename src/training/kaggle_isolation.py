"""Create a project-only venv that borrows the image's protected CUDA stack."""
import json
from pathlib import Path
import sys
import sysconfig
import venv

from src.training.kaggle_distributions import (
    active_distributions, ownership_index, protected_file_plan, verified_origins,
)


BRIDGE_DIR = 'kaggle-image-packages'
PROVENANCE = 'kaggle-image.json'


def protected_import_paths(distributions, protected):
    """Expose only authorized files, preserving their relative wheel layout."""
    ownership = ownership_index(distributions)
    for name in protected:
        verified_origins(name, distributions[name], ownership=ownership, allowed=protected)
    links, _ = protected_file_plan(ownership, protected, source=True)
    return links


def verify_bridge_files(bridge, protected, ownership):
    links, _ = protected_file_plan(ownership, protected)
    actual = {path for path in bridge.rglob('*') if path.is_symlink() or not path.is_dir()}
    if actual != set(links.values()) or any(not path.is_symlink() or not path.is_file() for path in actual):
        raise RuntimeError('Protected bridge has missing or unauthorized files')


def create_modern_venv(destination):
    from src.training.modern_environment import (
        dependency_graph, package_inventory, pip_check, protected_packages,
        require_target, runtime_identity,
    )

    destination = Path(destination).resolve()
    if not destination.is_relative_to(Path('/kaggle/working')) or destination.exists():
        raise RuntimeError('Choose a new directory under /kaggle/working for the modern venv')
    if sys.prefix != sys.base_prefix:
        raise RuntimeError('Create the modern venv with the Kaggle image interpreter')
    selected = active_distributions()
    inventory = package_inventory(selected)
    borrowed, issues = dependency_graph(inventory, list(protected_packages(inventory)))
    if issues:
        raise RuntimeError(f'The Kaggle image CUDA dependency closure is inconsistent: {issues}')
    links = protected_import_paths(selected, borrowed)
    runtime = runtime_identity(inventory, borrowed, selected)
    require_target(runtime)
    return create_isolated_venv(destination, links, {'runtime': runtime, 'global_pip': pip_check()})


def create_isolated_venv(destination, links, provenance):
    from src.training.modern_environment import canonical_json

    destination.mkdir(parents=True)
    venv.EnvBuilder(with_pip=True).create(destination)
    bridge = destination / BRIDGE_DIR
    bridge.mkdir()
    for name, source in links.items():
        link = bridge / name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(source)
    site = Path(sysconfig.get_path('purelib', scheme='venv', vars={'base': str(destination)}))
    (site / 'kaggle-image.pth').write_text(str(bridge) + '\n')
    (destination / PROVENANCE).write_text(canonical_json(provenance))
    return destination / 'bin' / 'python'


def isolated_image():
    """Reject base Python, inherited image site-packages and warm kernels."""
    prefix = Path(sys.prefix)
    provenance = prefix / PROVENANCE
    bridge = prefix / BRIDGE_DIR
    if not (sys.flags.isolated and sys.flags.dont_write_bytecode and sys.prefix != sys.base_prefix
            and provenance.is_file() and str(bridge) in sys.path):
        raise RuntimeError('Run modern environment operations with the Kaggle venv Python using -I -B')
    for entry in sys.path:
        path = Path(entry)
        if {'site-packages', 'dist-packages'} & set(path.parts) and not path.is_relative_to(prefix):
            raise RuntimeError('The modern venv inherits an external package directory')
    return json.loads(provenance.read_text())
