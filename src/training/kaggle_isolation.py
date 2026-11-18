"""Create a project-only venv that borrows the image's protected CUDA stack."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import sysconfig
import venv

from src.training.kaggle_distributions import active_distributions, record_files, verified_origins


BRIDGE_DIR = 'kaggle-image-packages'
PROVENANCE = 'kaggle-image.json'


def protected_import_paths(distributions, protected):
    """Use wheel RECORD ownership to link complete package directories safely."""

    records = defaultdict(list)
    for name, dist in distributions.items():
        if name in protected:
            verified_origins(name, dist)
        files = record_files(dist)
        root = Path(dist.locate_file('')).resolve()
        for file in files:
            if name in protected:
                actual = Path(dist.locate_file(file)).resolve()
                if not actual.is_relative_to(root):
                    raise RuntimeError(f'Protected RECORD points outside its installation: {name}')
            top = file.parts[0]
            if top == '__pycache__':
                continue
            path = Path(dist.locate_file(top)).absolute()
            records[path].append((file.parts[1:], name))

    def select(path, claims):
        names = {name for _, name in claims}
        if names.isdisjoint(protected):
            return
        file_owners = defaultdict(set)
        for parts, name in claims:
            file_owners[parts].add(name)
        if any(len(owners) > 1 and not owners.isdisjoint(protected)
               for owners in file_owners.values()):
            raise RuntimeError(f'Protected file shares ownership with {sorted(names)}')
        if names <= protected:
            if path.exists():
                yield path
            return
        if any(not parts for parts, _ in claims):
            raise RuntimeError(f'Protected file shares ownership with {sorted(names)}')
        # A namespace package can contain independent subpackages from many wheels.
        children = defaultdict(list)
        for parts, name in claims:
            children[parts[0]].append((parts[1:], name))
        for child, child_claims in children.items():
            yield from select(path / child, child_claims)

    links = {}
    roots = {}
    for path, claims in records.items():
        for source in select(path, claims):
            relative = Path(path.name) / source.relative_to(path)
            if path.name in roots and roots[path.name] != path:
                raise RuntimeError(f'Protected import path collision: {path.name}')
            roots[path.name] = path
            links[relative] = source
    return links


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
    runtime = runtime_identity(inventory, borrowed, selected)
    require_target(runtime)
    links = protected_import_paths(selected, borrowed)
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
