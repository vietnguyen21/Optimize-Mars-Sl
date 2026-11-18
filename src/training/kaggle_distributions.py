"""Resolve image metadata in import-path order and verify RECORD ownership."""
from collections import defaultdict
from importlib import machinery, metadata
import os
from pathlib import Path
import sys


def distribution_groups(paths=None):
    from packaging.utils import canonicalize_name

    paths = sys.path if paths is None else paths
    groups = defaultdict(list)
    seen = set()
    for rank, entry in enumerate(paths):
        if not isinstance(entry, str):
            continue
        root = Path(entry or os.curdir).absolute()
        if root in seen:
            continue
        seen.add(root)
        for dist in metadata.distributions(path=[str(root)]):
            name = dist.metadata['Name']
            if name:
                groups[canonicalize_name(name)].append((rank, dist))
    return groups


def active_distributions(groups=None):
    groups = distribution_groups() if groups is None else groups
    result = {}
    for name, entries in groups.items():
        rank = min(rank for rank, _ in entries)
        candidates = [dist for index, dist in entries if index == rank]
        if len(candidates) != 1:
            raise RuntimeError(f'Ambiguous distribution metadata for {name} at sys.path index {rank}')
        result[name] = candidates[0]
    return result


def record_files(dist):
    # importlib can synthesize files from egg-info/SOURCES.txt or OS inventories.
    # Only a wheel RECORD qualifies for borrowing files from another environment.
    if not dist.read_text('RECORD'):
        return ()
    return tuple(file for file in dist.files or ()
                 if file.parts and not file.is_absolute() and '..' not in file.parts)


def python_roots(files):
    roots = set()
    for file in files:
        top = file.parts[0]
        if len(file.parts) > 1 and top.isidentifier() and file.suffix == '.py':
            roots.add(top)
        elif len(file.parts) == 1:
            for suffix in ('.py', *machinery.EXTENSION_SUFFIXES):
                if top.endswith(suffix) and top[:-len(suffix)].isidentifier():
                    roots.add(top[:-len(suffix)])
                    break
    return roots


def verified_origins(name, dist, paths=None):
    """Check discoverable and already-loaded modules, without importing them."""
    files = record_files(dist)
    if not files:
        raise RuntimeError(f'Active distribution {name} has no verified wheel RECORD')
    owned = {Path(dist.locate_file(file)).absolute() for file in files}
    origins = {}
    for module in sorted(python_roots(files)):
        spec = machinery.PathFinder.find_spec(module, paths)
        loaded = sys.modules.get(module)
        specs = [spec] if loaded is None else [spec, getattr(loaded, '__spec__', None)]
        for candidate in specs:
            origin = getattr(candidate, 'origin', None)
            if origin:
                path = Path(origin).absolute()
                if path not in owned or not path.is_file():
                    raise RuntimeError(f'Distribution/import origin mismatch for {name} ({module})')
                origins[module] = str(path.resolve())
            else:
                root = Path(dist.locate_file(module)).absolute()
                locations = getattr(candidate, 'submodule_search_locations', None)
                if not locations or root not in {Path(path).absolute() for path in locations}:
                    raise RuntimeError(f'Distribution/import origin mismatch for {name} ({module})')
                # A split namespace has a synthetic directory in the bridge.
                # Pin to its owned files so image and bridge report the same origins.
                namespace_files = sorted(str(path.resolve()) for path in owned
                                         if path.is_relative_to(root) and path.is_file())
                if not namespace_files:
                    raise RuntimeError(f'Namespace ownership cannot be verified for {name} ({module})')
                origins[module] = namespace_files
    return origins


def duplicate_report(groups=None):
    groups = distribution_groups() if groups is None else groups
    report = {}
    for name, entries in sorted(groups.items()):
        if len(entries) < 2:
            continue
        selected = None
        origins = None
        error = None
        try:
            selected = active_distributions({name: entries})[name]
            origins = verified_origins(name, selected)
        except RuntimeError as exc:
            error = str(exc)
        report[name] = {
            'distributions': [
                {'version': dist.version, 'sys_path_index': rank,
                 'location': str(dist.locate_file('')), 'wheel_record': bool(record_files(dist)),
                 'selected_by_precedence': dist is selected}
                for rank, dist in entries
            ],
            'verified_import_origins': origins, 'error': error,
        }
    return report
