"""Resolve image metadata in import-path order and verify RECORD ownership."""
from collections import defaultdict
import base64
import csv
import hashlib
from io import StringIO
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
    record = dist.read_text('RECORD')
    if not record:
        return ()
    files = []
    for name, digest, size in csv.reader(StringIO(record)):
        file = metadata.PackagePath(name)
        if file.parts and not file.is_absolute() and '..' not in file.parts:
            file.hash = metadata.FileHash(digest) if digest else None
            file.size = int(size) if size else None
            files.append(file)
    return tuple(files)


def python_modules(files):
    modules = set()
    for file in files:
        parents = file.parts[:-1]
        if not all(part.isidentifier() and part != '__pycache__' for part in parents):
            continue
        # An untagged .so can be a native library rather than a Python module.
        # Such files are covered by the native-library fingerprint.
        for suffix in ('.py', *(ext for ext in machinery.EXTENSION_SUFFIXES if ext != '.so')):
            if file.name.endswith(suffix):
                stem = file.name[:-len(suffix)]
                if stem.isidentifier():
                    parts = parents if stem == '__init__' else (*parents, stem)
                    if parts:
                        modules.add('.'.join(parts))
                break
    return modules


def python_roots(files):
    return {module.split('.')[0] for module in python_modules(files)}


def ownership_index(distributions):
    claims = defaultdict(list)
    for name, dist in distributions.items():
        for file in record_files(dist):
            # Bytecode caches and Windows launchers are outside the Linux bridge.
            if '__pycache__' not in file.parts and file.suffix not in {'.pyc', '.exe'}:
                path = Path(dist.locate_file(file)).absolute()
                claims[path].append((name, dist.version, file))
    return claims


def file_sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').digest()


def verified_claims(path, claims):
    """A shared file needs matching SHA-256 claims and matching installed bytes."""
    if not path.is_file() or not claims:
        raise RuntimeError(f'Missing RECORD ownership or installed file: {path}')
    owners = [(name, version) for name, version, _ in claims]
    if len(owners) == 1:
        return None
    if len(set(owners)) != len(owners):
        raise RuntimeError(f'Duplicate RECORD rows for {path}')
    digest = file_sha256(path)
    encoded = base64.urlsafe_b64encode(digest).rstrip(b'=').decode('ascii')
    size = path.stat().st_size
    for _, _, file in claims:
        recorded = file.hash
        if (recorded is None or recorded.mode != 'sha256' or recorded.value != encoded
                or (file.size is not None and file.size != size)):
            raise RuntimeError(f'Conflicting shared RECORD ownership for {path}: {sorted(owners)}')
    return {'source': str(path.resolve()), 'sha256': digest.hex(), 'owners': sorted(owners)}


def verified_origins(name, dist, paths=None, ownership=None, allowed=None):
    """Walk import specs segment by segment without importing any parent code."""
    files = record_files(dist)
    if not files:
        raise RuntimeError(f'Active distribution {name} has no verified wheel RECORD')
    owned = {Path(dist.locate_file(file)).absolute() for file in files}
    modules = python_modules(files)
    origins = {}
    discovered = {}
    for module in sorted(modules):
        search = paths
        segments = module.split('.')
        for index, segment in enumerate(segments):
            qualified = '.'.join(segments[:index + 1])
            if qualified not in discovered:
                # Passing a short name and explicit search paths also avoids the
                # namespace finder consulting sys.modules for an unloaded parent.
                discovered[qualified] = machinery.PathFinder.find_spec(segment, search)
            spec = discovered[qualified]
            loaded = sys.modules.get(qualified)
            origin = getattr(spec, 'origin', None)
            locations = getattr(spec, 'submodule_search_locations', None)
            if loaded is not None:
                attributes = vars(loaded)
                loaded_spec = attributes.get('__spec__')
                # Some verified modules replace themselves with a fileless
                # instance of a class declared in that same nested module.
                synthetic = (index > 0 and loaded_spec is None and not attributes.get('__file__')
                             and type(loaded).__module__ == qualified)
                if not synthetic and (getattr(loaded_spec, 'origin', None) != origin
                                      or list(getattr(loaded_spec, 'submodule_search_locations', None) or ())
                                      != list(locations or ())):
                    raise RuntimeError(f'Distribution/import origin mismatch for {name} ({qualified})')
            if index == len(segments) - 1:
                path = Path(origin).absolute() if origin else None
                if path not in owned or not path.is_file():
                    raise RuntimeError(f'Distribution/import origin mismatch for {name} ({qualified})')
                origins[qualified] = str(path.resolve())
            else:
                if not locations:
                    raise RuntimeError(f'Distribution/import origin mismatch for {name} ({qualified})')
                if origin and Path(origin).absolute() not in owned:
                    if ownership is None:
                        ownership = ownership_index(active_distributions())
                    path = Path(origin).absolute()
                    claims = ownership.get(path, ())
                    if allowed is not None and any(owner not in allowed for owner, _, _ in claims):
                        raise RuntimeError(f'Unauthorized parent RECORD ownership for {qualified}: {path}')
                    verified_claims(path, claims)
                    origins[qualified] = str(path.resolve())
                search = locations
    return origins


def protected_file_plan(ownership, protected, *, source=False):
    links, shared, roots = {}, {}, {}
    for path, claims in ownership.items():
        owners = {name for name, _, _ in claims}
        if owners.isdisjoint(protected):
            continue
        if not owners.issubset(protected):
            raise RuntimeError(f'Protected file shares ownership with unprotected distributions: {path}')
        evidence = verified_claims(path, claims)
        relative = Path(claims[0][2])
        if any(Path(file) != relative for _, _, file in claims):
            raise RuntimeError(f'Conflicting relative RECORD paths for {path}')
        root = path.parents[len(relative.parts) - 1]
        if source:
            real_root, real_path = root.resolve(), path.resolve()
            if not real_path.is_relative_to(real_root):
                raise RuntimeError(f'Protected RECORD points outside its installation: {path}')
            target = root / real_path.relative_to(real_root)
            if target != path:
                target_owners = {name for name, _, _ in ownership.get(target, ())}
                if not target_owners or not target_owners.issubset(protected):
                    raise RuntimeError(f'Protected symlink targets an unauthorized file: {path}')
        top = relative.parts[0]
        if top in roots and roots[top] != root:
            raise RuntimeError(f'Protected import path collision: {top}')
        roots[top] = root
        links[relative] = path
        if evidence:
            shared[str(relative)] = evidence
    return links, shared


def inspect_record_claims(relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('Specify a relative path within wheel RECORD')
    report = []
    for path, claims in sorted(ownership_index(active_distributions()).items()):
        matching = [(name, version, file) for name, version, file in claims if Path(file) == relative]
        if not matching:
            continue
        entry = {
            'path': str(path), 'resolved_path': str(path.resolve()),
            'installed_sha256': file_sha256(path).hex() if path.is_file() else None,
            'claims': [{'name': name, 'version': version, 'size': file.size,
                        'hash': f'{file.hash.mode}={file.hash.value}' if file.hash else None}
                       for name, version, file in matching],
        }
        try:
            entry['shared_ownership'] = verified_claims(path, matching)
        except RuntimeError as exc:
            entry['error'] = str(exc)
        report.append(entry)
    return report


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
