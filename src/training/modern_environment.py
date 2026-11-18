"""Image-bound, reviewable Kaggle application dependency plans."""
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import sysconfig
import tempfile

from src.training.kaggle_distributions import (
    active_distributions, file_sha256, ownership_index, protected_file_plan, python_roots,
    record_files, verified_origins,
)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, indent=2) + '\n'


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def package_inventory(distributions=None):
    selected = active_distributions() if distributions is None else distributions
    return {name: {'version': dist.version, 'requires_dist': list(dist.requires or [])}
            for name, dist in selected.items()}


def protected_name(name):
    # packaging is borrowed because the resolver uses it before any install.
    return name in {'torch', 'torchvision', 'torchaudio', 'triton', 'pytorch-triton', 'packaging'} or (
        name.startswith(('nvidia-', 'cuda-', 'cupy-', 'pytorch-triton-', 'jax-cuda', 'cudf-', 'cuml-'))
        or name.endswith(('-cu11', '-cu12', '-cu13'))
    )


def protected_packages(inventory, borrowed=()):
    return {name: package['version'] for name, package in inventory.items()
            if protected_name(name) or name in borrowed}


def command_result(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return {'returncode': 127, 'stdout': '', 'stderr': 'Executable unavailable'}
    # Keep diagnostic lines while stripping credentials and query tokens from URLs.
    def redact(value):
        return re.sub(r'(https?://)[^\s]+', r'\1[redacted]', value)
    return {'returncode': result.returncode, 'stdout': redact(result.stdout), 'stderr': redact(result.stderr)}


def safe_env(name, pattern):
    value = os.environ.get(name, '')
    return value if re.fullmatch(pattern, value) else None


def shared_library_fingerprint(inventory, borrowed=(), distributions=None):
    protected = protected_packages(inventory, borrowed)
    selected = active_distributions() if distributions is None else distributions
    ownership = ownership_index(selected)
    links, shared = protected_file_plan(ownership, protected)
    origins = {name: verified_origins(name, selected[name], ownership=ownership, allowed=protected)
               for name in protected}
    native = {}
    for relative, path in links.items():
        if re.search(r'\.so(?:\.|$)', relative.name):
            native[str(relative)] = [str(path.resolve()), file_sha256(path).hex()]
    return sha256(canonical_json({'origins': origins, 'native': native, 'shared': shared}).encode())


def runtime_identity(inventory, borrowed=(), distributions=None, *, expected_fingerprint=None):
    if 'torch' not in inventory:
        raise RuntimeError('Modern M0 requires the preinstalled Kaggle CUDA torch distribution')
    selected = active_distributions() if distributions is None else distributions
    library_fingerprint = shared_library_fingerprint(inventory, borrowed, selected)
    if expected_fingerprint is not None and library_fingerprint != expected_fingerprint:
        raise RuntimeError('Isolated runtime differs from the recorded image CUDA fingerprint')
    import torch
    from src.training.environment import validate_torch_build

    validate_torch_build(torch, inventory['torch']['version'])
    verified_origins('torch', selected['torch'], ownership=ownership_index(selected))
    image = {
        'digest': safe_env('KAGGLE_IMAGE_DIGEST', r'sha256:[a-f0-9]{64}'),
        'build_commit': safe_env('GIT_COMMIT', r'[a-f0-9]{40}'),
        'build_date': safe_env('BUILD_DATE', r'[0-9]{8}-[0-9]{6}'),
        'kaggle_working_present': Path('/kaggle/working').is_dir(),
    }
    image['identity_gap'] = None if image['digest'] else 'Kaggle image digest is unavailable'
    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append({'name': props.name, 'capability': [props.major, props.minor],
                        'memory_bytes': props.total_memory})
    driver = command_result(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'])
    return {
        'python': platform.python_version(), 'abi': sysconfig.get_config_var('SOABI'),
        'machine': platform.machine(), 'libc': list(platform.libc_ver()), 'image': image,
        'torch_build': str(torch.__version__), 'cuda_runtime': torch.version.cuda,
        'driver_versions': sorted(set(driver['stdout'].split())) if driver['returncode'] == 0 else [],
        'devices': devices, 'torch_arches': torch.cuda.get_arch_list(),
        'protected_packages': protected_packages(inventory, borrowed),
        'protected_library_fingerprint': library_fingerprint,
    }


def require_target(runtime):
    builds = {'torch': '2.11.0', 'torchvision': '0.26.0', 'torchaudio': '2.11.0', 'triton': '3.6.0'}
    protected = runtime['protected_packages']
    if (runtime['python'] != '3.13.16' or runtime['machine'] != 'x86_64'
            or not (runtime['abi'] or '').startswith('cpython-313-')
            or runtime['torch_build'] != '2.11.0+cu130' or runtime['cuda_runtime'] != '13.0'
            or not runtime['driver_versions'] or not runtime['image']['kaggle_working_present']
            or not runtime['devices']
            or any(gpu['capability'] != [7, 5] or 'T4' not in gpu['name'] for gpu in runtime['devices'])
            or any(protected.get(name) not in {version, version + '+cu130'}
                   for name, version in builds.items())):
        raise RuntimeError('Running environment does not match the M0 Kaggle T4 candidate')


def repository_commit(repo):
    result = command_result(['git', '-C', str(repo), 'rev-parse', 'HEAD'])
    if result['returncode'] == 0 and re.fullmatch(r'[a-f0-9]{40}', result['stdout'].strip()):
        return {'commit': result['stdout'].strip(), 'source': 'git'}
    commit = safe_env('KAGGLE_REPO_COMMIT', r'[a-f0-9]{40}')
    return {'commit': commit, 'source': 'operator supplied' if commit else 'unavailable'}


def dependency_graph(inventory, roots):
    """Evaluate Requires-Dist recursively, including requested extras and markers."""
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    active, issues, seen = set(), set(), set()
    pending = [(None, Requirement(root)) for root in roots]
    while pending:
        owner, req = pending.pop()
        if owner is None and req.marker is not None and not req.marker.evaluate({'extra': ''}):
            continue
        name = canonicalize_name(req.name)
        active.add(name)
        package = inventory.get(name)
        version = package['version'] if package else None
        if req.url or version is None or version not in req.specifier:
            issues.add((owner or '<root>', str(req), version or '<missing>'))
        if package is None:
            continue
        for extra in {'', *map(canonicalize_name, req.extras)}:
            if (name, extra) in seen:
                continue
            seen.add((name, extra))
            for value in package['requires_dist']:
                child = Requirement(value)
                if child.marker is None or child.marker.evaluate({'extra': extra}):
                    pending.append((name, child))
    return active, sorted(issues)


def classify(before, after, roots):
    active, fatal = dependency_graph(after, roots)
    _, previous = dependency_graph(before, sorted(before))
    _, current = dependency_graph(after, sorted(after))
    # Compare broken edges, since a changed version can leave an old edge broken.
    baseline = {issue[:2] for issue in previous}
    return {
        'active_packages': sorted(active),
        'fatal_active': fatal,
        'unrelated_preinstalled': [issue for issue in current
                                   if issue[:2] in baseline and issue[0] not in active],
        'introduced': [issue for issue in current if issue[:2] not in baseline],
    }


def run_pip_install(command):
    # Discard pip settings that could redirect a reviewed install out of the venv.
    environment = {name: value for name, value in os.environ.items() if not name.startswith('PIP_')}
    environment['PIP_CONFIG_FILE'] = os.devnull
    subprocess.run(command, env=environment, check=True)


def report_plan(command, path):
    run_pip_install(command + ['--dry-run', '--report', str(path)])
    return json.loads(path.read_text())['install']


def wheel_record(entry):
    from packaging.utils import canonicalize_name, parse_wheel_filename
    from urllib.parse import unquote, urlsplit

    info = entry['download_info']
    filename = Path(unquote(urlsplit(info['url']).path)).name
    wheel_name, wheel_version, _, _ = parse_wheel_filename(filename)
    digest = info.get('archive_info', {}).get('hashes', {}).get('sha256', '')
    package = entry['metadata']
    if (wheel_name != canonicalize_name(package['name']) or str(wheel_version) != package['version']
            or not re.fullmatch(r'[a-f0-9]{64}', digest)):
        raise RuntimeError('Resolver entry lacks a matching hashed binary wheel')
    return {'filename': filename, 'sha256': digest}


def isolated_runtime(inventory):
    from src.training.kaggle_isolation import BRIDGE_DIR, isolated_image, verify_bridge_files

    image = isolated_image()
    captured = image['runtime']
    selected = active_distributions()
    verify_bridge_files(Path(sys.prefix) / BRIDGE_DIR, captured['protected_packages'], ownership_index(selected))
    runtime = runtime_identity(inventory, captured['protected_packages'], selected,
                               expected_fingerprint=captured['protected_library_fingerprint'])
    if runtime != captured:
        raise RuntimeError('Isolated runtime differs from the recorded image CUDA fingerprint')
    require_target(runtime)
    return runtime, image['global_pip']


def resolve_candidate(repo, stage, output):
    """Resolve without installing. Produce a draft to review on the running image."""
    from packaging.utils import canonicalize_name
    from src.training.environment import profile_path, requirements

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    profile = profile_path(repo, stage, profile='MODERN')
    roots = sorted(str(req) for req in requirements(profile).values())
    inventory = package_inventory()
    runtime, image_pip = isolated_runtime(inventory)
    diagnostics = {'schema_version': 1, 'profile': 'MODERN', 'stage': stage,
                   'runtime': runtime, 'repository': repository_commit(repo),
                   'image_global_pip': image_pip, 'global_pip_before': pip_check()}
    (output / 'modern-resolution.json').write_text(canonical_json(diagnostics))
    pip = pip_install_command()
    protected = runtime['protected_packages']
    with tempfile.TemporaryDirectory() as tmp:
        constraints = Path(tmp) / 'protected.txt'
        constraints.write_text(''.join(f'{name}=={version}\n'
                                       for name, version in runtime['protected_packages'].items()))
        report = Path(tmp) / 'resolver.json'
        entries = report_plan(pip + ['-c', str(constraints), '-r', str(profile)], report)
        projected = dict(inventory)
        planned = {}
        for entry in entries:
            package = entry['metadata']
            name = canonicalize_name(package['name'])
            if protected_name(name) or name in protected:
                raise RuntimeError(f'Resolver plans to replace protected package {name}')
            projected[name] = {'version': package['version'], 'requires_dist': package.get('requires_dist', [])}
            planned[name] = wheel_record(entry)
        diagnosis = classify(inventory, projected, roots + list(runtime['protected_packages']))
        diagnostics['conflicts'] = diagnosis
        (output / 'modern-resolution.json').write_text(canonical_json(diagnostics))
        if diagnosis['fatal_active'] or diagnosis['introduced']:
            raise RuntimeError('Resolver introduced conflicts or failed active closure; see modern-resolution.json')
        active = {name: projected[name] for name in diagnosis['active_packages']}
        pinned = [f"{name}=={package['version']}" for name, package in active.items()
                  if name not in protected and name not in planned]
        if pinned:
            for entry in report_plan(pip + ['--ignore-installed', '--no-deps', *pinned], report):
                package = entry['metadata']
                name = canonicalize_name(package['name'])
                if sorted(package.get('requires_dist', [])) != sorted(active[name]['requires_dist']):
                    raise RuntimeError(f'Published wheel metadata differs from installed metadata for {name}')
                planned[name] = wheel_record(entry)
    expected_wheels = {name for name in active if name not in protected}
    if planned.keys() != expected_wheels:
        raise RuntimeError('Resolver did not produce wheel hashes for the full application closure')
    lock = {'schema_version': 1, 'profile': 'MODERN', 'stage': stage, 'roots': roots,
            'runtime': runtime, 'packages': active, 'wheels': planned}
    path = output / 'modern.lock.json'
    with path.open('x') as handle:
        handle.write(canonical_json(lock))
    print({'draft_lock': str(path), 'sha256': sha256(path.read_bytes())})
    return path


def pip_install_command():
    return [sys.executable, '-I', '-B', '-m', 'pip', '--isolated',
            'install', '--no-compile', '--only-binary=:all:']


def pip_check():
    return command_result([sys.executable, '-I', '-B', '-m', 'pip', 'check'])


def validate_global_check(result, diagnosis, report_file):
    from packaging.utils import canonicalize_name

    if result['returncode'] == 0:
        return
    unrelated_owners = {issue[0] for issue in diagnosis['unrelated_preinstalled']}
    owners = {canonicalize_name(line.split()[0])
              for line in result['stdout'].splitlines() if line.strip()}
    if result['returncode'] != 1 or not owners or not owners <= unrelated_owners:
        raise RuntimeError(f'Unclassified pip check failure; see {report_file}')
    print(f'Unrelated preinstalled conflicts reported by pip check; full diagnostics in {report_file}')


def checked_lock(repo, stage, runtime):
    from src.training.environment import profile_path, requirements

    require_target(runtime)
    path = os.environ.get('KAGGLE_MODERN_LOCK')
    approved = os.environ.get('KAGGLE_MODERN_LOCK_SHA256')
    if not path or not approved:
        raise RuntimeError('Set KAGGLE_MODERN_LOCK and its reviewed KAGGLE_MODERN_LOCK_SHA256')
    payload = Path(path).read_bytes()
    digest = sha256(payload)
    if digest != approved:
        raise RuntimeError('Modern lock SHA-256 mismatch')
    lock = json.loads(payload)
    roots = sorted(str(req) for req in requirements(profile_path(repo, stage, profile='MODERN')).values())
    if (lock['schema_version'], lock['profile'], lock['stage'], lock['roots'], lock['runtime']) != (
            1, 'MODERN', stage, roots, runtime):
        raise RuntimeError('Modern lock does not match the requested stage, roots or running image fingerprint')
    return lock, digest


def validate_lock_packages(lock, inventory):
    roots = lock['roots'] + list(lock['runtime']['protected_packages'])
    active, issues = dependency_graph(inventory, roots)
    if issues or active != set(lock['packages']):
        raise RuntimeError('Modern active dependency closure differs from reviewed lock')
    for name in active:
        expected = lock['packages'][name]
        installed = inventory[name]
        if (installed['version'] != expected['version']
                or sorted(installed['requires_dist']) != sorted(expected['requires_dist'])):
            raise RuntimeError(f'Modern dependency metadata drift: {name}')


def verify_application_imports(lock):

    selected = active_distributions()
    borrowed = lock['runtime']['protected_packages']
    prefix = Path(sys.prefix).resolve()
    ownership = ownership_index(selected)
    for name in lock['packages'].keys() - borrowed.keys():
        dist = selected[name]
        if not Path(dist.locate_file('')).resolve().is_relative_to(prefix):
            raise RuntimeError(f'Application distribution is outside the isolated venv: {name}')
        origins = verified_origins(name, dist, ownership=ownership)
        for value in origins.values():
            paths = value if isinstance(value, list) else [value]
            if any(not Path(path).is_relative_to(prefix) for path in paths):
                raise RuntimeError(f'Application import is outside the isolated venv: {name}')


def modern_operation(repo, stage, output, *, install=False):
    from packaging.utils import canonicalize_name
    from src.training.environment import check_disk, requirements, profile_path

    # Validate the requested stage before probing CUDA or reading an operator lock.
    requirements(profile_path(repo, stage, profile='MODERN'))
    before = package_inventory()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    check_disk(output, 0)
    runtime, image_pip = isolated_runtime(before)
    report = {'schema_version': 1, 'profile': 'MODERN', 'stage': stage, 'runtime': runtime,
              'repository': repository_commit(repo),
              'image_global_pip': image_pip, 'global_pip_before': pip_check()}
    report_file = output / ('modern-bootstrap.json' if install else 'modern-preflight.json')

    def save():
        report_file.write_text(canonical_json(report))

    save()
    lock, digest = checked_lock(repo, stage, runtime)
    report.update(lock_sha256=digest, effective_lock=lock)
    save()
    if not report['repository']['commit']:
        raise RuntimeError('Repository commit unavailable; set KAGGLE_REPO_COMMIT for an attached snapshot')
    if install:
        projected = dict(before)
        projected.update(lock['packages'])
        validate_lock_packages(lock, projected)
        diagnosis = classify(before, projected, lock['roots'] + list(runtime['protected_packages']))
        report['projected_conflicts'] = diagnosis
        save()
        if diagnosis['fatal_active'] or diagnosis['introduced']:
            raise RuntimeError('Modern installation would introduce dependency conflicts')
        changes = {name: package for name, package in lock['packages'].items() if name not in before
                   or before[name]['version'] != package['version']
                   or sorted(before[name]['requires_dist']) != sorted(package['requires_dist'])}
        if any(protected_name(name) or name in runtime['protected_packages'] for name in changes):
            raise RuntimeError('Modern installation would replace a protected distribution')
        for name in changes:
            if name.replace('-', '_') in sys.modules:
                raise RuntimeError(f'{name} is already loaded; restart with a clean kernel')
        if changes:
            # Catch import names that differ from distribution names, e.g. sklearn.
            selected = active_distributions()
            for name in changes.keys() & selected.keys():
                if python_roots(record_files(selected[name])).intersection(sys.modules):
                    raise RuntimeError(f'{name} is already loaded; restart with a clean kernel')
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'wheels.txt'
                lines = []
                for name, package in sorted(changes.items()):
                    wheel = lock['wheels'][name]
                    if not re.fullmatch(r'[a-f0-9]{64}', wheel['sha256']):
                        raise RuntimeError(f'Invalid wheel hash for {name}')
                    lines.append(f"{name}=={package['version']} --hash=sha256:{wheel['sha256']}\n")
                path.write_text(''.join(lines))
                pip = pip_install_command() + [
                    '--require-hashes', '--no-deps', '--force-reinstall', '-r', str(path),
                ]
                entries = report_plan(pip, Path(tmp) / 'verified.json')
                resolved = {canonicalize_name(entry['metadata']['name']): wheel_record(entry) for entry in entries}
                if resolved != {name: lock['wheels'][name] for name in changes}:
                    raise RuntimeError('Hashed dry-run disagrees with reviewed lock')
                run_pip_install(pip)
    after = package_inventory()
    report['global_pip_after'] = pip_check()
    report['conflicts'] = classify(before, after, lock['roots'] + list(runtime['protected_packages']))
    save()
    borrowed = runtime['protected_packages']
    if (protected_packages(after, borrowed) != borrowed
            or shared_library_fingerprint(after, borrowed) != runtime['protected_library_fingerprint']):
        raise RuntimeError('Protected CUDA distribution inventory changed')
    diagnosis = report['conflicts']
    if diagnosis['fatal_active'] or diagnosis['introduced']:
        raise RuntimeError(f'Modern dependency conflicts; see {report_file}')
    validate_lock_packages(lock, after)
    verify_application_imports(lock)
    validate_global_check(report['global_pip_after'], diagnosis, report_file)
    return report
