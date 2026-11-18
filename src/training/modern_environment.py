"""Image-bound, reviewable Kaggle application dependency plans."""
import hashlib
from importlib.machinery import ModuleSpec, PathFinder
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
    active_distributions, distribution_groups, file_sha256, ownership_index, protected_file_plan, python_roots,
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
    return name in {'torch', 'torchvision', 'torchaudio', 'triton', 'pytorch-triton', 'packaging', 'numpy'} or (
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


def runtime_identity(inventory, borrowed, image_sites):
    from src.training.kaggle_native import mapped_gpu_libraries, native_digest, native_inventory

    import torch
    from src.training.environment import validate_torch_build

    validate_torch_build(torch, inventory['torch']['version'])
    cuda_available = torch.cuda.is_available()
    cudnn = torch.backends.cudnn.version()
    nccl = list(torch.cuda.nccl.version()) if cuda_available and torch.distributed.is_nccl_available() else None
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
    arches = torch.cuda.get_arch_list()
    mapped = mapped_gpu_libraries()
    sources = native_inventory(image_sites, mapped)
    driver = command_result(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'])
    return {
        'python': platform.python_version(), 'abi': sysconfig.get_config_var('SOABI'),
        'machine': platform.machine(), 'libc': list(platform.libc_ver()), 'image': image,
        'torch_build': str(torch.__version__), 'cuda_runtime': torch.version.cuda,
        'driver_versions': sorted(set(driver['stdout'].split())) if driver['returncode'] == 0 else [],
        'devices': devices, 'torch_arches': arches,
        'protected_packages': protected_packages(inventory, borrowed),
        'cuda_available': cuda_available, 'cudnn_version': cudnn, 'nccl_version': nccl,
        'native_runtime': {'installed_libraries': sources,
                           'loaded_libraries': {str(path): native_digest(path) for path in sorted(mapped)},
                           'wheel_record_integrity': 'unverified_overlapping_image_claims'},
    }


def require_target(runtime):
    builds = {'torch': '2.11.0', 'torchvision': '0.26.0', 'torchaudio': '2.11.0', 'triton': '3.6.0'}
    protected = runtime['protected_packages']
    mapped = [Path(path).name for path in runtime['native_runtime']['loaded_libraries']]
    if (runtime['python'] != '3.13.16' or runtime['machine'] != 'x86_64'
            or not (runtime['abi'] or '').startswith('cpython-313-')
            or runtime['torch_build'] != '2.11.0+cu130' or runtime['cuda_runtime'] != '13.0'
            or not runtime['driver_versions'] or not runtime['image']['kaggle_working_present']
            or not runtime['cuda_available'] or not runtime['cudnn_version'] or not runtime['nccl_version']
            or not any(name.startswith('libcudnn.so') for name in mapped)
            or not any(name.startswith('libnccl.so') for name in mapped)
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


def import_locations(module, paths=None):
    spec = PathFinder.find_spec(module, paths)
    if spec is None:
        return []
    if spec.origin:
        return [str(Path(spec.origin).resolve())]
    return sorted(str(Path(path).resolve()) for path in spec.submodule_search_locations or ())


def capture_image_baseline(image_sites, *, enforce_target=True):
    from src.training.kaggle_isolation import LAYOUT

    selected = active_distributions(distribution_groups(image_sites))
    inventory = package_inventory(selected)
    if 'torch' not in inventory or not import_locations('torch', image_sites):
        raise RuntimeError('Image torch installation is unavailable')
    runtime = runtime_identity(inventory, (), image_sites)
    ownership = {path.resolve(): claims for path, claims in ownership_index(selected).items()}
    observed = set()
    for module, path in loaded_site_origins(image_sites, sys.modules):
        claims = ownership.get(path)
        if not claims:
            raise RuntimeError(f'No installed image metadata owns runtime import {module}: {path}')
        observed.update(name for name, _, _ in claims)
    protected, issues = dependency_graph(inventory, sorted(set(protected_packages(inventory)) | observed))
    if issues:
        raise RuntimeError(f'Image CUDA dependency closure is inconsistent: {issues}')
    modules = {'torch', 'torchvision', 'torchaudio', 'triton', 'nvidia', 'cuda'}
    for name in protected:
        modules.update(python_roots(record_files(selected[name])))
    origins = {name: import_locations(name, image_sites) for name in sorted(modules)}
    if str(Path(sys.modules['torch'].__file__).resolve()) not in origins['torch']:
        raise RuntimeError('Loaded torch is outside the original image distribution')
    runtime['protected_packages'] = protected_packages(inventory, protected)
    runtime.update(layout=LAYOUT, image_sites=image_sites,
                   protected_import_origins=origins,
                   image_sqlglot={'version': inventory.get('sqlglot', {}).get('version'),
                                  'origins': import_locations('sqlglot', image_sites)})
    if enforce_target:
        require_target(runtime)
        if runtime['image_sqlglot']['version'] != '25.20.2':
            raise RuntimeError('Expected image SQLGlot 25.20.2 before creating the application overlay')
    return {'layout': LAYOUT, 'image_sites': image_sites, 'runtime': runtime, 'global_pip': pip_check()}


def verify_application_prefix(prefix, image):
    """Gate the overlay before placing it on sys.path or importing torch."""
    from src.training.kaggle_native import GPU_LIBRARY

    protected = image['runtime']['protected_packages']
    reserved = set(image['runtime']['protected_import_origins']) | sys.stdlib_module_names | {
        'src', 'sitecustomize', 'usercustomize', 'pip',
    }
    packages = active_distributions(distribution_groups([str(prefix)]))
    if any(protected_name(name) or name in protected for name in packages):
        raise RuntimeError('Application prefix contains a protected distribution')
    for path in prefix.iterdir():
        if path.name.split('.')[0] in reserved:
            raise RuntimeError(f'Application prefix shadows a protected import: {path.name}')
    claimed = ownership_index(packages)
    for path in prefix.rglob('*'):
        has_code = path.suffix in {'.py', '.pyc', '.so'} or '.so.' in path.name
        if (path.is_symlink() or path.suffix == '.pth' or GPU_LIBRARY.match(path.name)
                or (has_code and path.is_file() and path not in claimed)):
            raise RuntimeError(f'Unapproved application prefix file: {path}')


def isolated_runtime(inventory):
    from src.training.kaggle_isolation import activate_application, isolated_image
    from src.training.kaggle_native import native_digest

    image = isolated_image()
    captured = image['runtime']
    activate_application()
    current = package_inventory()
    if protected_packages(current, captured['protected_packages']) != captured['protected_packages']:
        raise RuntimeError('Protected CUDA distribution inventory changed')
    for name, origins in captured['protected_import_origins'].items():
        if import_locations(name) != origins:
            raise RuntimeError(f'Protected image import origin changed: {name}')
        loaded = sys.modules.get(name)
        if loaded is not None:
            location = vars(loaded).get('__file__')
            if location and str(Path(location).resolve()) not in origins:
                raise RuntimeError(f'Loaded protected image import origin changed: {name}')
    expected = captured['native_runtime']['installed_libraries']
    if any(not Path(path).is_file() or native_digest(Path(path)) != digest for path, digest in expected.items()):
        raise RuntimeError('Image native library bytes changed before torch import')
    rebuilt = capture_image_baseline(image['image_sites'], enforce_target=False)['runtime']
    if rebuilt != captured:
        raise RuntimeError('Application runtime differs from the recorded image CUDA fingerprint')
    require_target(rebuilt)
    return rebuilt, image['global_pip']


def application_diagnosis(before, after, roots, image_owned):
    diagnosis = classify(before, after, roots)
    active = set(diagnosis['active_packages'])
    outside = [issue for issue in diagnosis['introduced'] if issue[0] in image_owned - active]
    diagnosis['introduced'] = [issue for issue in diagnosis['introduced'] if issue not in outside]
    diagnosis['excluded_image_conflicts'] = outside
    return diagnosis


def resolve_candidate(repo, stage, output):
    """Resolve without installing. Produce a draft to review on the running image."""
    from packaging.utils import canonicalize_name
    from src.training.environment import profile_path, requirements

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    profile = profile_path(repo, stage, profile='MODERN')
    roots = sorted(str(req) for req in requirements(profile).values())
    runtime, image_pip = isolated_runtime(package_inventory())
    from src.training.kaggle_isolation import APPLICATION_DIR
    if any((Path(sys.prefix) / APPLICATION_DIR).iterdir()):
        raise RuntimeError('Resolve into a fresh virtual environment for each stage')
    inventory = package_inventory()
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
        while True:
            diagnosis = application_diagnosis(inventory, projected, roots + list(protected), set(inventory))
            diagnostics['conflicts'] = diagnosis
            (output / 'modern-resolution.json').write_text(canonical_json(diagnostics))
            if diagnosis['fatal_active'] or diagnosis['introduced']:
                raise RuntimeError('Resolver introduced conflicts or failed active closure; see modern-resolution.json')
            active = {name: projected[name] for name in diagnosis['active_packages']}
            pending = active.keys() - protected.keys() - planned.keys()
            if not pending:
                break
            # Image distributions may have patched metadata. The overlay
            # closure must use the exact metadata of the wheel we will install.
            specs = [f"{name}=={active[name]['version']}" for name in sorted(pending)]
            fetched = set()
            for entry in report_plan(pip + ['--ignore-installed', '--no-deps', *specs], report):
                package = entry['metadata']
                name = canonicalize_name(package['name'])
                if name not in pending or package['version'] != active[name]['version']:
                    raise RuntimeError('Exact application wheel resolution changed a requested pin')
                projected[name] = {'version': package['version'], 'requires_dist': package.get('requires_dist', [])}
                planned[name] = wheel_record(entry)
                fetched.add(name)
            if fetched != pending:
                raise RuntimeError('Resolver did not produce wheel hashes for the full application closure')
    planned = {name: planned[name] for name in active.keys() - protected.keys()}
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


def pip_check(*, application=False):
    from src.training.kaggle_isolation import APPLICATION_DIR

    command = [sys.executable, '-I', '-B']
    prefix = str(Path(sys.prefix) / APPLICATION_DIR)
    if application and prefix in sys.path:
        command += ['-c', 'import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); '
                    'runpy.run_module("pip", run_name="__main__")', prefix, 'check']
    else:
        command += ['-m', 'pip', 'check']
    return command_result(command)


def validate_global_check(result, diagnosis, report_file):
    from packaging.utils import canonicalize_name

    if result['returncode'] == 0:
        return
    unrelated_owners = {issue[0] for issue in diagnosis['unrelated_preinstalled']
                        + diagnosis.get('excluded_image_conflicts', [])}
    owners = {canonicalize_name(line.split()[0])
              for line in result['stdout'].splitlines() if line.strip()}
    if result['returncode'] != 1 or not owners or not owners <= unrelated_owners:
        raise RuntimeError(f'Unclassified pip check failure; see {report_file}')
    print(f'Image conflicts outside the application closure; full pip diagnostics in {report_file}')


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
    from src.training.kaggle_isolation import APPLICATION_DIR

    selected = active_distributions()
    borrowed = lock['runtime']['protected_packages']
    prefix = (Path(sys.prefix) / APPLICATION_DIR).resolve()
    expected = lock['packages'].keys() - borrowed.keys()
    if set(distribution_groups([str(prefix)])) != expected:
        raise RuntimeError('Application prefix contains packages outside the reviewed lock')
    ownership = ownership_index(selected)
    observed = {}
    for name in expected:
        dist = selected[name]
        if not Path(dist.locate_file('')).resolve().is_relative_to(prefix):
            raise RuntimeError(f'Application distribution is outside the isolated venv: {name}')
        origins = verified_origins(name, dist, ownership=ownership)
        for value in origins.values():
            paths = value if isinstance(value, list) else [value]
            if any(not Path(path).is_relative_to(prefix) for path in paths):
                raise RuntimeError(f'Application import is outside the application prefix: {name}')
        observed[name] = {'version': dist.version, 'module_roots': {
            module: value for module, value in origins.items() if '.' not in module
        }}
    return observed


def loaded_site_origins(sites, modules):
    roots = tuple(Path(path) for path in sites)
    for name, module in list(modules.items()):
        location = getattr(module, '__dict__', {}).get('__file__')
        if location:
            path = Path(location).resolve()
            if any(path.is_relative_to(site) for site in roots):
                yield name, path


def reviewed_image_files(lock):
    selected = active_distributions()
    approved = set(lock['packages'])
    ownership = ownership_index({name: dist for name, dist in selected.items() if name in approved})
    return {path.resolve() for path in ownership}


class ReviewedImageFinder:
    """Keep unreviewed image packages out of normal optional import discovery."""

    def __init__(self, sites, files):
        self.sites = tuple(Path(site) for site in sites)
        self.files = files
        self.directories = {parent for file in files for parent in file.parents}

    def permits(self, path, approved):
        path = Path(path).resolve()
        return not any(path.is_relative_to(site) for site in self.sites) or path in approved

    def find_distributions(self, context):
        # PathFinder also exposes the installed distribution metadata;
        # dependency/version probes still need the shared image inventory.
        return PathFinder.find_distributions(context)

    def find_spec(self, fullname, path=None, target=None):
        spec = PathFinder.find_spec(fullname, path, target)
        if spec is None:
            return None
        if spec.origin:
            return spec if self.permits(spec.origin, self.files) else None
        locations = list(spec.submodule_search_locations or ())
        allowed = [entry for entry in locations if self.permits(entry, self.directories)]
        if allowed == locations:
            return spec
        if not allowed:
            return None
        narrowed = ModuleSpec(fullname, loader=None, is_package=True)
        narrowed.submodule_search_locations = allowed
        return narrowed


def restrict_image_imports(lock):
    finder = ReviewedImageFinder(lock['runtime']['image_sites'], reviewed_image_files(lock))
    for index, current in enumerate(sys.meta_path):
        if current is PathFinder or isinstance(current, ReviewedImageFinder):
            sys.meta_path[index] = finder
            return
    raise RuntimeError('Cannot guard application imports without the standard filesystem finder')


def verify_loaded_image_imports(lock, modules=None):
    """Reject ambient image imports, including imports that bypass the guard."""
    modules = sys.modules if modules is None else modules
    allowed = reviewed_image_files(lock)
    for name, path in loaded_site_origins(lock['runtime']['image_sites'], modules):
        if path not in allowed:
            raise RuntimeError(f'Unexpected image dependency imported outside the reviewed closure: {name} ({path})')


def install_application_wheels(lock, digest, changes):
    from src.training.kaggle_isolation import APPLICATION_DIR, APPLICATION_RECEIPT, isolated_image
    from packaging.utils import canonicalize_name

    prefix = Path(sys.prefix) / APPLICATION_DIR
    if any(prefix.iterdir()):
        raise RuntimeError('Create a fresh venv to install a different application lock')
    image = isolated_image()
    with tempfile.TemporaryDirectory(dir=sys.prefix) as directory:
        staged = Path(directory) / 'wheels'
        staged.mkdir()
        path = Path(directory) / 'wheels.txt'
        lines = []
        for name, package in sorted(changes.items()):
            wheel = lock['wheels'][name]
            if not re.fullmatch(r'[a-f0-9]{64}', wheel['sha256']):
                raise RuntimeError(f'Invalid wheel hash for {name}')
            lines.append(f"{name}=={package['version']} --hash=sha256:{wheel['sha256']}\n")
        path.write_text(''.join(lines))
        pip = pip_install_command() + ['--require-hashes', '--no-deps', '--ignore-installed',
                                       '--target', str(staged), '-r', str(path)]
        entries = report_plan(pip, Path(directory) / 'verified.json')
        resolved = {canonicalize_name(entry['metadata']['name']): wheel_record(entry) for entry in entries}
        if resolved != {name: lock['wheels'][name] for name in changes}:
            raise RuntimeError('Hashed dry-run disagrees with reviewed lock')
        run_pip_install(pip)
        verify_application_prefix(staged, image)
        actual = package_inventory(active_distributions(distribution_groups([str(staged)])))
        if set(actual) != set(changes) or any(
                actual[name]['version'] != expected['version']
                or sorted(actual[name]['requires_dist']) != sorted(expected['requires_dist'])
                for name, expected in changes.items()):
            raise RuntimeError('Staged application wheel metadata differs from reviewed lock')
        staged.replace(prefix)
    (Path(sys.prefix) / APPLICATION_RECEIPT).write_text(digest + '\n')


def modern_operation(repo, stage, output, *, install=False):
    from src.training.environment import check_disk, requirements, profile_path
    from src.training.kaggle_isolation import APPLICATION_DIR, APPLICATION_RECEIPT

    # Validate the requested stage before probing CUDA or reading an operator lock.
    requirements(profile_path(repo, stage, profile='MODERN'))
    before = package_inventory()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    check_disk(output, 0)
    runtime, image_pip = isolated_runtime(before)
    before = package_inventory()
    image_owned = {name for name in distribution_groups(runtime['image_sites'])} if 'layout' in runtime else set()
    report = {'schema_version': 1, 'profile': 'MODERN', 'stage': stage, 'runtime': runtime,
              'repository': repository_commit(repo),
              'image_global_pip': image_pip, 'global_pip_before': pip_check(application=True)}
    report_file = output / ('modern-bootstrap.json' if install else 'modern-preflight.json')

    def save():
        report_file.write_text(canonical_json(report))

    save()
    lock, digest = checked_lock(repo, stage, runtime)
    report.update(lock_sha256=digest, effective_lock=lock)
    save()
    if not report['repository']['commit']:
        raise RuntimeError('Repository commit unavailable; set KAGGLE_REPO_COMMIT for an attached snapshot')
    borrowed = runtime['protected_packages']
    prefix = Path(sys.prefix) / APPLICATION_DIR
    receipt = Path(sys.prefix) / APPLICATION_RECEIPT
    overlay = 'layout' in runtime
    if overlay and receipt.is_file() and receipt.read_text().strip() != digest:
        raise RuntimeError('Application installation receipt differs from the reviewed lock')
    roots = lock['roots'] + list(borrowed)
    if install:
        projected = dict(before)
        projected.update(lock['packages'])
        validate_lock_packages(lock, projected)
        diagnosis = application_diagnosis(before, projected, roots, image_owned)
        report['projected_conflicts'] = diagnosis
        save()
        if diagnosis['fatal_active'] or diagnosis['introduced']:
            raise RuntimeError('Modern installation would introduce dependency conflicts')
        changes = {name: package for name, package in lock['packages'].items() if name not in borrowed}
        if any(protected_name(name) for name in changes):
            raise RuntimeError('Modern installation would replace a protected distribution')
        if changes and not (overlay and receipt.is_file()):
            selected = active_distributions()
            for name in changes:
                if (name.replace('-', '_') in sys.modules or
                        (name in selected and python_roots(record_files(selected[name])).intersection(sys.modules))):
                    raise RuntimeError(f'{name} is already loaded; restart with a clean kernel')
            install_application_wheels(lock, digest, changes)
    if overlay and (not receipt.is_file() or str(prefix) not in sys.path):
        raise RuntimeError('Install the reviewed application lock in the controlled venv first')
    after = package_inventory()
    report['global_pip_after'] = pip_check(application=True)
    report['conflicts'] = application_diagnosis(before, after, roots, image_owned)
    save()
    if protected_packages(after, borrowed) != borrowed:
        raise RuntimeError('Protected CUDA distribution inventory changed')
    if overlay:
        isolated_runtime(after)
    diagnosis = report['conflicts']
    if diagnosis['fatal_active'] or diagnosis['introduced']:
        raise RuntimeError(f'Modern dependency conflicts; see {report_file}')
    validate_lock_packages(lock, after)
    report['application_import_origins'] = verify_application_imports(lock)
    save()
    validate_global_check(report['global_pip_after'], diagnosis, report_file)
    if overlay:
        verify_loaded_image_imports(lock)
        # Installation and preflight use separate launch processes. Only the
        # ready application process needs its optional imports restricted.
        if not install:
            restrict_image_imports(lock)
    return report
