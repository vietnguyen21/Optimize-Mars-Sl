"""Synchronous, immutable HF Hub checkpoint backup. No credentials are serialized.

Only atomic Hub commits publish manifests. Downloads are pinned to a commit and
verified before an atomic local rename. Training state is trusted-team pickle.
"""
from contextvars import ContextVar
from functools import wraps
from pathlib import Path, PurePosixPath
import hashlib
import inspect
import json
import os
import re
import shutil
import tempfile
import time
import warnings

ACTIVE_SYNC = ContextVar('checkpoint_sync', default=None)
MANIFEST = 'checkpoint_manifest.json'


class SyncError(RuntimeError):
    pass


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def files_metadata(path):
    result = {}
    for p in sorted(Path(path).rglob('*')):
        if p.is_symlink():
            raise ValueError('Checkpoint symlinks are not supported')
        if p.is_file() and p.relative_to(path).as_posix() != MANIFEST:
            result[p.relative_to(path).as_posix()] = dict(size=p.stat().st_size, sha256=digest(p))
    if not {'COMPLETE', 'training_state.pt'} <= result.keys():
        raise ValueError('Incomplete checkpoint: COMPLETE and training_state.pt required')
    return result


def validate_manifest(m):
    if not isinstance(m, dict):
        raise ValueError('Invalid checkpoint manifest object')
    if m.get('version') != 1 or m.get('status') != 'complete' or type(m.get('step')) is not int or m['step'] < 0:
        raise ValueError('Invalid checkpoint manifest')
    if not isinstance(m.get('stage'), str) or not isinstance(m.get('experiment_id'), str):
        raise ValueError('Missing checkpoint namespace metadata')
    files = m.get('files', {})
    if not isinstance(files, dict) or not {'COMPLETE', 'training_state.pt'} <= files.keys():
        raise ValueError('Missing checkpoint state/marker')
    for name, meta in files.items():
        if not isinstance(name, str) or not isinstance(meta, dict):
            raise ValueError('Invalid manifest file entry')
        p = PurePosixPath(name)
        if not p.parts or p.is_absolute() or '..' in p.parts or '\\' in name or p.as_posix() != name or name == MANIFEST:
            raise ValueError('Unsafe manifest path')
        if type(meta.get('size')) is not int or meta['size'] < 0 or not isinstance(meta.get('sha256'), str) or not re.fullmatch('[0-9a-f]{64}', meta.get('sha256', '')):
            raise ValueError('Invalid file metadata')
    return m


def verify(path, manifest=None):
    path = Path(path)
    m = manifest or (json.loads((path / MANIFEST).read_text()) if (path / MANIFEST).exists() else None)
    actual = files_metadata(path)
    if m is not None:
        validate_manifest(m)
        if actual != m['files']:
            raise ValueError('Checkpoint checksum/file inventory mismatch')
        if path.name.startswith('checkpoint-') and path.name != f"checkpoint-{m['step']}":
            raise ValueError('Checkpoint step mismatch')
    return m


class CheckpointSync:
    def __init__(self, output_dir, stage, compatibility=None, *, api=None, download=None, sleep=time.sleep, read_only=False):
        self.output = Path(output_dir)
        self.read_only = read_only
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', stage):
            raise ValueError('Invalid checkpoint stage')
        self.stage, self.compatibility = stage, compatibility
        self.enabled = os.environ.get('CHECKPOINT_SYNC_ENABLED', '0') == '1'
        self.experiment = os.environ.get('CHECKPOINT_EXPERIMENT_ID', '')
        self.repo = os.environ.get('CHECKPOINT_HF_REPO_ID', '')
        self.policy = os.environ.get('CHECKPOINT_SYNC_FAILURE_POLICY', 'warn')
        self.every = int(os.environ.get('CHECKPOINT_SYNC_EVERY_N_SAVES', '1'))
        self.attempts = int(os.environ.get('CHECKPOINT_SYNC_MAX_ATTEMPTS', '3'))
        if self.policy not in ('warn', 'stop') or self.every < 1 or not 1 <= self.attempts <= 6:
            raise ValueError('Invalid checkpoint sync policy/interval/attempts (1..6)')
        self.api, self.download, self.sleep = api, download, sleep
        self.count, self.last = 0, None
        self.verified_uploads = set()
        self.prefix = f'checkpoints/{self.experiment}/{stage}'
        if self.enabled:
            self.output.mkdir(parents=True, exist_ok=True)
            if os.environ.get('CHECKPOINT_SYNC_BACKEND', 'hf_hub') != 'hf_hub':
                raise ValueError('Only hf_hub checkpoint backend is supported')
            if not re.fullmatch(r'[\w.-]+/[\w.-]+', self.repo) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', self.experiment):
                raise ValueError('Set CHECKPOINT_HF_REPO_ID and a unique CHECKPOINT_EXPERIMENT_ID')
            if api is None:
                from huggingface_hub import HfApi, hf_hub_download
                secret = os.environ.get('CHECKPOINT_HF_TOKEN_SECRET', 'HF_TOKEN')
                token = os.environ.get(secret)
                if not token:
                    try:
                        from kaggle_secrets import UserSecretsClient
                        token = UserSecretsClient().get_secret(secret)
                    except Exception:
                        raise SyncError('HF token unavailable: configure Kaggle Secrets or secure environment') from None
                self.api = HfApi(token=token)
                self.download = lambda name, revision: hf_hub_download(self.repo, name, repo_type='model', revision=revision, token=token)
            # Privacy/ownership preflight is mandatory even under warn policy.
            self._retry(self._info if self.read_only else self._claim)

    def _retry(self, action):
        for attempt in range(self.attempts):
            try:
                return action()
            except (ValueError, PermissionError):
                raise
            except Exception as exc:
                if attempt + 1 == self.attempts:
                    # Never interpolate provider exception messages (may include credentials).
                    raise SyncError(f'Checkpoint service failed ({type(exc).__name__}); check authentication/network/storage') from None
                self.sleep(min(2 ** attempt, 16))

    def _info(self):
        info = self.api.repo_info(self.repo, repo_type='model')
        if info.private is not True:
            raise PermissionError('Checkpoint repository privacy cannot be confirmed; upload refused')
        return info

    def _read(self, name, revision):
        return json.loads(Path(self.download(name, revision)).read_text())

    def _claim(self):
        from huggingface_hub import CommitOperationAdd
        info = self._info()
        names = self.api.list_repo_files(self.repo, repo_type='model', revision=info.sha)
        name = self.prefix + '/owner.json'
        owner = self.api.whoami()['name']
        if name in names:
            if self._read(name, info.sha) != {'owner': owner}:
                raise PermissionError('Experiment namespace belongs to another HF account; use a new experiment ID')
        else:
            self.api.create_commit(self.repo, repo_type='model', parent_commit=info.sha,
                operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=json.dumps({'owner': owner}).encode())],
                commit_message='Reserve checkpoint experiment namespace')

    def _event(self, **values):
        self.output.mkdir(parents=True, exist_ok=True)
        record = dict(event='checkpoint_sync', stage=self.stage, **values)
        print(json.dumps(record))
        with (self.output / 'checkpoint_sync.jsonl').open('a') as handle:
            handle.write(json.dumps(record) + '\n')

    def _failure(self, exc):
        self._event(success=False, error=type(exc).__name__)
        if self.policy == 'stop':
            raise SyncError('Remote backup unsuccessful; local checkpoint preserved') from None
        warnings.warn('REMOTE BACKUP UNSUCCESSFUL; local checkpoint preserved. Check HF authentication/network/storage.', RuntimeWarning)

    def prepare_manifest(self, path, step, extra):
        m = dict(version=1, status='complete', step=step, stage=self.stage,
                 experiment_id=self.experiment, compatibility=self.compatibility,
                 dataset_configuration_fingerprint=fingerprint(dict(contract=extra.get('contract'), dataset=extra.get('dataset_fingerprint'))),
                 files=files_metadata(path))
        (Path(path) / MANIFEST).write_text(json.dumps(m, indent=2, sort_keys=True))
        return m

    def completed(self, path, step, extra):
        if not (Path(path) / MANIFEST).exists():
            self.prepare_manifest(path, step, extra)
        verify(path)
        self.last = str(path)
        self.count += 1
        if self.enabled and self.count % self.every == 0:
            self.sync(path)

    def sync(self, path):
        started = time.monotonic()
        try:
            # Cache only successful verified transfers in this invocation.
            m = verify(path)
            key = (str(Path(path).resolve()), fingerprint(m))
            if key in self.verified_uploads:
                self._event(success=True, step=m['step'], bytes=0, skipped=True,
                            duration_seconds=time.monotonic() - started)
                return True
            result = self._retry(lambda: self._upload(path))
            self.verified_uploads.add(key)
            self._event(success=True, step=result['step'], bytes=result['bytes'],
                        skipped=result['skipped'], duration_seconds=time.monotonic() - started)
            return True
        except Exception as exc:
            self._failure(exc)
            return False

    def _upload(self, path):
        from huggingface_hub import CommitOperationAdd
        if self.read_only:
            raise PermissionError("Read-only checkpoint backend cannot upload")
        self._claim()
        m = verify(path)
        if m is None or m['stage'] != self.stage or m['experiment_id'] != self.experiment:
            raise ValueError('Checkpoint namespace mismatch or missing manifest')
        info = self._info()
        names = self.api.list_repo_files(self.repo, repo_type='model', revision=info.sha)
        prefix = f"{self.prefix}/checkpoint-{m['step']}"
        manifest_name = prefix + '/' + MANIFEST
        skipped = manifest_name in names
        if skipped:
            if self._read(manifest_name, info.sha) != m:
                raise ValueError('Completed remote checkpoint is immutable; conflicting step')
            revision = info.sha
        else:
            if any(n.startswith(prefix + '/') for n in names):
                raise ValueError('Unexpected partial remote namespace; refusing overwrite')
            operations = [CommitOperationAdd(path_in_repo=prefix + '/' + n, path_or_fileobj=str(Path(path) / n)) for n in m['files']]
            operations.append(CommitOperationAdd(path_in_repo=manifest_name, path_or_fileobj=json.dumps(m, sort_keys=True).encode()))
            commit = self.api.create_commit(self.repo, repo_type='model', operations=operations,
                parent_commit=info.sha, commit_message=f"Complete {self.stage} checkpoint {m['step']}")
            revision = commit.oid
        # Verify the immutable remote bytes, including on retry after a lost response.
        self._verify_remote(prefix, revision, m)
        return dict(step=m['step'], bytes=0 if skipped else sum(f['size'] for f in m['files'].values()), skipped=skipped)

    def _verify_remote(self, prefix, revision, m):
        validate_manifest(m)
        for name, meta in m['files'].items():
            p = Path(self.download(prefix + '/' + name, revision))
            if p.stat().st_size != meta['size'] or digest(p) != meta['sha256']:
                raise ValueError('Remote checkpoint checksum mismatch')

    def remote_candidates(self, compatible_only=True):
        if not self.enabled:
            return []
        info = self._retry(self._info)
        names = self._retry(lambda: self.api.list_repo_files(self.repo, repo_type='model', revision=info.sha))
        result = []
        invalid = False
        for name in names:
            if not re.fullmatch(re.escape(self.prefix) + r'/checkpoint-\d+/' + MANIFEST, name):
                continue
            try:
                m = validate_manifest(self._retry(lambda: self._read(name, info.sha)))
                if m['stage'] != self.stage or m['experiment_id'] != self.experiment:
                    raise ValueError('Remote manifest namespace mismatch')
                if compatible_only and m.get('compatibility') != self.compatibility:
                    continue
                prefix = name.rsplit('/', 1)[0]
                if prefix.rsplit('/', 1)[1] != f"checkpoint-{m['step']}":
                    raise ValueError('Remote step mismatch')
                result.append((m['step'], prefix, info.sha, m))
            except ValueError:
                invalid = True
                self._event(success=False, error='InvalidRemoteManifest')
        if invalid and not result:
            raise ValueError('Remote manifests exist but none are valid; refusing fresh restart')
        return sorted(result, reverse=True)

    def fetch(self, candidate):
        step, prefix, revision, m = candidate
        validate_manifest(m)
        dest = self.output / f'checkpoint-{step}'
        if dest.exists():
            if verify(dest) != m:
                raise FileExistsError('Local checkpoint conflicts with remote; use a new output directory')
            return str(dest)
        required = sum(f['size'] for f in m['files'].values())
        if shutil.disk_usage(self.output).free < required * 2 + 1024 * 1024:
            raise OSError('Insufficient disk space for checkpoint download and staging')
        # Preserve interrupted staging folders; Hub cache resumes partial transfers.
        staging = Path(tempfile.mkdtemp(prefix='.checkpoint-download-', dir=self.output))
        for name in m['files']:
            source = self._retry(lambda: self.download(prefix + '/' + name, revision))
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        (staging / MANIFEST).write_text(json.dumps(m, sort_keys=True))
        verify(staging, m)
        staging.rename(dest)
        return str(dest)

    def resolve(self, source=None):
        source = source or os.environ.get('CHECKPOINT_RESUME_SOURCE', 'none')
        if source == 'none':
            return None
        if source not in ('latest', 'local', 'remote', 'auto'):
            if not Path(source).exists():
                raise FileNotFoundError('Explicit resume checkpoint does not exist')
            m = verify(source)
            if m and (m['stage'] != self.stage or m.get('compatibility') != self.compatibility):
                raise ValueError('Resume dataset/configuration mismatch')
            self.last = str(source)
            return str(source)
        local = []
        incompatible_local = False
        had_local_checkpoints = False
        for p in self.output.glob('checkpoint-*'):
            if not re.fullmatch(r'checkpoint-\d+', p.name):
                continue
            had_local_checkpoints = True
            try:
                m = verify(p)
                if m and (m['stage'] != self.stage or m.get('compatibility') != self.compatibility):
                    incompatible_local = True
                    continue
                local.append((int(p.name.split('-')[-1]), str(p)))
            except ValueError:
                continue
        local.sort(reverse=True)
        remote = []
        if source in ('remote', 'auto'):
            if not self.enabled:
                raise ValueError('Remote resume requires CHECKPOINT_SYNC_ENABLED=1')
            # Resume failures must never silently turn into fresh training.
            all_remote = self.remote_candidates(compatible_only=False)
            remote = [c for c in all_remote if c[3].get('compatibility') == self.compatibility]
            if all_remote and not remote:
                raise ValueError('Remote checkpoint dataset/configuration mismatch; choose the original configuration or a new experiment')
        print(json.dumps(dict(latest_completed_local_step=local[0][0] if local else None,
                              latest_remote_manifest_step=remote[0][0] if remote else None)))
        choices = [(s, 'local', p) for s, p in local] if source != 'remote' else []
        choices += [(c[0], 'remote', c) for c in remote]
        for _, kind, value in sorted(choices, key=lambda v: (v[0], v[1] == 'local'), reverse=True):
            try:
                self.last = value if kind == 'local' else self.fetch(value)
                print(json.dumps(dict(resume_checkpoint=self.last, verified_completed_step=int(Path(self.last).name.split('-')[-1]))))
                return self.last
            except ValueError:
                if kind == 'local':
                    raise
                self._event(success=False, error='InvalidRemoteCheckpoint')
        if source == 'auto' and not local and not remote:
            if incompatible_local:
                raise ValueError('Local checkpoint dataset/configuration mismatch')
            if had_local_checkpoints:
                raise ValueError('Local checkpoints exist but none are valid; refusing fresh restart')
            return None
        raise FileNotFoundError('No valid compatible completed checkpoint available for requested resume')


def persistent_training(stage):
    """Scope checkpoint hooks to one invocation; smoke and evaluation stay offline."""
    def decorate(fn):
        signature = inspect.signature(fn)
        @wraps(fn)
        def run(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            values = dict(bound.arguments)
            values.update(values.pop('kwargs', {}))
            if values.get('dry_run'):
                return fn(*args, **kwargs)
            output = values.get('output_dir') or f'/kaggle/working/{stage}'
            excluded = {'output_dir', 'export_dir', 'drive_dir', 'resume_from_checkpoint', 'resume', 'max_steps', 'epochs', 'num_epochs', 'save_steps', 'device', 'model', 'tokenizer', 'sft_checkpoint', 'adapter_path', 'dry_run', 'eval_path'}
            contract = {}
            for key, value in values.items():
                if key in excluded:
                    continue
                if key.endswith('_path') and value and Path(str(value)).is_file():
                    value = digest(value)
                if key == 'db_dir' and value:
                    value = {p.relative_to(value).as_posix(): digest(p) for p in sorted(Path(value).rglob('*.sqlite'))}
                contract[key] = value
            manager = CheckpointSync(output, stage, fingerprint(contract))
            source = values.get('resume_from_checkpoint') or values.get('resume')
            resolved = manager.resolve(source)
            if 'resume_from_checkpoint' in bound.arguments:
                bound.arguments['resume_from_checkpoint'] = resolved
            else:
                bound.arguments.setdefault('kwargs', {})['resume_from_checkpoint'] = resolved
            token = ACTIVE_SYNC.set(manager)
            try:
                result = fn(*bound.args, **bound.kwargs)
                if manager.enabled and manager.last:
                    if not (Path(manager.last) / MANIFEST).exists():
                        # Legacy checkpoints are accepted locally; do not modify a
                        # read-only attached artifact just to add backup metadata.
                        warnings.warn('Legacy checkpoint has no sync manifest; save a new training step to back it up.', RuntimeWarning)
                    else:
                        manifest = verify(manager.last)
                        if manifest['experiment_id'] != manager.experiment:
                            warnings.warn('No new updates: source checkpoint retained in its original remote namespace.', RuntimeWarning)
                        else:
                            manager.sync(manager.last)  # Force final save even outside sync interval.
                return result
            finally:
                ACTIVE_SYNC.reset(token)
        return run
    return decorate


def download_checkpoint_artifact(output_dir, stage):
    """Explicit read-only Hub download for handoff/evaluation; never claims namespace.

    Uses CHECKPOINT_HF_REPO_ID/EXPERIMENT_ID, requires enabled sync and a read token.
    Returns the highest integrity-verified checkpoint, without unpickling state.
    """
    backend = CheckpointSync(output_dir, stage, read_only=True)
    if not backend.enabled:
        raise ValueError('Enable checkpoint backend explicitly to download artifacts')
    for candidate in backend.remote_candidates(compatible_only=False):
        try:
            return backend.fetch(candidate)
        except ValueError:
            continue
    raise FileNotFoundError('No verified remote artifact available')
