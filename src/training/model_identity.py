"""Pin Hub revisions and verify relocated local bases without exporting base weights."""
from pathlib import Path
import hashlib
import json
import os
from src.training.hf_auth import model_hf_options

IDENTITY_FILE = 'base_identity.json'


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def model_identity(source, expected=None):
    source = str(source)
    path = Path(source)
    if path.is_dir():
        files = {}
        for p in sorted(path.rglob('*')):
            name = p.relative_to(path).as_posix()
            if p.is_file() and (p.suffix in ('.safetensors', '.bin') or
                               p.name in ('config.json', 'tokenizer.json', 'tokenizer_config.json',
                                          'special_tokens_map.json', 'vocab.json', 'merges.txt', 'modules.json')):
                files[name] = file_digest(p)
        if not files:
            raise ValueError('Local base has no model/config/tokenizer files')
        return {'kind': 'local', 'files': files}, {'local_files_only': True}
    from huggingface_hub import HfApi
    options = model_hf_options(source)
    revision = os.environ.get('MODEL_REVISION') or (expected.get('revision') if expected and expected.get('kind') == 'hub' and expected.get('repo') == source else 'main')
    try:
        info = HfApi(token=options.get('token')).model_info(source, revision=revision, files_metadata=True)
    except Exception as exc:
        status = getattr(getattr(exc, 'response', None), 'status_code', None)
        raise RuntimeError(f'Model access/revision preflight failed ({type(exc).__name__}, HTTP {status}); check model Secret and gated access') from None
    # LFS hashes identify weights without downloading them for preflight. Read
    # only small config/tokenizer files here; normal model loading fetches weights.
    from huggingface_hub import hf_hub_download
    files = {}
    names = [entry.rfilename for entry in info.siblings]
    use_safetensors = any(name.endswith('.safetensors') for name in names)
    for entry in info.siblings:
        name = entry.rfilename
        is_weight = name.endswith('.safetensors') if use_safetensors else name.endswith('.bin')
        if is_weight:
            lfs = getattr(entry, 'lfs', None)
            sha = lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)
            if sha:
                files[name] = sha
        elif Path(name).name in ('config.json', 'tokenizer.json', 'tokenizer_config.json',
                                'special_tokens_map.json', 'vocab.json', 'merges.txt', 'modules.json'):
            try:
                local = hf_hub_download(source, name, revision=info.sha, token=options.get('token'))
            except Exception as exc:
                raise RuntimeError(f'Model metadata download failed ({type(exc).__name__}); check model access/network') from None
            files[name] = file_digest(local)
    return {'kind': 'hub', 'repo': source, 'revision': info.sha, 'files': files}, dict(options, revision=info.sha)


def read_identity(checkpoint):
    path = Path(checkpoint) / IDENTITY_FILE
    if not path.exists():
        return None
    identity = json.loads(path.read_text())
    if not isinstance(identity, dict) or identity.get('kind') not in ('hub', 'local'):
        raise ValueError('Invalid base identity metadata')
    if identity['kind'] == 'hub' and not (identity.get('repo') and identity.get('revision')):
        raise ValueError('Missing immutable Hub base identity')
    files = identity.get('files', {})
    if not isinstance(files, dict) or (identity['kind'] == 'local' and not files):
        raise ValueError('Invalid local base identity inventory')
    return identity


def verify_identity(expected, actual):
    if expected is None:
        return
    if expected.get('kind') == 'hub' and actual.get('kind') == 'local':
        files = expected.get('files', {})
        if files and any(n.endswith(('.safetensors', '.bin')) for n in files) and all(actual.get('files', {}).get(n) == sha for n, sha in files.items()):
            return
    if expected != actual:
        raise ValueError('Base model/tokenizer identity mismatch; use the exact original base/revision')
