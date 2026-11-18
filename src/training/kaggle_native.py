"""Measure loaded and installed image libraries independently of wheel RECORD claims."""
from functools import lru_cache
import os
from pathlib import Path
import re

from src.training.kaggle_distributions import file_sha256


NATIVE_FILE = re.compile(r'\.so(?:\.|$)')
GPU_LIBRARY = re.compile(
    r'^lib(?:cuda|cudnn|nccl|cu(?:blas|fft|rand|solver|sparse|pti|file|tensor)|'
    r'nv(?:rtc|JitLink|ToolsExt|comp|idia)|torch|c10).*\.so(?:\.|$)', re.IGNORECASE,
)
NATIVE_DIRECTORIES = ('nvidia', 'cuda', 'torch', 'torchvision', 'torchaudio', 'triton')
SYSTEM_CUDA_DIRECTORIES = (
    Path('/usr/lib/x86_64-linux-gnu'), Path('/usr/local/cuda/lib64'),
    Path('/usr/local/cuda/targets/x86_64-linux/lib'),
)


@lru_cache(maxsize=4096)
def _digest(path, device, inode, size, modified, changed):
    return file_sha256(path).hex()


def native_digest(path):
    path = path.resolve(strict=True)
    stat = path.stat()
    return _digest(path, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def mapped_gpu_libraries(maps=None):
    if maps is None:
        maps = Path('/proc/self/maps').read_text()
    result = set()
    for line in maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6 or not fields[5].startswith('/'):
            continue
        filename = fields[5].removesuffix(' (deleted)')
        path = Path(filename)
        if not (GPU_LIBRARY.match(path.name) or
                (NATIVE_FILE.search(path.name) and set(path.parts).intersection(NATIVE_DIRECTORIES))):
            continue
        if filename != fields[5]:
            raise RuntimeError(f'Mapped GPU library was deleted: {path}')
        path = path.resolve(strict=True)
        stat = path.stat()
        device = tuple(int(value, 16) for value in fields[3].split(':'))
        if int(fields[4]) != stat.st_ino or device != (os.major(stat.st_dev), os.minor(stat.st_dev)):
            raise RuntimeError(f'Mapped GPU library no longer matches its on-disk inode: {path}')
        result.add(path)
    return result


def native_inventory(image_sites, mapped):
    candidates = set(mapped)
    for site in image_sites:
        for directory in NATIVE_DIRECTORIES:
            for path in (Path(site) / directory).rglob('*'):
                if NATIVE_FILE.search(path.name) and path.is_file():
                    candidates.add(path.resolve(strict=True))
    for directory in {*SYSTEM_CUDA_DIRECTORIES, *(path.parent for path in mapped)}:
        for path in directory.glob('*.so*'):
            if GPU_LIBRARY.match(path.name) and path.is_file():
                candidates.add(path.resolve(strict=True))
    return {str(path): native_digest(path) for path in sorted(candidates)}


def verify_mapped_libraries(images, mapped=None):
    mapped = mapped_gpu_libraries() if mapped is None else mapped
    observed = {str(path): native_digest(path) for path in sorted(mapped)}
    baseline = images['loaded_libraries']
    approved = images['installed_libraries']
    if (any(observed.get(path) != digest for path, digest in baseline.items())
            or any(approved.get(path) != digest for path, digest in observed.items())):
        raise RuntimeError('Mapped CUDA/PyTorch libraries differ from the recorded image runtime')
    return observed
