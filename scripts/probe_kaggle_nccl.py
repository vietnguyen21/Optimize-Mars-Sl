"""Verify a two-device NCCL all-reduce using the reviewed Kaggle application lock."""
from datetime import timedelta
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from src.training.modern_environment import (
        modern_operation, verify_loaded_image_imports, canonical_json,
    )
    from src.training.kaggle_native import verify_mapped_libraries

    rank = int(os.environ['LOCAL_RANK'])
    if (int(os.environ['WORLD_SIZE']) != 2 or rank not in (0, 1)
            or int(os.environ['RANK']) != rank):
        raise RuntimeError('Use one node and exactly two ranks for the NCCL probe')
    output = Path(sys.argv[1]) / f'rank-{rank}'
    report = modern_operation(ROOT, 'grounding', output)
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', timeout=timedelta(seconds=45))
    try:
        tensor = torch.tensor([rank + 1], device=f'cuda:{rank}')
        dist.all_reduce(tensor)
        torch.cuda.synchronize(rank)
        if tensor.item() != 3:
            raise RuntimeError('Unexpected NCCL all-reduce result')
        verify_loaded_image_imports(report['effective_lock'])
        loaded = verify_mapped_libraries(report['runtime']['native_runtime'])
        (output / 'nccl-probe.json').write_text(canonical_json({
            'rank': rank, 'all_reduce_result': tensor.item(), 'loaded_image_libraries': loaded,
            'lock_sha256': report['lock_sha256'],
        }))
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
