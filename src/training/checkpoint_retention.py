"""Opt-in pruning of verified, same-experiment local checkpoints only."""
import os
from pathlib import Path
import re
import shutil


class LocalCheckpointRetention:
    def __init__(self, output, keep, experiment, stage, compatible):
        if type(keep) is not int or keep < 0:
            raise ValueError('Local retention keep must be a nonnegative integer')
        self.keep = keep
        self.output = Path(output).absolute()
        self.experiment, self.stage, self.compatible = experiment, stage, compatible
        self._identity = None
        if keep:
            self._check_root()
            if not shutil.rmtree.avoids_symlink_attacks:
                raise ValueError('Local retention requires symlink-safe shutil.rmtree')
            stat = self.output.stat()
            self._identity = (stat.st_dev, stat.st_ino)

    def _check_root(self):
        if '..' in self.output.parts or self.output == Path(self.output.anchor):
            raise ValueError('Unsafe checkpoint retention output root')
        if any(p.is_symlink() for p in (self.output, *self.output.parents)):
            raise ValueError('Checkpoint retention rejects symlink output paths')
        stat = self.output.stat()
        if not self.output.is_dir() or (self._identity is not None and
                (stat.st_dev, stat.st_ino) != self._identity):
            raise ValueError('Checkpoint retention output root changed')

    def prune_after_verified_upload(self, path, uploaded):
        """Called only by sync after pinned remote manifest AND file verification.

        No disk-pressure deletion. Validate the entire plan before any removal.
        Legacy, incomplete, foreign and incompatible checkpoints are preserved.
        """
        if not self.keep:
            return []
        from src.training.checkpoint_sync import verify
        self._check_root()
        path = Path(path).absolute()
        if path.parent != self.output or path.is_symlink():
            raise ValueError('Uploaded checkpoint is outside retention output root or a symlink')
        if (uploaded is None or uploaded['experiment_id'] != self.experiment or
                uploaded['stage'] != self.stage or not self.compatible(uploaded) or
                path.name != f"checkpoint-{uploaded['step']}" or verify(path) != uploaded):
            raise ValueError('Uploaded checkpoint namespace/configuration mismatch')
        candidates = []
        for child in self.output.iterdir():
            if not re.fullmatch(r'checkpoint-\d+', child.name):
                continue  # Includes all interrupted staging directories.
            if child.is_symlink():
                raise ValueError('Checkpoint retention rejects symlink checkpoints')
            if not child.is_dir() or not (child / 'COMPLETE').exists():
                continue
            manifest = verify(child)  # Also rejects symlink descendants/unsafe inventory.
            if manifest is None:
                continue
            if (manifest['experiment_id'] != self.experiment or manifest['stage'] != self.stage
                    or not self.compatible(manifest)):
                continue
            if any(p.is_mount() or p.stat().st_dev != self.output.stat().st_dev
                   for p in (child, *child.rglob('*')) if p.is_dir()):
                raise ValueError('Checkpoint retention rejects mounted checkpoints')
            candidates.append((manifest['step'], child, manifest))
        candidates.sort(key=lambda item: item[0], reverse=True)
        # Syncing an older checkpoint never authorizes pruning newer local work.
        if not candidates or candidates[0][1] != path:
            return []
        removals = candidates[self.keep:]
        # Revalidate before deletion; refuse the whole plan on a changed inventory.
        self._check_root()
        for _, child, manifest in candidates:
            if child.is_symlink() or verify(child) != manifest:
                raise ValueError('Local checkpoint changed before retention')
        removed = []
        root_fd = os.open(self.output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            stat = os.fstat(root_fd)
            if (stat.st_dev, stat.st_ino) != self._identity:
                raise ValueError('Checkpoint retention output root changed')
            for _, child, _ in removals:
                # fd-relative, symlink-safe deletion cannot follow a substituted link.
                shutil.rmtree(child.name, dir_fd=root_fd)
                removed.append(child.name)
        finally:
            os.close(root_fd)
        return removed
