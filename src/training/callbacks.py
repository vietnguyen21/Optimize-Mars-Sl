"""
Google Drive Checkpoint Synchronization and Session Recovery Callbacks.

Provides automated checkpoint synchronization between high-speed local runtime SSD
(e.g., Google Colab /content/checkpoints/) and persistent storage (Google Drive
/content/drive/MyDrive/Text2SQL_Colab/checkpoints/), with session recovery
capabilities via `--resume_from_checkpoint latest`.
"""

import os
import re
import shutil
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Provide base callback class compatible with or without HuggingFace transformers
try:
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

    class TrainerCallback:
        """Fallback base TrainerCallback when transformers is not installed."""
        def on_init_end(self, args, state, control, **kwargs):
            pass

        def on_train_begin(self, args, state, control, **kwargs):
            pass

        def on_step_end(self, args, state, control, **kwargs):
            pass

        def on_save(self, args, state, control, **kwargs):
            pass

        def on_train_end(self, args, state, control, **kwargs):
            pass

    class TrainerControl:
        pass

    class TrainerState:
        def __init__(self, global_step: int = 0):
            self.global_step = global_step

    class TrainingArguments:
        def __init__(self, output_dir: str = "/content/checkpoints"):
            self.output_dir = output_dir


def find_latest_checkpoint(search_dir: str) -> Optional[str]:
    """
    Scans a directory for checkpoint subdirectories (e.g. 'checkpoint-1000' or 'checkpoint_1000')
    and returns the path to the one with the highest global step.

    Args:
        search_dir: Directory containing checkpoints.

    Returns:
        Absolute path to the highest step checkpoint, or None if no checkpoints found.
    """
    if not os.path.exists(search_dir) or not os.path.isdir(search_dir):
        return None

    pattern = re.compile(r"^checkpoint[-_](\d+)$")
    candidates = []

    for entry in os.listdir(search_dir):
        full_path = os.path.join(search_dir, entry)
        if os.path.isdir(full_path):
            match = pattern.match(entry)
            if match:
                step = int(match.group(1))
                candidates.append((step, full_path))

    if not candidates:
        # Check for 'best_model' folder
        best_path = os.path.join(search_dir, "best_model")
        if os.path.isdir(best_path):
            return best_path
        return None

    # Sort descending by global step
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def get_resume_checkpoint(
    resume_flag: Optional[str],
    local_dir: str = "/content/checkpoints",
    drive_dir: Optional[str] = None
) -> Optional[str]:
    """
    Resolves the checkpoint path to resume training from.
    If resume_flag is 'latest', checks persistent Google Drive first, then local SSD.
    If resume_flag is a specific path, returns it if it exists.

    Args:
        resume_flag: Value passed to --resume_from_checkpoint ('latest', explicit path, or None).
        local_dir: Local checkpoints directory.
        drive_dir: Persistent Google Drive checkpoints directory.

    Returns:
        Checkpoint path or None if starting from scratch.
    """
    if not resume_flag:
        return None

    flag_clean = str(resume_flag).strip()

    if flag_clean.lower() == "latest":
        # 1. Prefer persistent storage (Google Drive) for crash recovery
        if drive_dir and os.path.exists(drive_dir):
            drive_checkpoint = find_latest_checkpoint(drive_dir)
            if drive_checkpoint:
                logger.info(f"[SessionRecovery] Found latest checkpoint in Drive: {drive_checkpoint}")
                return drive_checkpoint

        # 2. Fall back to local SSD directory
        if os.path.exists(local_dir):
            local_checkpoint = find_latest_checkpoint(local_dir)
            if local_checkpoint:
                logger.info(f"[SessionRecovery] Found latest checkpoint in local storage: {local_checkpoint}")
                return local_checkpoint

        logger.warning(f"[SessionRecovery] 'latest' requested but no checkpoints found in {drive_dir} or {local_dir}")
        return None

    # Specific path provided
    if os.path.exists(flag_clean):
        return flag_clean

    logger.warning(f"[SessionRecovery] Specified checkpoint path not found: {flag_clean}")
    return None


class DriveCheckpointCallback(TrainerCallback):
    """
    TrainerCallback that automatically synchronizes checkpoints from local fast storage
    to persistent Google Drive storage upon every on_save event.
    """

    def __init__(
        self,
        drive_checkpoint_dir: Optional[str] = None,
        local_checkpoint_dir: Optional[str] = None,
        sync_best: bool = True,
        max_drive_checkpoints: Optional[int] = 5,
        drive_dir: Optional[str] = None,
        sync_every_n_steps: Optional[int] = 100,
        **kwargs: Any
    ):
        """
        Args:
            drive_checkpoint_dir: Destination path on Google Drive.
            local_checkpoint_dir: Source directory on local disk (defaults to args.output_dir).
            sync_best: Whether to also sync 'best_model' or final checkpoint on train end.
            max_drive_checkpoints: Maximum number of recent checkpoints to keep in Drive (None = keep all).
            drive_dir: Alias for drive_checkpoint_dir.
            sync_every_n_steps: Step interval for synchronization (defaults to 100).
        """
        super().__init__()
        resolved_drive_dir = drive_dir or drive_checkpoint_dir or "/content/drive/MyDrive/Text2SQL_Colab/checkpoints"
        self.drive_dir = resolved_drive_dir
        self.drive_checkpoint_dir = resolved_drive_dir
        self.local_dir = local_checkpoint_dir
        self.local_checkpoint_dir = local_checkpoint_dir
        self.sync_best = sync_best
        self.max_drive_checkpoints = max_drive_checkpoints
        self.sync_every_n_steps = sync_every_n_steps

        # Ensure drive directory exists or can be created
        try:
            os.makedirs(self.drive_dir, exist_ok=True)
            logger.info(f"[DriveSync] Initialized callback with target Drive directory: {self.drive_dir}")
        except Exception as e:
            logger.warning(f"[DriveSync] Warning: Could not create drive directory '{self.drive_dir}': {e}")

    def _sync_folder(self, src: str, dst: str) -> bool:
        """Copies a checkpoint directory recursively with verification."""
        if not os.path.exists(src):
            logger.warning(f"[DriveSync] Source path does not exist: {src}")
            return False

        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copytree(src, dst, dirs_exist_ok=True)
            
            # Count files and size for verification
            file_count = 0
            total_bytes = 0
            for root, _, files in os.walk(dst):
                for f in files:
                    file_count += 1
                    total_bytes += os.path.getsize(os.path.join(root, f))
            
            size_mb = total_bytes / (1024 * 1024)
            logger.info(f"[DriveSync] ✅ Synchronized '{os.path.basename(src)}' to Drive ({file_count} files, {size_mb:.2f} MB)")
            return True
        except Exception as e:
            logger.error(f"[DriveSync] ❌ Failed to copy '{src}' to '{dst}': {e}")
            return False

    def _prune_old_drive_checkpoints(self):
        """Removes older checkpoints if count exceeds max_drive_checkpoints."""
        if not self.max_drive_checkpoints or self.max_drive_checkpoints <= 0:
            return

        pattern = re.compile(r"^checkpoint[-_](\d+)$")
        checkpoints = []

        if not os.path.exists(self.drive_dir):
            return

        for entry in os.listdir(self.drive_dir):
            full_path = os.path.join(self.drive_dir, entry)
            if os.path.isdir(full_path):
                match = pattern.match(entry)
                if match:
                    step = int(match.group(1))
                    checkpoints.append((step, full_path))

        if len(checkpoints) > self.max_drive_checkpoints:
            checkpoints.sort(key=lambda x: x[0])  # Oldest first
            to_remove = checkpoints[:len(checkpoints) - self.max_drive_checkpoints]
            for step, path in to_remove:
                try:
                    shutil.rmtree(path)
                    logger.info(f"[DriveSync] Pruned old Drive checkpoint step {step}: {path}")
                except Exception as e:
                    logger.warning(f"[DriveSync] Could not prune old checkpoint {path}: {e}")

    def on_save(self, args, state, control, **kwargs):
        """Called by HuggingFace Trainer whenever a checkpoint is saved."""
        output_dir = self.local_dir if self.local_dir else getattr(args, "output_dir", "/content/checkpoints")
        checkpoint_folder = f"checkpoint-{getattr(state, 'global_step', 'latest')}"
        local_path = os.path.join(output_dir, checkpoint_folder)
        drive_path = os.path.join(self.drive_dir, checkpoint_folder)

        if os.path.exists(local_path):
            success = self._sync_folder(local_path, drive_path)
            if success:
                self._prune_old_drive_checkpoints()
        else:
            logger.warning(f"[DriveSync] on_save triggered but local path not found: {local_path}")

        return control

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training to sync the final or best model."""
        if not self.sync_best:
            return control

        output_dir = self.local_dir if self.local_dir else getattr(args, "output_dir", "/content/checkpoints")

        # Check for best_model directory or final checkpoint
        best_local = os.path.join(output_dir, "best_model")
        if os.path.exists(best_local):
            best_drive = os.path.join(self.drive_dir, "best_model")
            self._sync_folder(best_local, best_drive)
        else:
            # Sync final latest checkpoint if available
            latest = find_latest_checkpoint(output_dir)
            if latest:
                folder_name = os.path.basename(latest)
                drive_path = os.path.join(self.drive_dir, folder_name)
                self._sync_folder(latest, drive_path)

        return control
