"""Retention fixtures use temporary directories and an in-memory Hub only."""
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.training.checkpoint_sync import (
    ACTIVE_SYNC, CheckpointSync, MANIFEST, SyncError, persistent_training, verify,
)
from src.training.environment import check_checkpoint_disk
from src.training.embedding_sampling import EmbeddingSampler, validate_embedding_extra
from test_checkpoint_sync import Add, Hub
from test_runtime import real_torch_available


class TestCheckpointRetention(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.hub = Hub(self.root / 'cache')
        self.env = patch.dict(os.environ, {
            'CHECKPOINT_SYNC_ENABLED': '1', 'CHECKPOINT_HF_REPO_ID': 'team/private',
            'CHECKPOINT_EXPERIMENT_ID': 'grounding-run', 'CHECKPOINT_LOCAL_KEEP': '1',
            'CHECKPOINT_SYNC_FAILURE_POLICY': 'stop', 'CHECKPOINT_SYNC_EVERY_N_SAVES': '1',
            'CHECKPOINT_SYNC_MAX_ATTEMPTS': '1', 'CHECKPOINT_RESUME_SOURCE': 'none',
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        # Restore only this module; broad patch.dict(sys.modules) removes lazy
        # torch imports and can cause duplicate registry initialization later.
        previous_hub = sys.modules.get('huggingface_hub')
        sys.modules['huggingface_hub'] = SimpleNamespace(CommitOperationAdd=Add)
        def restore_hub():
            if previous_hub is None:
                sys.modules.pop('huggingface_hub', None)
            else:
                sys.modules['huggingface_hub'] = previous_hub
        self.addCleanup(restore_hub)
        self.manager = self.backend()

    def backend(self, output='output', **kwargs):
        return CheckpointSync(self.root / output, 'grounding', 'strict-shuffled-config',
            api=self.hub, download=self.hub.download, sleep=lambda _: None, **kwargs)

    def checkpoint(self, step, *, complete=True, foreign=False, legacy=False):
        path = self.manager.output / f'checkpoint-{step}'
        path.mkdir()
        (path / 'training_state.pt').write_bytes(f'step {step}: optimizer scaler RNG'.encode())
        if complete:
            (path / 'COMPLETE').write_text('complete\n')
            if not legacy:
                self.manager.prepare_manifest(path, step, {'contract': {'sampling_strategy': 'epoch_shuffle_v1',
                    'sampling_version': 1, 'grounding_label_mode': 'strict'}})
                if foreign:
                    manifest = json.loads((path / MANIFEST).read_text())
                    manifest['experiment_id'] = 'foreign-experiment'
                    (path / MANIFEST).write_text(json.dumps(manifest))
        return path

    def test_verified_upload_then_prune_and_local_resume(self):
        old = self.checkpoint(1)
        self.assertTrue(self.manager.sync(old))
        new = self.checkpoint(2)
        original = self.manager._verify_remote
        def remote_check(prefix, revision, manifest):
            self.assertTrue(old.exists(), 'Pruned before remote hash verification')
            return original(prefix, revision, manifest)
        with patch.object(self.manager, '_verify_remote', side_effect=remote_check):
            self.assertTrue(self.manager.sync(new))
        self.assertFalse(old.exists())
        self.assertTrue(new.exists())
        self.assertEqual(self.manager.resolve('local'), str(new))
        self.assertEqual(verify(new)['step'], 2)
        # Old remote backups remain immutable and discoverable.
        self.assertEqual([item[0] for item in self.manager.remote_candidates()], [2, 1])

    def test_keep_latest_n_and_always_keep_latest(self):
        with patch.dict(os.environ, CHECKPOINT_LOCAL_KEEP='2'):
            self.manager = self.backend()
        paths = [self.checkpoint(i) for i in (1, 2, 10)]
        self.manager.sync(paths[-1])
        self.assertEqual([p.exists() for p in paths], [False, True, True])
        # Syncing an older step cannot prune a newer unsynced checkpoint.
        newer = self.checkpoint(11)
        self.manager.sync(paths[-1])
        self.assertTrue(paths[-1].exists())
        self.assertTrue(newer.exists())

    def test_only_checkpoint_is_never_deleted(self):
        path = self.checkpoint(1)
        self.manager.sync(path)
        self.manager.sync(path)
        self.assertTrue(path.exists())

    def test_upload_failure_preserves_all_then_retry_recovers(self):
        old, new = self.checkpoint(1), self.checkpoint(2)
        self.hub.fail = 1
        with self.assertRaises(SyncError):
            self.manager.sync(new)
        self.assertTrue(old.exists())
        self.assertTrue(new.exists())
        self.manager.sync(new)
        self.assertFalse(old.exists())
        self.assertTrue(new.exists())

    def test_warn_policy_failure_preserves_all(self):
        old, new = self.checkpoint(1), self.checkpoint(2)
        self.manager.policy = 'warn'
        self.hub.fail = 1
        with self.assertWarns(RuntimeWarning):
            self.assertFalse(self.manager.sync(new))
        self.assertTrue(old.exists())
        self.assertTrue(new.exists())

    def test_remote_checksum_mismatch_preserves_all(self):
        old, new = self.checkpoint(1), self.checkpoint(2)
        download = self.manager.download
        def corrupt(name, revision):
            path = Path(download(name, revision))
            if '/checkpoint-2/' in name and name.endswith('training_state.pt'):
                path.write_bytes(b'corrupt published download')
            return str(path)
        with patch.object(self.manager, 'download', side_effect=corrupt):
            with self.assertRaises(SyncError):
                self.manager.sync(new)
        self.assertTrue(old.exists())
        self.assertTrue(new.exists())

    def test_published_manifest_mismatch_preserves_all(self):
        old, new = self.checkpoint(1), self.checkpoint(2)
        read = self.manager._read
        def corrupt(name, revision):
            manifest = read(name, revision)
            if '/checkpoint-2/' in name and name.endswith(MANIFEST):
                manifest['experiment_id'] = 'wrong-experiment'
            return manifest
        with patch.object(self.manager, '_read', side_effect=corrupt):
            with self.assertRaises(SyncError):
                self.manager.sync(new)
        self.assertTrue(old.exists())
        self.assertTrue(new.exists())

    def test_local_checksum_mismatch_preserves_all(self):
        oldest, old, new = self.checkpoint(0), self.checkpoint(1), self.checkpoint(2)
        (old / 'training_state.pt').write_bytes(b'corrupt old checkpoint')
        with self.assertRaises(SyncError):
            self.manager.sync(new)
        self.assertTrue(oldest.exists())
        self.assertTrue(old.exists())
        self.assertTrue(new.exists())

    def test_cache_disk_failure_prevents_pruning(self):
        old, new = self.checkpoint(1), self.checkpoint(2)
        with patch('shutil.disk_usage', return_value=SimpleNamespace(free=0)):
            with self.assertRaises(SyncError):
                self.manager.sync(new)
        self.assertTrue(old.exists())
        self.assertTrue(new.exists())

    def test_staging_incomplete_legacy_foreign_and_incompatible_are_preserved(self):
        old, new = self.checkpoint(1), self.checkpoint(2)
        foreign = self.checkpoint(3, foreign=True)
        legacy = self.checkpoint(4, legacy=True)
        incomplete = self.checkpoint(5, complete=False)
        incompatible = self.checkpoint(6)
        manifest = json.loads((incompatible / MANIFEST).read_text())
        manifest['compatibility'] = 'different-label-or-sampling-contract'
        (incompatible / MANIFEST).write_text(json.dumps(manifest))
        staging = self.manager.output / '.checkpoint-7-interrupted'
        staging.mkdir()
        (staging / 'partial.weights').write_bytes(b'preserve')
        self.manager.sync(new)
        self.assertFalse(old.exists())
        for path in (new, foreign, legacy, incomplete, incompatible, staging):
            self.assertTrue(path.exists(), str(path))

    def test_namespace_mismatch_upload_never_authorizes_pruning(self):
        old, foreign = self.checkpoint(1), self.checkpoint(2, foreign=True)
        with self.assertRaises(SyncError):
            self.manager.sync(foreign)
        self.assertTrue(old.exists())
        self.assertTrue(foreign.exists())

    def test_default_disabled(self):
        with patch.dict(os.environ):
            os.environ.pop('CHECKPOINT_LOCAL_KEEP')
            self.manager = self.backend()
        old, new = self.checkpoint(1), self.checkpoint(2)
        self.manager.sync(new)
        self.assertIsNone(self.manager.retention)
        self.assertTrue(old.exists())
        self.assertTrue(new.exists())

    def test_invalid_configuration_rejected_before_hub_access(self):
        for value in ('-1', 'invalid', '1.5'):
            with patch.dict(os.environ, CHECKPOINT_LOCAL_KEEP=value), patch.object(self.hub, 'repo_info') as info:
                with self.assertRaises(ValueError):
                    self.backend()
                info.assert_not_called()
        with patch.dict(os.environ, CHECKPOINT_SYNC_ENABLED='0'):
            with self.assertRaises(ValueError):
                self.backend()

    def test_read_only_backend_never_prunes(self):
        manager = self.backend(read_only=True)
        self.assertIsNone(manager.retention)

    def test_repeated_sync_is_idempotent_and_reverifies_remote(self):
        old, new = self.checkpoint(1), self.checkpoint(2)
        self.manager.sync(new)
        commits = self.hub.commits
        manifest = (new / MANIFEST).read_bytes()
        with patch.object(self.manager, '_verify_remote', wraps=self.manager._verify_remote) as verified:
            self.manager.sync(new)
            self.assertTrue(verified.called)
        self.assertEqual(self.hub.commits, commits)
        self.assertEqual((new / MANIFEST).read_bytes(), manifest)
        self.assertFalse(old.exists())
        # Previous success must not conceal a subsequent verification failure.
        older = self.checkpoint(0)
        with patch.object(self.manager, '_verify_remote', side_effect=ValueError('checksum mismatch')):
            with self.assertRaises(SyncError):
                self.manager.sync(new)
        self.assertTrue(older.exists())

    def test_symlink_checkpoint_and_descendant_rejected_before_any_deletion(self):
        old, new = self.checkpoint(1), self.checkpoint(2)
        outside = self.root / 'outside'
        outside.mkdir()
        marker = outside / 'user-file'
        marker.write_text('preserve')
        link = self.manager.output / 'checkpoint-0'
        link.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(SyncError):
            self.manager.sync(new)
        self.assertTrue(old.exists())
        self.assertTrue(marker.exists())
        link.unlink()  # Fixture-only link, never a user checkpoint.
        (old / 'escape').symlink_to(outside, target_is_directory=True)
        with self.assertRaises(SyncError):
            self.manager.sync(new)
        self.assertTrue(old.exists())
        self.assertTrue(marker.exists())

    def test_unsafe_root_and_parent_symlink_rejected(self):
        outside = self.root / 'outside'
        outside.mkdir()
        (self.root / 'alias').symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.backend('alias')
        with self.assertRaises(ValueError):
            self.backend('alias/nested')
        with self.assertRaises(ValueError):
            self.backend('output/../output')

    def test_outside_root_and_unsafe_inventory_refused(self):
        old, new = self.checkpoint(1), self.checkpoint(2)
        from src.training.checkpoint_retention import LocalCheckpointRetention
        other = self.root / 'other'
        other.mkdir()
        retention = LocalCheckpointRetention(other, 1, self.manager.experiment,
                                            self.manager.stage, self.manager.compatible)
        with self.assertRaises(ValueError):
            retention.prune_after_verified_upload(new, verify(new))
        manifest = json.loads((old / MANIFEST).read_text())
        manifest['files']['../outside'] = {'size': 0, 'sha256': '0' * 64}
        (old / MANIFEST).write_text(json.dumps(manifest))
        with self.assertRaises(SyncError):
            self.manager.sync(new)
        self.assertTrue(old.exists())

    def test_changed_root_and_mounted_subtree_refused(self):
        old, new = self.checkpoint(1), self.checkpoint(2)
        nested = old / 'nested'
        nested.mkdir()
        is_mount = Path.is_mount
        with patch.object(Path, 'is_mount', lambda p: p == nested or is_mount(p)):
            with self.assertRaises(SyncError):
                self.manager.sync(new)
        self.assertTrue(old.exists())
        self.assertTrue(new.exists())
        manifest = verify(new)
        moved = self.root / 'moved-output'
        self.manager.output.rename(moved)
        self.manager.output.mkdir()
        with self.assertRaisesRegex(ValueError, 'root changed'):
            self.manager.retention.prune_after_verified_upload(new, manifest)
        self.assertTrue((moved / old.name).exists())
        self.assertTrue((moved / new.name).exists())

    def test_sampling_and_label_contracts_survive_retention_configuration(self):
        @persistent_training('grounding')
        def runner(output_dir, sampling_strategy='epoch_shuffle_v1', grounding_label_mode='strict',
                   seed=42, resume_from_checkpoint=None):
            return ACTIVE_SYNC.get().compatibility
        with patch('src.training.checkpoint_sync.CheckpointSync._claim'), \
             patch('src.training.checkpoint_sync.CheckpointSync.resolve', return_value=None):
            # Injecting fake API prevents construction of a real Hub client.
            with patch('src.training.checkpoint_sync.CheckpointSync', wraps=CheckpointSync) as backend:
                backend.side_effect = lambda *a, **kw: CheckpointSync(*a, **kw, api=self.hub,
                    download=self.hub.download, sleep=lambda _: None)
                enabled = runner(self.manager.output)
                with patch.dict(os.environ, CHECKPOINT_LOCAL_KEEP='0'):
                    self.assertEqual(enabled, runner(self.manager.output))
                self.assertNotEqual(enabled, runner(self.manager.output, sampling_strategy='sequential'))
                self.assertNotEqual(enabled, runner(self.manager.output, grounding_label_mode='legacy'))
        expected = dict(seed=42, sampling_strategy='epoch_shuffle_v1', sampling_version=1,
                        grounding_label_mode='strict')
        validate_embedding_extra({'contract': expected}, {'contract': expected})
        with self.assertRaises(ValueError):
            validate_embedding_extra({'contract': expected},
                                     {'contract': dict(expected, grounding_label_mode='legacy')})
        sampler = EmbeddingSampler(8149, 42, 'epoch_shuffle_v1')
        before = [sampler.index(1100 * 8 + i) for i in range(16)]
        self.manager.sync(self.checkpoint(1100))
        resumed = EmbeddingSampler(8149, 42, 'epoch_shuffle_v1')
        self.assertEqual(before, [resumed.index(1100 * 8 + i) for i in range(16)])

    @unittest.skipUnless(real_torch_available(), 'PyTorch is not installed')
    def test_retained_checkpoint_restores_optimizer_scaler_scheduler_and_rng(self):
        import numpy as np
        import torch
        from src.training.runtime import load_checkpoint, save_checkpoint, seed_everything
        seed_everything(7)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        # Populate optimizer fixture tensors without loading/training any HF model.
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad()
        weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
        scaler = SimpleNamespace(state_dict=lambda: {'scale': 16}, load_state_dict=Mock())
        scheduler = SimpleNamespace(state_dict=lambda: {'last_epoch': 13}, load_state_dict=Mock())
        contract = dict(seed=7, sampling_strategy='epoch_shuffle_v1', sampling_version=1,
                        grounding_label_mode='strict', gradient_accumulation_steps=8)
        old = self.checkpoint(1)
        token = ACTIVE_SYNC.set(self.manager)
        try:
            path = save_checkpoint(self.manager.output, model, optimizer, 13,
                                   scaler=scaler, scheduler=scheduler, extra={'contract': contract})
        finally:
            ACTIVE_SYNC.reset(token)
        self.assertFalse(old.exists())
        expected_python, expected_numpy, expected_torch = random.random(), np.random.rand(), torch.rand(3)
        with torch.no_grad():
            model.weight.fill_(99)
        optimizer.state.clear()
        state = load_checkpoint(path, model, optimizer, scaler=scaler, scheduler=scheduler,
                                expected_extra={'contract': contract}, extra_validator=validate_embedding_extra)
        self.assertEqual(state['step'], 13)
        self.assertEqual(random.random(), expected_python)
        self.assertEqual(np.random.rand(), expected_numpy)
        self.assertTrue(torch.equal(torch.rand(3), expected_torch))
        self.assertTrue(optimizer.state)
        for parameter in optimizer.state.values():
            self.assertEqual(parameter['step'].item(), 1)
            self.assertTrue(torch.count_nonzero(parameter['exp_avg']))
        for key, value in weights.items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]))
        scaler.load_state_dict.assert_called_once_with({'scale': 16})
        scheduler.load_state_dict.assert_called_once_with({'last_epoch': 13})
        self.assertEqual(self.manager.resolve('local'), path)
        with self.assertRaises(ValueError):
            load_checkpoint(path, model, optimizer,
                expected_extra={'contract': dict(contract, grounding_label_mode='legacy')},
                extra_validator=validate_embedding_extra)

    @unittest.skipUnless(real_torch_available(), 'PyTorch is not installed')
    def test_corrupt_staging_manifest_never_published_or_pruned(self):
        import torch
        from src.training.runtime import save_checkpoint
        old = self.checkpoint(1)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        prepare = self.manager.prepare_manifest
        def corrupt(path, step, extra):
            prepare(path, step, extra)
            (Path(path) / 'training_state.pt').write_bytes(b'corrupt staging')
        token = ACTIVE_SYNC.set(self.manager)
        try:
            with patch.object(self.manager, 'prepare_manifest', side_effect=corrupt):
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    save_checkpoint(self.manager.output, model, optimizer, 2)
        finally:
            ACTIVE_SYNC.reset(token)
        self.assertTrue(old.exists())
        self.assertFalse((self.manager.output / 'checkpoint-2').exists())
        self.assertTrue(list(self.manager.output.glob('.checkpoint-2-*')))


class TestCheckpointDiskBudget(unittest.TestCase):
    def test_shared_filesystem_budget_and_export_override(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('src.training.environment.check_disk') as check, \
             patch.dict(os.environ, CHECKPOINT_FINAL_EXPORT_RESERVE_BYTES='300'):
            check_checkpoint_disk(directory, directory, 100, 200)
            check.assert_called_once_with(Path(directory), 500)

    def test_separate_filesystem_budget(self):
        # Fake paths avoid requiring two actual filesystems in the test host.
        output, cache = Mock(), Mock()
        output.stat.return_value.st_dev = 1
        cache.stat.return_value.st_dev = 2
        with patch('src.training.environment.Path', side_effect=[output, cache]), \
             patch('src.training.environment.check_disk') as check, \
             patch.dict(os.environ, CHECKPOINT_FINAL_EXPORT_RESERVE_BYTES='0'):
            check_checkpoint_disk('output', 'cache', 100, 40)
        self.assertEqual(check.call_args_list[0].args, (output, 140))
        self.assertEqual(check.call_args_list[1].args, (cache, 100))

    def test_no_sync_budget_and_nonnegative_validation(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('src.training.environment.check_disk') as check, \
             patch.dict(os.environ, CHECKPOINT_FINAL_EXPORT_RESERVE_BYTES='0'):
            check_checkpoint_disk(directory, None, 100, 40)
            check.assert_called_once_with(Path(directory), 140)
            with self.assertRaises(ValueError):
                check_checkpoint_disk(directory, None, -1, 40)

    def test_low_disk_blocks_save_before_staging_and_preserves_checkpoint(self):
        from src.training.runtime import save_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory) / 'checkpoint-1'
            old.mkdir()
            (old / 'COMPLETE').write_text('preserve')
            model = SimpleNamespace(state_dict=lambda: {})
            optimizer = SimpleNamespace(state_dict=lambda: {})
            with patch.dict('sys.modules', {'torch': SimpleNamespace(), 'numpy': SimpleNamespace()}), \
                 patch('shutil.disk_usage', return_value=SimpleNamespace(free=0)):
                with self.assertRaisesRegex(OSError, 'no checkpoints deleted'):
                    save_checkpoint(directory, model, optimizer, 2)
            self.assertTrue(old.exists())
            self.assertFalse((Path(directory) / 'checkpoint-2').exists())
            self.assertFalse(list(Path(directory).glob('.checkpoint-*')))

    def test_exact_shared_budget_threshold_includes_reserve(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, CHECKPOINT_DISK_RESERVE_BYTES='10',
                        CHECKPOINT_FINAL_EXPORT_RESERVE_BYTES='0'):
            with patch('shutil.disk_usage', return_value=SimpleNamespace(free=249)):
                with self.assertRaises(OSError):
                    check_checkpoint_disk(directory, directory, 100, 40)
            with patch('shutil.disk_usage', return_value=SimpleNamespace(free=250)):
                check_checkpoint_disk(directory, directory, 100, 40)


if __name__ == '__main__':
    unittest.main()
