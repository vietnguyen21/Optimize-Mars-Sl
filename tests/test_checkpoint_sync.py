"""Hub protocol tests without network, credentials or pretrained downloads."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.training.checkpoint_sync import CheckpointSync, SyncError, MANIFEST, verify


class Add:
    def __init__(self, path_in_repo, path_or_fileobj):
        self.path_in_repo, self.path_or_fileobj = path_in_repo, path_or_fileobj


class Hub:
    def __init__(self, cache):
        self.history = {'0': {}}
        self.head = '0'
        self.private = True
        self.owner = 'member1'
        self.fail = 0
        self.lost_response = False
        self.partial = {}
        self.commits = 0
        self.download_fail = 0
        self.cache = Path(cache)

    def repo_info(self, *args, **kwargs):
        return SimpleNamespace(private=self.private, sha=self.head)

    def whoami(self):
        return {'name': self.owner}

    def list_repo_files(self, *args, revision, **kwargs):
        return list(self.history[revision])

    def create_commit(self, *args, parent_commit, operations, **kwargs):
        if parent_commit != self.head:
            raise OSError('concurrent update')
        payload = {}
        for op in operations:
            v = op.path_or_fileobj
            payload[op.path_in_repo] = v if isinstance(v, bytes) else Path(v).read_bytes()
        if self.fail:
            self.fail -= 1
            self.partial.update(payload)  # Unpublished LFS blobs are not tree files.
            raise ConnectionError('secret=DO_NOT_LOG')
        self.commits += 1
        self.head = str(self.commits)
        self.history[self.head] = dict(self.history[parent_commit], **payload)
        if self.lost_response:
            self.lost_response = False
            raise ConnectionError('response lost after commit')
        return SimpleNamespace(oid=self.head)

    def download(self, name, revision):
        if self.download_fail:
            self.download_fail -= 1
            raise ConnectionError('interrupted download')
        path = self.cache / revision / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.history[revision][name])
        return str(path)


class TestCheckpointSync(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.hub = Hub(self.root / 'cache')
        self.env = patch.dict(os.environ, dict(CHECKPOINT_SYNC_ENABLED='1',
            CHECKPOINT_HF_REPO_ID='team/private', CHECKPOINT_EXPERIMENT_ID='member1-run1',
            CHECKPOINT_SYNC_FAILURE_POLICY='stop', CHECKPOINT_SYNC_EVERY_N_SAVES='1'), clear=False)
        self.env.start(); self.addCleanup(self.env.stop)
        self.imports = patch.dict('sys.modules', {'huggingface_hub': SimpleNamespace(CommitOperationAdd=Add)})
        self.imports.start(); self.addCleanup(self.imports.stop)
        self.sleeps = []
        self.manager = self.new_manager('local')

    def new_manager(self, output):
        return CheckpointSync(self.root / output, 'sft', 'data-config', api=self.hub,
                              download=self.hub.download, sleep=self.sleeps.append)

    def checkpoint(self, step=1, sync=True):
        p = self.manager.output / f'checkpoint-{step}'
        p.mkdir()
        (p / 'COMPLETE').write_text('complete')
        (p / 'training_state.pt').write_bytes(b'optimizer RNG JEPA cursor')
        (p / 'adapter_config.json').write_text('{}')
        (p / 'reference').mkdir()
        (p / 'reference' / 'adapter_model.safetensors').write_bytes(b'immutable SFT')
        if sync:
            self.manager.completed(p, step, {'contract': {'length': 100}})
        return p

    def test_upload_download_new_directory_reference_and_integrity(self):
        p = self.checkpoint()
        new = self.new_manager('new-session')
        downloaded = Path(new.resolve('remote'))
        self.assertNotEqual(p, downloaded)
        self.assertEqual(verify(p), verify(downloaded))
        self.assertEqual((downloaded/'reference/adapter_model.safetensors').read_bytes(), b'immutable SFT')
        self.assertEqual((downloaded/'training_state.pt').read_bytes(), b'optimizer RNG JEPA cursor')

    def test_partial_failure_not_discoverable_and_retry_recovers(self):
        self.hub.fail = 3
        with self.assertRaises(SyncError): self.checkpoint()
        self.assertEqual(self.manager.remote_candidates(), [])
        self.assertEqual(self.sleeps, [1, 2])
        self.assertTrue(self.manager.sync(self.manager.output / 'checkpoint-1'))
        self.assertEqual(self.manager.remote_candidates()[0][0], 1)

    def test_commit_response_lost_is_idempotent(self):
        self.hub.lost_response = True
        # Claim already exists; next commit is the checkpoint commit.
        self.checkpoint()
        self.assertEqual(self.hub.commits, 3)  # owner claim + staged payload + completion publication
        self.assertTrue(self.manager.sync(self.manager.last))
        self.assertEqual(self.hub.commits, 3)

    def test_download_interruption_bounded_retry(self):
        self.checkpoint()
        self.hub.download_fail = 2
        restored = self.new_manager('new').resolve('remote')
        self.assertTrue(Path(restored, 'COMPLETE').exists())
        self.assertEqual(self.sleeps, [1, 2])

    def test_corruption_rejected_before_publish_locally(self):
        self.checkpoint()
        key = next(n for n in self.hub.history[self.hub.head] if '/checkpoint-' in n and n.endswith('training_state.pt'))
        self.hub.history[self.hub.head][key] = b'corrupt'
        with self.assertRaises(FileNotFoundError): self.new_manager('new').resolve('remote')
        self.assertFalse((self.root/'new/checkpoint-1').exists())

    def test_incomplete_and_modified_local_rejected(self):
        p = self.checkpoint(sync=False)
        (p/'COMPLETE').unlink()
        with self.assertRaises(ValueError): self.manager.completed(p, 1, {})
        (p/'COMPLETE').touch()
        self.manager.completed(p, 1, {})
        (p/'training_state.pt').write_bytes(b'changed')
        with self.assertRaises(ValueError): verify(p)

    def test_latest_and_auto_choose_highest_verified_step(self):
        self.checkpoint(2)
        self.checkpoint(10)
        new = self.new_manager('new')
        self.assertEqual(Path(new.resolve('auto')).name, 'checkpoint-10')
        self.assertEqual(Path(new.resolve('local')).name, 'checkpoint-10')
        self.assertIsNone(new.resolve('none'))
        with self.assertRaises(FileNotFoundError): new.resolve(str(self.root/'missing'))

    def test_mismatch_and_experiment_isolation(self):
        p = self.checkpoint()
        self.manager.compatibility = 'other-data'
        with self.assertRaises(ValueError): self.manager.resolve('remote')
        with self.assertRaises(ValueError): self.manager.resolve(str(p))
        with patch.dict(os.environ, CHECKPOINT_EXPERIMENT_ID='member2-run1'):
            other = self.new_manager('other')
            self.assertEqual(other.remote_candidates(), [])

    def test_private_repo_and_owner_are_mandatory(self):
        self.hub.private = False
        with self.assertRaises(PermissionError): self.new_manager('other')
        self.hub.private = True
        self.hub.owner = 'member2'
        with self.assertRaises(PermissionError): self.new_manager('other')

    def test_remote_completed_checkpoint_never_overwritten(self):
        p = self.checkpoint()
        (p/'training_state.pt').write_bytes(b'other-state')
        with self.assertRaises(ValueError): self.manager.completed(p, 1, {})
        self.manager.prepare_manifest(p, 1, {})
        with self.assertRaises(SyncError): self.manager.sync(p)
        self.assertEqual(self.hub.commits, 3)

    def test_warn_policy_preserves_local_and_no_secret_logging(self):
        self.manager.policy = 'warn'
        self.hub.fail = 3
        with self.assertWarns(RuntimeWarning): self.checkpoint()
        self.assertTrue(Path(self.manager.last, 'COMPLETE').exists())
        self.assertNotIn('DO_NOT_LOG', (self.manager.output/'checkpoint_sync.jsonl').read_text())

    def test_malicious_manifest_path_and_disk_full(self):
        self.checkpoint()
        candidate = self.manager.remote_candidates()[0]
        new = self.new_manager('new')
        with patch('shutil.disk_usage', return_value=SimpleNamespace(free=0)):
            with self.assertRaisesRegex(OSError, 'disk space'): new.fetch(candidate)
        candidate[3]['files']['../escape'] = dict(size=0, sha256='0'*64)
        from src.training.checkpoint_sync import validate_manifest
        with self.assertRaises(ValueError): validate_manifest(candidate[3])

    def test_local_corruption_falls_back_to_completed(self):
        self.checkpoint(1)
        p = self.checkpoint(2)
        (p/'training_state.pt').write_bytes(b'bad')
        self.assertEqual(Path(self.manager.resolve('local')).name, 'checkpoint-1')

    def test_separate_sync_interval_and_final_sync(self):
        self.manager.every = 3
        self.checkpoint(1)
        self.assertEqual(self.manager.remote_candidates(), [])
        self.manager.sync(self.manager.last)
        self.assertEqual(self.manager.remote_candidates()[0][0], 1)
        from src.training.checkpoint_sync import ACTIVE_SYNC, finalize_checkpoint_backup
        with patch.dict(os.environ, CHECKPOINT_EXPERIMENT_ID='handoff-with-no-new-updates'):
            receiver = self.new_manager('handoff')
        receiver.last = self.manager.last
        context = ACTIVE_SYNC.set(receiver)
        try:
            with patch.object(receiver, 'sync') as upload:
                finalize_checkpoint_backup()
                upload.assert_not_called()
        finally:
            ACTIVE_SYNC.reset(context)


    def test_cross_account_read_only_download_without_claim(self):
        self.checkpoint()
        self.hub.owner = 'member2'
        backend = CheckpointSync(self.root/'shared', 'sft', 'data-config', api=self.hub,
            download=self.hub.download, sleep=self.sleeps.append, read_only=True)
        restored = backend.resolve('remote')
        self.assertTrue(Path(restored, 'COMPLETE').exists())
        with self.assertRaises(SyncError): backend.sync(restored)
        self.assertEqual(self.hub.commits, 3)

    def test_real_optimizer_rng_scheduler_scaler_and_jepa_remote_restore(self):
        try:
            import torch
            import numpy as np
            if not hasattr(torch, 'optim'):
                self.skipTest('Native PyTorch required')
        except ImportError:
            self.skipTest('Native PyTorch required')
        import random
        from src.training.runtime import save_checkpoint, load_checkpoint, seed_everything
        from src.training.checkpoint_sync import ACTIVE_SYNC
        seed_everything(17)
        model = torch.nn.Linear(2, 1)
        predictor = torch.nn.Linear(1, 1)
        opt = torch.optim.AdamW(list(model.parameters()) + list(predictor.parameters()))
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.9)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        predictor(model(torch.ones(1, 2))).sum().backward()
        opt.step(); scheduler.step(); opt.zero_grad()
        token = ACTIVE_SYNC.set(self.manager)
        try:
            saved = save_checkpoint(self.manager.output, model, opt, 3, scheduler=scheduler,
                scaler=scaler, extra={'predictor': predictor.state_dict(), 'epoch': 2,
                'batch_index': 4, 'contract': {'dataset': 'exact'}})
        finally:
            ACTIVE_SYNC.reset(token)
        expected = (random.random(), np.random.rand(), torch.rand(2))
        restored = self.new_manager('fresh-runtime').resolve('remote')
        model2, predictor2 = torch.nn.Linear(2, 1), torch.nn.Linear(1, 1)
        opt2 = torch.optim.AdamW(list(model2.parameters()) + list(predictor2.parameters()))
        scheduler2 = torch.optim.lr_scheduler.StepLR(opt2, 1, gamma=0.9)
        state = load_checkpoint(restored, model2, opt2, scheduler=scheduler2, scaler=scaler,
                                expected_extra={'contract': {'dataset': 'exact'}})
        predictor2.load_state_dict(state['extra']['predictor'])
        self.assertEqual(state['step'], 3)
        self.assertEqual(state['extra']['batch_index'], 4)
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.rand(), expected[1])
        torch.testing.assert_close(torch.rand(2), expected[2], rtol=0, atol=0)
        for a,b in zip(list(model.parameters())+list(predictor.parameters()),
                       list(model2.parameters())+list(predictor2.parameters())):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(scheduler.state_dict(), scheduler2.state_dict())
        self.assertEqual(opt.state_dict()['param_groups'], opt2.state_dict()['param_groups'])
        for key, value in opt.state_dict()['state'].items():
            for name, tensor in value.items():
                torch.testing.assert_close(tensor, opt2.state_dict()['state'][key][name])
        with self.assertRaises(ValueError):
            load_checkpoint(restored, model2, opt2, expected_extra={'contract': {'dataset': 'different'}})



    def test_malformed_remote_manifest_never_silently_restarts(self):
        self.checkpoint()
        key = next(n for n in self.hub.history[self.hub.head] if n.endswith(MANIFEST))
        for malformed in ('null', '[]', '{"version":1}', '{invalid'):
            self.hub.history[self.hub.head][key] = malformed.encode()
            with self.assertRaises(ValueError): self.new_manager('new').resolve('auto')

    def test_environment_token_and_kaggle_secret_authentication(self):
        import sys
        from io import StringIO
        sentinel = 'hf_private_never_log_this'
        module = SimpleNamespace(CommitOperationAdd=Add)
        received = []
        def api_factory(*, token):
            received.append(token)
            return self.hub
        module.HfApi = api_factory
        module.hf_hub_download = lambda repo, name, **kw: self.hub.download(name, kw['revision'])
        output = StringIO()
        with patch.dict(sys.modules, {'huggingface_hub': module}), patch('sys.stdout', output):
            with patch.dict(os.environ, HF_TOKEN=sentinel):
                backend = CheckpointSync(self.root/'token-env', 'sft', 'data-config')
            class Secrets:
                def get_secret(self, name):
                    self_name.append(name)
                    return sentinel
            self_name = []
            with patch.dict(sys.modules, {'kaggle_secrets': SimpleNamespace(UserSecretsClient=Secrets)}), \
                 patch.dict(os.environ, CHECKPOINT_HF_TOKEN_SECRET='KAGGLE_TEAM_TOKEN'):
                os.environ.pop('KAGGLE_TEAM_TOKEN', None)
                backend = CheckpointSync(self.root/'token-secret', 'sft', 'data-config')
            self.assertEqual(self_name, ['KAGGLE_TEAM_TOKEN'])
            self.assertEqual(received, [sentinel, sentinel])
        self.assertNotIn(sentinel, output.getvalue())
        self.assertNotIn(sentinel.encode(), b''.join(self.hub.history[self.hub.head].values()))

    def test_invalid_stage_and_cross_stage_isolation(self):
        self.checkpoint()
        with self.assertRaises(ValueError):
            CheckpointSync(self.root/'escape', '../escape', api=self.hub)
        backend = CheckpointSync(self.root/'other-stage', 'grounding', 'data-config',
            api=self.hub, download=self.hub.download, sleep=self.sleeps.append)
        self.assertEqual(backend.remote_candidates(), [])



    def test_corrupt_only_local_checkpoint_does_not_restart_auto(self):
        p = self.checkpoint()
        (p/'training_state.pt').write_bytes(b'corrupt')
        # No configured remote candidates remain in this namespace.
        for revision in self.hub.history.values():
            for key in list(revision):
                if '/checkpoint-' in key:
                    del revision[key]
        with self.assertRaisesRegex(ValueError, 'refusing fresh restart'):
            self.manager.resolve('auto')

    def test_concurrent_commit_retries_without_overwriting(self):
        original = self.hub.create_commit
        injected = []
        def concurrent(*args, **kwargs):
            if not injected:
                injected.append(True)
                self.hub.commits += 1
                self.hub.head = str(self.hub.commits)
                self.hub.history[self.hub.head] = dict(self.hub.history['1'], unrelated=b'concurrent stage')
            return original(*args, **kwargs)
        with patch.object(self.hub, 'create_commit', side_effect=concurrent):
            self.checkpoint()
        self.assertTrue(self.manager.remote_candidates())
        self.assertEqual(self.hub.history[self.hub.head]['unrelated'], b'concurrent stage')
        self.assertEqual(self.sleeps, [1])



    def test_local_null_manifest_is_not_legacy(self):
        p = self.checkpoint()
        for malformed in ('null', 'false', '[]', '{}'):
            (p / MANIFEST).write_text(malformed)
            with self.assertRaises(ValueError): verify(p)
            with self.assertRaises(ValueError): self.manager.resolve(str(p))

    def test_staging_verification_failure_never_publishes_completion(self):
        p = self.checkpoint(sync=False)
        self.manager.prepare_manifest(p, 1, {})
        with patch.object(self.manager, '_verify_remote', side_effect=ConnectionError('read interrupted')):
            with self.assertRaises(SyncError): self.manager.sync(p)
        self.assertEqual(self.manager.remote_candidates(), [])
        self.assertFalse(any('/checkpoint-' in n for n in self.hub.history[self.hub.head]))
        self.assertTrue(self.manager.sync(p))
        self.assertEqual(self.manager.remote_candidates()[0][0], 1)
        self.assertEqual(self.hub.commits, 3)

    def test_lost_publication_response_is_idempotent(self):
        original = self.hub.create_commit
        def commit(*args, **kwargs):
            if kwargs.get('commit_message', '').startswith('Publish'):
                self.hub.lost_response = True
            return original(*args, **kwargs)
        with patch.object(self.hub, 'create_commit', side_effect=commit):
            self.checkpoint()
        self.assertEqual(self.hub.commits, 3)
        self.assertEqual(self.manager.remote_candidates()[0][0], 1)

    def test_auto_falls_back_to_verified_local_with_warn(self):
        p = self.checkpoint()
        self.manager.policy = 'warn'
        with patch.object(self.manager, 'remote_candidates', side_effect=SyncError('offline')):
            with self.assertWarns(RuntimeWarning):
                self.assertEqual(self.manager.resolve('auto'), str(p))
        with patch.object(self.manager, 'remote_candidates', return_value=[(99,'unused','sha',{'compatibility':'other'})]):
            self.assertEqual(self.manager.resolve('auto'), str(p))
        self.manager.policy = 'stop'
        with patch.object(self.manager, 'remote_candidates', side_effect=SyncError('offline')):
            with self.assertRaises(SyncError): self.manager.resolve('auto')

    def test_auto_disabled_remote_can_resume_local_but_offline_empty_never_restarts(self):
        p = self.checkpoint()
        self.manager.enabled = False
        self.assertEqual(self.manager.resolve('auto'), str(p))
        new = self.new_manager('empty')
        new.policy = 'warn'
        with patch.object(new, 'remote_candidates', side_effect=SyncError('offline')):
            with self.assertRaises(SyncError): new.resolve('auto')

    def test_effective_generation_defaults_and_aliases_are_canonical(self):
        from src.training.checkpoint_sync import persistent_training, ACTIVE_SYNC
        @persistent_training('sft_jepa_3b')
        def options(output_dir, **kwargs):
            return ACTIVE_SYNC.get().compatibility
        with patch.dict(os.environ, CHECKPOINT_SYNC_ENABLED='0',CHECKPOINT_RESUME_SOURCE='none'):
            self.assertEqual(options(self.root), options(self.root,max_length=1024))
            self.assertEqual(options(self.root), options(self.root,max_length='1024',lambda_jepa=1))
            self.assertEqual(options(self.root), options(self.root,max_seq_length=1024))
            self.assertNotEqual(options(self.root),options(self.root,max_length=512))


if __name__ == '__main__':
    unittest.main()
