"""Local CPU/mocked protocol validation, never evidence of real Kaggle/Hub execution."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid
from src.training.acceptance import (AcceptanceFailure, STAGE, MAX_UPLOAD_BYTES, assert_gpu_profile,
    bundle_proof, component_probe, environment, fixture_contract, make_bundle, parser,
    restore_and_continue, run, save_fixture, train_fixture_step, tree_digest, inspect_imports)
from src.training.checkpoint_sync import CheckpointSync, fingerprint, verify
from tests.test_runtime import real_torch_available


class TestAcceptanceSafety(unittest.TestCase):
    def test_upload_is_off_by_default_and_fresh_namespace_is_mandatory(self):
        args=parser().parse_args([])
        self.assertFalse(args.hub_upload)
        self.assertFalse(args.bootstrap)
        with tempfile.TemporaryDirectory() as d:
            for options in (
                ['--cpu-test','--hub-upload'], ['--cpu-test','--hub-read-check'],
                ['--cpu-test','--bootstrap'], ['--cpu-test','--check-model-secret'],
                ['--phase','session-a','--gpu','t4'],
                ['--phase','session-b','--gpu','t4'],
                ['--cpu-test','--run-id',uuid.uuid4().hex],
                ['--cpu-test','--steps','100'],
            ):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    run(parser().parse_args(options+['--output',d]))
            self.assertEqual(list(Path(d).iterdir()),[])

    def test_t4_and_p100_profiles_are_separate(self):
        assert_gpu_profile({'gpu':'Tesla T4','capability':[7,5]},'t4')
        assert_gpu_profile({'gpu':'Tesla P100-PCIE-16GB','capability':[6,0]},'p100')
        with self.assertRaises(AcceptanceFailure):
            assert_gpu_profile({'gpu':'Tesla P100','capability':[6,0]},'t4')
        with self.assertRaises(AcceptanceFailure):
            assert_gpu_profile({'gpu':'RTX 2080','capability':[7,5]},'t4')

    def test_trl_is_explicitly_optional_for_custom_project_loops(self):
        def version(name):
            if name=='trl':raise importlib.metadata.PackageNotFoundError(name)
            return '1.0'
        with patch('importlib.metadata.version',side_effect=version), \
             patch('importlib.import_module',return_value=SimpleNamespace()):
            imports=inspect_imports('sft_jepa_3b')
            self.assertEqual(imports['trl']['status'],'NOT_INSTALLED_OPTIONAL')
            with self.assertRaisesRegex(AcceptanceFailure,'trl'):
                inspect_imports('sft_jepa_3b',require_trl=True)

    def test_environment_scope_restores_user_training_settings(self):
        with patch.dict(os.environ,CHECKPOINT_EXPERIMENT_ID='real-training',CHECKPOINT_SYNC_ENABLED='1'):
            with environment({'CHECKPOINT_EXPERIMENT_ID':'acceptance-only','CHECKPOINT_SYNC_ENABLED':'0'}):
                self.assertEqual(os.environ['CHECKPOINT_SYNC_ENABLED'],'0')
            self.assertEqual(os.environ['CHECKPOINT_EXPERIMENT_ID'],'real-training')
            self.assertEqual(os.environ['CHECKPOINT_SYNC_ENABLED'],'1')


@unittest.skipUnless(real_torch_available(),'Native PyTorch required for local fixture; GPU/Hub remain NOT EXECUTED')
class TestAcceptanceCpuFixture(unittest.TestCase):
    def setUp(self):
        import torch
        self.threads=torch.get_num_threads();torch.set_num_threads(1)
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(torch.set_num_threads,self.threads)
        self.contract=fixture_contract('cpu','cpu')

    def manager(self, directory='local'):
        with environment({'CHECKPOINT_SYNC_ENABLED':'0','CHECKPOINT_EXPERIMENT_ID':'acceptance-'+uuid.uuid4().hex}):
            manager=CheckpointSync(self.root/directory,STAGE,fingerprint(self.contract))
        manager.configuration=self.contract
        return manager

    def checkpoint(self, manager=None):
        manager=manager or self.manager()
        bundle=make_bundle('cpu')
        for _ in range(2):train_fixture_step(bundle,'cpu')
        return save_fixture(bundle,manager,'cpu',self.contract)

    def test_exact_optimizer_predictor_scheduler_rng_and_cursor_continuation(self):
        checkpoint=self.checkpoint()
        restored,proof=restore_and_continue(checkpoint,'cpu',self.contract)
        self.assertEqual(proof['restored']['step'],2)
        self.assertEqual(proof['restored']['optimizer_steps'],[2])
        self.assertEqual(proof['restored']['scheduler_last_epoch'],2)
        self.assertEqual(proof['restored']['next_data_cursor'],4)
        self.assertEqual(proof['continued']['step'],3)
        self.assertEqual(proof['continued']['next_data_cursor'],6)
        self.assertEqual(proof['continued']['optimizer_steps'],[3])
        self.assertEqual(proof['exact_continuation'],'PASS')
        self.assertEqual(proof['restored']['hashes']['frozen_reference'],proof['continued']['hashes']['frozen_reference'])
        self.assertLess(sum(f['size'] for f in verify(checkpoint)['files'].values()),MAX_UPLOAD_BYTES)
        self.assertEqual(tree_digest(restored['predictor'].state_dict()),proof['continued']['hashes']['predictor_weights'])


    def test_separate_local_process_restores_fixture_without_parent_model_state(self):
        import subprocess,sys
        checkpoint=self.checkpoint()
        code="""import json,sys,torch
from src.training.acceptance import fixture_contract,restore_and_continue
torch.set_num_threads(1)
_,proof=restore_and_continue(sys.argv[1],'cpu',fixture_contract('cpu','cpu'))
print(json.dumps(proof))
"""
        result=subprocess.run([sys.executable,'-B','-c',code,str(checkpoint)],
            cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)['continued']['step'],3)

    def test_incomplete_and_dataset_mismatch_fail_before_resume(self):
        checkpoint=Path(self.checkpoint())
        bad=dict(self.contract,dataset_fingerprint='wrong-dataset')
        with self.assertRaises(ValueError):restore_and_continue(checkpoint,'cpu',bad)
        (checkpoint/'COMPLETE').unlink()
        with self.assertRaises(ValueError):restore_and_continue(checkpoint,'cpu',self.contract)

    def test_report_cannot_claim_gpu_or_hub_from_cpu_execution(self):
        with patch.dict(os.environ,CHECKPOINT_SYNC_ENABLED='1',CHECKPOINT_EXPERIMENT_ID='real-training'), \
             patch.object(CheckpointSync,'_claim',side_effect=AssertionError('Hub accessed')):
            report=run(parser().parse_args(['--cpu-test','--output',str(self.root)]))
        self.assertEqual(report['result'],'PASS_LOCAL_CPU_ONLY')
        self.assertEqual(report['scope'],'LOCAL_CPU_TEST')
        self.assertFalse(report['full_training_allowed'])
        for name in ('environment','nf4','bootstrap','private_hub_round_trip','cross_session_resume','pretrained_component','full_training'):
            self.assertEqual(report['checks'][name]['status'],'NOT_EXECUTED')
        self.assertTrue(Path(report['output'],'acceptance_report.json').exists())
        self.assertEqual(report['checks']['fixture_lifecycle']['details']['continued']['step'],3)

    def test_failure_report_does_not_leak_provider_token(self):
        sentinel='hf_secret_that_must_never_be_in_report'
        args=parser().parse_args(['--gpu','t4','--output',str(self.root)])
        with patch('src.training.environment.validate_environment',side_effect=RuntimeError(sentinel)):
            report=run(args)
        self.assertEqual(report['result'],'FAIL')
        self.assertEqual(report['checks']['environment']['status'],'FAIL')
        self.assertNotIn(sentinel,json.dumps(report))
        self.assertEqual(report['checks']['private_hub_round_trip']['status'],'NOT_EXECUTED')

    def test_private_round_trip_and_new_directory_use_real_backend_with_mocked_hub(self):
        from tests.test_checkpoint_sync import Hub,Add
        hub=Hub(self.root/'cache')
        env={'CHECKPOINT_SYNC_ENABLED':'1','CHECKPOINT_HF_REPO_ID':'team/private',
             'CHECKPOINT_EXPERIMENT_ID':'acceptance-'+uuid.uuid4().hex,
             'CHECKPOINT_SYNC_EVERY_N_SAVES':'1000000','CHECKPOINT_SYNC_FAILURE_POLICY':'stop'}
        with environment(env),patch.dict('sys.modules',huggingface_hub=SimpleNamespace(CommitOperationAdd=Add)):
            sender=CheckpointSync(self.root/'a',STAGE,fingerprint(self.contract),api=hub,download=hub.download,sleep=lambda _:None)
            sender.configuration=self.contract
            checkpoint=self.checkpoint(sender)
            self.assertTrue(sender.sync(checkpoint))
            receiver=CheckpointSync(self.root/'b',STAGE,fingerprint(self.contract),api=hub,download=hub.download,
                                    sleep=lambda _:None,read_only=True)
            downloaded=receiver.resolve('remote')
            self.assertNotEqual(downloaded,checkpoint)
            _,proof=restore_and_continue(downloaded,'cpu',self.contract)
            self.assertEqual(proof['continued']['step'],3)
            self.assertEqual(verify(downloaded),verify(checkpoint))
            self.assertEqual(hub.commits,3)  # owner, stage, publication; reader never writes
            self.assertTrue(all(name.startswith('checkpoints/acceptance-') for name in hub.history[hub.head]))


HF_AVAILABLE=all(importlib.util.find_spec(name) for name in ('torch','transformers','peft','sentence_transformers'))


@unittest.skipUnless(HF_AVAILABLE,'Native tiny-model dependencies required; this is not GPU acceptance')
class TestTinyComponentAcceptance(unittest.TestCase):
    def test_component_architecture_and_project_api_initialization_cpu_only(self):
        import torch
        threads=torch.get_num_threads();torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                for component in ('grounding','sft_jepa_3b','grpo_3b','system1','evaluation'):
                    with self.subTest(component=component):
                        result=component_probe(component,'cpu',Path(directory)/component)
                        self.assertEqual(result['status'],'PASS')
                        self.assertEqual(result['scope'],'TINY_COMPONENT_API_ONLY')
                        if component=='grpo_3b':self.assertEqual(result['immutable_reference_adapter'],'PASS')
                        if component=='evaluation':self.assertEqual(result['full_evaluation'],'NOT_EXECUTED')
        finally:torch.set_num_threads(threads)


if __name__=='__main__':unittest.main()
