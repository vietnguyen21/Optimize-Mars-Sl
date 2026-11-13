"""Lightweight environment/bootstrap/authentication gates; no installs or network."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock
from src.training.environment import (profile_path, requirements, validate_environment,
                                      check_disk, check_transfer_disk)
from src.training.kaggle import install_dependencies, KagglePaths
from src.training.hf_auth import hf_token, model_hf_options
from src.training.model_identity import model_identity, verify_identity, file_digest

ROOT = Path(__file__).resolve().parents[1]


class TestKaggleEnvironment(unittest.TestCase):

    def test_bootstrap_import_does_not_load_ml_packages(self):
        import subprocess
        result = subprocess.run([sys.executable, '-B', '-c',
            "import sys; import src.training.kaggle; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_profiles_keep_gpu_packages_component_specific(self):
        core = requirements(profile_path(ROOT, 'system1'))
        grounding = requirements(profile_path(ROOT, 'grounding'))
        generation = requirements(profile_path(ROOT, 'grpo_3b'))
        self.assertNotIn('bitsandbytes', core)
        self.assertNotIn('bitsandbytes', grounding)
        self.assertIn('bitsandbytes', generation)
        self.assertIn('sentence-transformers', grounding)
        self.assertNotIn('trl', generation)
        self.assertNotIn('datasets', generation)
        self.assertEqual(str(core['tokenizers'].specifier), '==0.21.4')
        with self.assertRaises(ValueError): profile_path(ROOT, '../bad')


    def test_loaded_torch_build_must_match_distribution(self):
        from src.training.environment import validate_torch_build
        validate_torch_build(SimpleNamespace(__version__='2.5.1+cu121'),'2.5.1')
        with self.assertRaisesRegex(RuntimeError,'PyTorch build mismatch'):
            validate_torch_build(SimpleNamespace(__version__='2.4.0+cu121'),'2.5.1')
        with self.assertRaisesRegex(RuntimeError,'PyTorch build mismatch'):
            validate_torch_build(SimpleNamespace(__version__='2.5.1+cpu'),'2.5.1+cu121')

    def test_smoke_never_installs_or_imports_gpu_packages(self):
        with patch.dict(os.environ, PIPELINE_MODE='smoke', INSTALL_DEPS='1'), \
             patch('subprocess.check_call', side_effect=AssertionError('pip called')):
            install_dependencies(ROOT, 'grounding')
            self.assertTrue(validate_environment(ROOT, 'grounding', '/not-created', smoke=True)['smoke'])

    def test_installer_rejects_unsupported_python_before_pip(self):
        with patch.dict(os.environ, PIPELINE_MODE='train', INSTALL_DEPS='1'), \
             patch('sys.version_info', (3, 14, 0)), patch('subprocess.check_call') as call:
            with self.assertRaisesRegex(RuntimeError, 'CPython 3.10'):
                install_dependencies(ROOT)
            call.assert_not_called()

    def test_grounding_python_and_torch_floors(self):
        from src.training.environment import validate_python_runtime
        for version, minimum in [((3,10,0),'2.5'), ((3,12,0),'2.5'),
                                 ((3,13,16),'2.6')]:
            with self.subTest(version=version), patch('sys.version_info', version), \
                 patch('sysconfig.get_config_var', return_value=0):
                self.assertEqual(validate_python_runtime('grounding'), minimum)
        for version in [(3,9,0), (3,14,0)]:
            with patch('sys.version_info', version), self.assertRaises(RuntimeError):
                validate_python_runtime('grounding')
        with patch('sys.version_info',(3,13,16)), patch('sysconfig.get_config_var',return_value=1):
            with self.assertRaisesRegex(RuntimeError, 'Free-threaded'):
                validate_python_runtime('grounding')

    def test_grounding_python313_keeps_existing_pins(self):
        with patch('sys.version_info',(3,13,16)):
            reqs = requirements(profile_path(ROOT,'grounding'))
            self.assertEqual(str(reqs['pydantic'].specifier),'==2.11.3')
            self.assertEqual(str(reqs['sentence-transformers'].specifier),'==4.1.0')
            self.assertEqual(str(reqs['transformers'].specifier),'==4.51.3')
            self.assertNotIn('bitsandbytes',reqs)

    def test_grounding_new_python_keeps_resolver_and_snapshot(self):
        for version, torch_version in [((3,13,16),'2.6.0+cu124')]:
            with self.subTest(version=version), tempfile.TemporaryDirectory() as d, \
                 patch.dict(os.environ, PIPELINE_MODE='train', INSTALL_DEPS='1', KAGGLE_OUTPUT_DIR=d), \
                 patch('sys.version_info',version), patch('sysconfig.get_config_var',return_value=0), \
                 patch('src.training.environment.cuda_stack',return_value={'torch':torch_version}), \
                 patch('src.training.environment.requirements',return_value={}), \
                 patch.dict(sys.modules,torch=SimpleNamespace(__version__=torch_version,version=SimpleNamespace(cuda='12.6'),cuda=SimpleNamespace(is_available=lambda:True))), \
                 patch('subprocess.check_call',side_effect=self.bootstrap_context(d,[])) as calls:
                install_dependencies(ROOT,'grounding')
                self.assertEqual(calls.call_count,3)
                dry_run = calls.call_args_list[0].args[0]
                self.assertIn('--only-binary=:all:',dry_run)
                self.assertIn('--dry-run',dry_run)
                self.assertIn('-c',dry_run)
                snapshot=json.loads((Path(d)/'dependency_snapshot.json').read_text())
                self.assertEqual(snapshot['protected_cuda_stack']['torch'],torch_version)
                self.assertIn('python',snapshot)
                self.assertIn('effective_requirements',snapshot)

    def test_grounding_rejects_old_image_torch_before_pip(self):
        for version, torch_version in [((3,13,16),'2.5.1+cu121')]:
            with patch.dict(os.environ, PIPELINE_MODE='train', INSTALL_DEPS='1'), \
                 patch('sys.version_info',version), patch('sysconfig.get_config_var',return_value=0), \
                 patch('src.training.environment.cuda_stack',return_value={'torch':torch_version}), \
                 patch('subprocess.check_call') as calls:
                with self.assertRaisesRegex(RuntimeError,'Select an image'):
                    install_dependencies(ROOT,'grounding')
                calls.assert_not_called()

    def test_preflight_uses_same_new_python_torch_floor(self):
        with patch('sys.version_info',(3,13,16)), patch('sysconfig.get_config_var',return_value=0), \
             patch('src.training.environment.metadata.version',return_value='2.5.1'), \
             patch.dict(sys.modules,torch=SimpleNamespace(__version__='2.5.1')):
            with self.assertRaisesRegex(RuntimeError,'torch >=2.6'):
                validate_environment(ROOT,'grounding','/not-created',device='cpu')
        with patch('sys.version_info',(3,14,0)):
            with self.assertRaisesRegex(RuntimeError,'CPython 3.10–3.12'):
                validate_environment(ROOT,'evaluation','/not-created',device='cpu')

    def test_missing_input_fails_before_model_load(self):
        with tempfile.TemporaryDirectory() as d:
            paths = KagglePaths(ROOT, Path(d), Path(d), False, 'cuda:0')
            with patch.dict(os.environ, AUDIT_MISSING_INPUT=str(Path(d)/'missing')):
                with self.assertRaises(FileNotFoundError): paths.input('AUDIT_MISSING_INPUT', 'x')

    def test_disk_and_transfer_budget_never_delete(self):
        with tempfile.TemporaryDirectory() as d:
            keep = Path(d)/'COMPLETE'; keep.touch()
            with patch('shutil.disk_usage', return_value=SimpleNamespace(free=100)):
                with self.assertRaisesRegex(OSError, 'disk space'):
                    check_disk(d, 101, reserve=0)
                with self.assertRaises(OSError): check_transfer_disk(d, d, 101)
            self.assertTrue(keep.exists())

    def test_model_secret_is_independent_and_never_logged(self):
        with patch.dict(os.environ, MODEL_HF_TOKEN_SECRET='MODEL_ACCESS', MODEL_ACCESS='private-test-token'):
            self.assertEqual(hf_token(model=True, required=True), 'private-test-token')
            self.assertEqual(model_hf_options('owner/model'), {'token':'private-test-token'})
        with tempfile.TemporaryDirectory() as d, patch('src.training.hf_auth.hf_token', side_effect=AssertionError('secret read')):
            self.assertEqual(model_hf_options(d), {'local_files_only':True})

    def test_secret_error_does_not_leak_provider_exception(self):
        secret = SimpleNamespace(UserSecretsClient=Mock(side_effect=RuntimeError('private-sensitive-value')))
        with patch.dict(sys.modules, kaggle_secrets=secret), \
             patch.dict(os.environ, CHECKPOINT_HF_TOKEN_SECRET='MISSING_TEST_SECRET'):
            with self.assertRaisesRegex(RuntimeError, 'credentials unavailable') as error:
                hf_token(required=True)
            self.assertNotIn('private-sensitive-value', str(error.exception))


    def test_invalid_base_identity_is_not_silently_treated_as_legacy(self):
        from src.training.model_identity import read_identity
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'base_identity.json'
            for malformed in ('null','[]','{}'):
                path.write_text(malformed)
                with self.assertRaises(ValueError):read_identity(d)

    def test_local_identity_accepts_relocation_rejects_changed_weights(self):
        with tempfile.TemporaryDirectory() as d:
            a,b = Path(d)/'a',Path(d)/'b';a.mkdir();b.mkdir()
            for p in (a,b):
                (p/'config.json').write_text('{}')
                (p/'model.safetensors').write_bytes(b'exact-base')
            expected,options = model_identity(a)
            self.assertTrue(options['local_files_only'])
            verify_identity(expected, model_identity(b)[0])
            (b/'model.safetensors').write_bytes(b'different-base')
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                verify_identity(expected, model_identity(b)[0])

    def test_hub_revision_and_local_override_verified_without_weight_download(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'config.json').write_text('{}')
            (root/'model.safetensors').write_bytes(b'exact-base')
            entries=[SimpleNamespace(rfilename='config.json', lfs=None),
                     SimpleNamespace(rfilename='model.safetensors', lfs={'sha256':file_digest(root/'model.safetensors')})]
            api=Mock()
            api.model_info.return_value=SimpleNamespace(sha='immutable-commit',siblings=entries)
            downloads=[]
            def download(repo, name, **kwargs):
                downloads.append(name)
                return str(root/name)
            module=SimpleNamespace(HfApi=Mock(return_value=api),hf_hub_download=download)
            with patch.dict(sys.modules, huggingface_hub=module), patch('src.training.hf_auth.hf_token',return_value='secret'):
                identity,options=model_identity('owner/model')
                self.assertEqual(options['revision'],'immutable-commit')
                self.assertEqual(downloads,['config.json'])
                verify_identity(identity, model_identity(root)[0])
                model_identity('owner/model', expected=identity)
                self.assertEqual(api.model_info.call_args.kwargs['revision'],'immutable-commit')

    def bootstrap_context(self, directory, planned):
        """Drive installer using a fake CUDA image and pip resolver; never executes pip."""
        def command(args):
            if '--report' in args:
                Path(args[args.index('--report')+1]).write_text(json.dumps({'install': planned}))
        return command

    def test_installer_rejects_resolver_changes_to_cuda_stack(self):
        planned=[{'metadata':{'name':'torch','version':'2.6.0'}}]
        with tempfile.TemporaryDirectory() as d, \
             patch.dict(os.environ, PIPELINE_MODE='train', INSTALL_DEPS='1', KAGGLE_OUTPUT_DIR=d), \
             patch('sys.version_info',(3,11,0)), \
             patch('src.training.environment.cuda_stack',return_value={'torch':'2.5.1+cu121'}), \
             patch('src.training.environment.requirements',return_value={}), \
             patch.dict(sys.modules,torch=SimpleNamespace(__version__='2.5.1+cu121',version=SimpleNamespace(cuda='12.1'),cuda=SimpleNamespace(is_available=lambda:True))), \
             patch('subprocess.check_call',side_effect=self.bootstrap_context(d, planned)) as calls:
            with self.assertRaisesRegex(RuntimeError,'protected CUDA stack'):
                install_dependencies(ROOT,'system1')
            self.assertEqual(calls.call_count,1)  # dry-run only

    def test_installer_writes_sanitized_snapshot_after_success(self):
        with tempfile.TemporaryDirectory() as d, \
             patch.dict(os.environ, PIPELINE_MODE='train', INSTALL_DEPS='1', KAGGLE_OUTPUT_DIR=d), \
             patch('sys.version_info',(3,11,0)), \
             patch('src.training.environment.cuda_stack',return_value={'torch':'2.5.1+cu121'}), \
             patch('src.training.environment.requirements',return_value={}), \
             patch.dict(sys.modules,torch=SimpleNamespace(__version__='2.5.1+cu121',version=SimpleNamespace(cuda='12.1'),cuda=SimpleNamespace(is_available=lambda:True))), \
             patch('subprocess.check_call',side_effect=self.bootstrap_context(d, [])) as calls:
            install_dependencies(ROOT,'system1')
            self.assertEqual(calls.call_count,3)  # dry-run, install, pip check
            snapshot=json.loads((Path(d)/'dependency_snapshot.json').read_text())
            self.assertEqual(snapshot['protected_cuda_stack']['torch'],'2.5.1+cu121')
            self.assertNotIn('download_info',snapshot)


    def test_installer_refuses_warm_incompatible_import(self):
        from packaging.requirements import Requirement
        with tempfile.TemporaryDirectory() as d, \
             patch.dict(os.environ, PIPELINE_MODE='train', INSTALL_DEPS='1', KAGGLE_OUTPUT_DIR=d), \
             patch('sys.version_info',(3,11,0)), \
             patch('src.training.environment.cuda_stack',return_value={'torch':'2.5.1+cu121'}), \
             patch('src.training.environment.requirements',return_value={'transformers':Requirement('transformers==4.51.3')}), \
             patch('importlib.metadata.version',return_value='4.50.0'), \
             patch.dict(sys.modules,torch=SimpleNamespace(__version__='2.5.1+cu121',version=SimpleNamespace(cuda='12.1'),cuda=SimpleNamespace(is_available=lambda:True)),transformers=SimpleNamespace(__version__='4.50.0')), \
             patch('subprocess.check_call') as calls:
            with self.assertRaisesRegex(RuntimeError,'already imported'):
                install_dependencies(ROOT,'system1')
            calls.assert_not_called()

    def test_installer_refuses_transitive_warm_replacement(self):
        planned=[{'metadata':{'name':'numpy','version':'2.0.0'}}]
        with tempfile.TemporaryDirectory() as d, \
             patch.dict(os.environ, PIPELINE_MODE='train', INSTALL_DEPS='1', KAGGLE_OUTPUT_DIR=d), \
             patch('sys.version_info',(3,11,0)), \
             patch('src.training.environment.cuda_stack',return_value={'torch':'2.5.1+cu121'}), \
             patch('src.training.environment.requirements',return_value={}), \
             patch.dict(sys.modules,torch=SimpleNamespace(__version__='2.5.1+cu121',version=SimpleNamespace(cuda='12.1'),cuda=SimpleNamespace(is_available=lambda:True)),numpy=SimpleNamespace(__version__='1.26.0')), \
             patch('subprocess.check_call',side_effect=self.bootstrap_context(d,planned)) as calls:
            with self.assertRaisesRegex(RuntimeError,'replace loaded packages'):
                install_dependencies(ROOT,'system1')
            self.assertEqual(calls.call_count,1)

    def test_environment_rejects_loaded_disk_version_mismatch(self):
        from packaging.requirements import Requirement
        with patch('sys.version_info',(3,11,0)), \
             patch('sysconfig.get_config_var',return_value=0), \
             patch('src.training.environment.requirements',return_value={'pydantic':Requirement('pydantic==2.11.3')}), \
             patch('importlib.metadata.version',side_effect=lambda name:'2.5.1' if name=='torch' else '2.11.3'), \
             patch('importlib.import_module',return_value=SimpleNamespace(__version__='2.10.0')), \
             patch.dict(sys.modules,torch=SimpleNamespace(__version__='2.5.1',version=SimpleNamespace(cuda='12.1'))):
            with self.assertRaisesRegex(RuntimeError,'Loaded/disk version mismatch'):
                validate_environment(ROOT,'system1','/not-created',device='cpu')


if __name__=='__main__': unittest.main()
