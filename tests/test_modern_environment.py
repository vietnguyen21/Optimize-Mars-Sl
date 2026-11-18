"""M0 contract tests using synthetic inventory and resolver metadata."""
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.training.environment import profile_path, requirements, selected_profile
from src.training.kaggle import install_dependencies, validate_modern_notebook_environment
from src.training.kaggle_isolation import isolated_image, protected_import_paths
from src.training import modern_environment as modern

ROOT = Path(__file__).resolve().parents[1]


def package(version, *deps):
    return {'version': version, 'requires_dist': list(deps)}


class TestModernEnvironment(unittest.TestCase):
    def test_opt_in_profile_and_scope(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(selected_profile(), 'LEGACY')
            self.assertEqual(profile_path(ROOT, 'grounding').name, 'kaggle-grounding.txt')
        roots = requirements(profile_path(ROOT, 'grounding', profile='MODERN'))
        self.assertEqual(str(roots['transformers']), 'transformers==5.20.2')
        self.assertEqual(str(roots['sentence-transformers']), 'sentence-transformers==5.3.0')
        self.assertNotIn('bitsandbytes', roots)
        self.assertEqual(str(roots['pydantic'].specifier), '==2.14.1')
        self.assertEqual(str(roots['networkx'].specifier), '==3.7')
        self.assertEqual(str(roots['sqlglot'].specifier), '==26.16.4')
        with self.assertRaises(ValueError):
            profile_path(ROOT, 'evaluation', profile='MODERN')
        with self.assertRaises(ValueError):
            selected_profile('anything-else')

    def test_resolves_extras_markers_and_transitive_conflicts(self):
        inventory = {
            'root': package('1', 'plugin[extra]', 'ignored; python_version < "3.0"'),
            'plugin': package('1', 'leaf>=2; extra == "extra"'),
            'leaf': package('1'),
        }
        active, issues = modern.dependency_graph(inventory, ['root'])
        self.assertEqual(active, {'root', 'plugin', 'leaf'})
        self.assertEqual(issues, [('plugin', 'leaf>=2; extra == "extra"', '1')])

    def test_conflict_classes(self):
        before = {
            'app': package('1', 'shared>=1'),
            'shared': package('1'),
            'baseline': package('1', 'missing'),
            'other': package('1', 'shared<2'),
        }
        after = dict(before, shared=package('2'), app=package('2', 'absent'))
        result = modern.classify(before, after, ['app'])
        self.assertEqual(result['fatal_active'], [('app', 'absent', '<missing>')])
        self.assertEqual(result['unrelated_preinstalled'], [('baseline', 'missing', '<missing>')])
        self.assertIn(('other', 'shared<2', '2'), result['introduced'])

    def test_protects_cuda_distribution_variants(self):
        names = ['torch', 'torchvision', 'torchaudio', 'triton', 'nvidia-nccl-cu13',
                 'cuda-bindings', 'cudf-cu13', 'pytorch-triton-rocm']
        self.assertTrue(all(modern.protected_name(name) for name in names))
        self.assertFalse(modern.protected_name('transformers'))

    def test_rejects_unreviewed_lock_and_image_drift(self):
        roots = sorted(str(r) for r in requirements(profile_path(ROOT, 'core', profile='MODERN')).values())
        runtime = {'image': 'observed'}
        payload = dict(schema_version=1, profile='MODERN', stage='core', runtime=runtime, roots=roots)
        with tempfile.TemporaryDirectory() as directory, patch.object(modern, 'require_target'):
            lock = Path(directory) / 'lock.json'
            lock.write_text(json.dumps(payload))
            with patch.dict(os.environ, {'KAGGLE_MODERN_LOCK': str(lock),
                                         'KAGGLE_MODERN_LOCK_SHA256': 'wrong'}):
                with self.assertRaisesRegex(RuntimeError, 'SHA-256'):
                    modern.checked_lock(ROOT, 'core', runtime)
            with patch.dict(os.environ, {'KAGGLE_MODERN_LOCK': str(lock),
                                         'KAGGLE_MODERN_LOCK_SHA256': modern.sha256(lock.read_bytes())}):
                self.assertEqual(modern.checked_lock(ROOT, 'core', runtime)[0], payload)
                with self.assertRaisesRegex(RuntimeError, 'running image'):
                    modern.checked_lock(ROOT, 'core', {'image': 'new'})

    def test_bootstrap_fails_before_installer_for_warm_dependency(self):
        runtime = {'protected_packages': {}}
        lock = {'roots': ['scikit-learn==2'], 'runtime': runtime,
                'packages': {'scikit-learn': package('2')}}
        before = {'scikit-learn': package('1')}
        with tempfile.TemporaryDirectory() as directory, (
            patch.object(modern, 'package_inventory', return_value=before)
        ), patch.object(modern, 'isolated_runtime', return_value=(runtime, {'returncode': 0})), (
            patch.object(modern, 'checked_lock', return_value=(lock, 'digest'))
        ), patch.object(modern, 'pip_check', return_value={'returncode': 0}), (
            patch.object(modern, 'repository_commit', return_value={'commit': 'a' * 40})
        ), patch.object(modern, 'active_distributions', return_value={'scikit-learn': object()}), (
            patch.object(modern, 'record_files', return_value=[PurePosixPath('sklearn/__init__.py')])
        ), patch.dict(sys.modules, {'sklearn': object()}), patch.object(modern, 'report_plan') as resolver:
            with self.assertRaisesRegex(RuntimeError, 'already loaded'):
                modern.modern_operation(ROOT, 'grounding', directory, install=True)
            resolver.assert_not_called()

    def test_preflight_records_unrelated_global_check_without_ignoring_active_failure(self):
        runtime = {'protected_packages': {}, 'protected_library_fingerprint': 'native'}
        lock = {'roots': ['project==1'], 'runtime': runtime, 'packages': {'project': package('1')}}
        inventory = {'project': package('1'), 'unrelated': package('1', 'missing')}
        diagnostics = {'returncode': 1, 'stdout': 'unrelated 1 requires missing, which is not installed.\n',
                       'stderr': ''}
        with tempfile.TemporaryDirectory() as directory, (
            patch.object(modern, 'package_inventory', return_value=inventory)
        ), patch.object(modern, 'isolated_runtime', return_value=(runtime, {'returncode': 0})), (
            patch.object(modern, 'checked_lock', return_value=(lock, 'digest'))
        ), patch.object(modern, 'pip_check', return_value=diagnostics), (
            patch.object(modern, 'repository_commit', return_value={'commit': 'a' * 40})
        ), patch.object(modern, 'shared_library_fingerprint', return_value='native'), (
            patch.object(modern, 'verify_application_imports')
        ):
            modern.modern_operation(ROOT, 'grounding', directory)
            report = json.loads((Path(directory) / 'modern-preflight.json').read_text())
            self.assertEqual(report['global_pip_after'], diagnostics)
            self.assertEqual(report['lock_sha256'], 'digest')
            self.assertTrue(report['conflicts']['unrelated_preinstalled'])
            inventory['project'] = package('1', 'missing')
            with self.assertRaisesRegex(RuntimeError, 'dependency conflicts'):
                modern.modern_operation(ROOT, 'grounding', directory)

    def test_smoke_does_not_install(self):
        with (
            patch.dict(os.environ, {'PIPELINE_MODE': 'smoke', 'INSTALL_DEPS': '1'}),
            patch.object(modern, 'modern_operation') as operation,
        ):
            install_dependencies(ROOT, 'grounding', profile='MODERN')
            operation.assert_not_called()

    def test_resolver_wheel_must_be_hashed_binary(self):
        entry = {'metadata': {'name': 'example', 'version': '1'},
                 'download_info': {'url': 'https://example.org/example-1-py3-none-any.whl',
                                   'archive_info': {'hashes': {'sha256': 'a' * 64}}}}
        self.assertEqual(modern.wheel_record(entry)['sha256'], 'a' * 64)
        entry['download_info']['archive_info'] = {}
        with self.assertRaisesRegex(RuntimeError, 'hashed binary'):
            modern.wheel_record(entry)

    def test_shared_namespace_exposes_only_borrowed_distribution(self):
        with tempfile.TemporaryDirectory() as directory:
            site = Path(directory)
            (site / 'namespace').mkdir()
            torch_file = site / 'namespace/kernels.so'
            torch_file.touch()
            (site / 'namespace/other.py').touch()

            def dist(name, path):
                return SimpleNamespace(metadata={'Name': name}, version='1', read_text=lambda _: f'{path},,\n',
                                       locate_file=lambda value: site / value)

            protected = dist('torch', 'namespace/kernels.so')
            unrelated = dist('unrelated', 'namespace/other.py')
            self.assertEqual(protected_import_paths({'torch': protected, 'unrelated': unrelated}, {'torch'}),
                             {Path('namespace/kernels.so'): torch_file})
            with self.assertRaisesRegex(RuntimeError, 'shares ownership'):
                protected_import_paths(
                    {'torch': protected, 'unrelated': dist('unrelated', 'namespace/kernels.so')}, {'torch'},
                )

    def test_normal_process_cannot_install_into_host(self):
        flags = SimpleNamespace(isolated=False, dont_write_bytecode=True)
        with patch.object(sys, 'flags', flags):
            with self.assertRaisesRegex(RuntimeError, 'venv Python'):
                isolated_image()

    def test_notebook_runs_install_and_preflight_in_separate_isolated_processes(self):
        with patch.dict(os.environ, {'KAGGLE_MODERN_PYTHON': '/tmp/test-venv/bin/python',
                                     'INSTALL_DEPS': '1'}), patch('src.training.kaggle.subprocess.run') as run:
            validate_modern_notebook_environment(ROOT, 'grounding', '/tmp/report', 'cuda:1')
        self.assertEqual(len(run.call_args_list), 2)
        install, validate = [call.args[0] for call in run.call_args_list]
        self.assertEqual(install, validate + ['--install-modern-lock'])
        self.assertEqual(validate[:3], ['/tmp/test-venv/bin/python', '-I', '-B'])
        self.assertEqual(validate[-2:], ['--device', 'cuda:1'])
