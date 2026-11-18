"""Distribution precedence and wheel ownership on isolated filesystem fixtures."""
from importlib.machinery import ModuleSpec
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.training.kaggle_distributions import (
    active_distributions, distribution_groups, duplicate_report, verified_origins,
)
from src.training.kaggle_isolation import protected_import_paths
from src.training.modern_environment import (
    dependency_graph, isolated_runtime, package_inventory, shared_library_fingerprint,
)


class TestKaggleDistributions(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.local = self.root / 'local'
        self.system = self.root / 'system'
        self.local.mkdir()
        self.system.mkdir()

    def install(self, site, name, version, files, *, record=True, requires=()):
        suffix = 'dist-info' if record else 'egg-info'
        info = site / f"{name.replace('-', '_')}-{version}.{suffix}"
        info.mkdir()
        entries = ''.join(f'Requires-Dist: {dep}\n' for dep in requires)
        (info / ('METADATA' if record else 'PKG-INFO')).write_text(f'Name: {name}\nVersion: {version}\n{entries}')
        for file in files:
            path = site / file
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        if record:
            (info / 'RECORD').write_text(''.join(f'{file},,\n' for file in files))
        else:
            # In particular, an alternate listing cannot authorize symlinks.
            (info / 'SOURCES.txt').write_text('\n'.join(files))

    def test_duplicate_six_uses_import_path_rank_even_when_enumeration_is_reversed(self):
        self.install(self.system, 'six', '1.16.0', ['six.py'], record=False)
        self.install(self.local, 'six', '1.17.0', ['six.py'])
        paths = [str(self.local), str(self.system)]
        groups = distribution_groups(paths)
        groups['six'].reverse()
        with patch.object(sys, 'path', paths + sys.path), patch.dict(sys.modules):
            sys.modules.pop('six', None)
            selected = active_distributions(groups)
            report = duplicate_report(groups)['six']
            self.assertEqual(package_inventory(selected)['six']['version'], '1.17.0')
            self.assertEqual(protected_import_paths(selected, {'six'}),
                             {Path('six.py'): self.local / 'six.py'})
            self.assertEqual(report['verified_import_origins'], {'six': str(self.local / 'six.py')})
            self.assertFalse(report['distributions'][0]['wheel_record'])
            self.assertIsNone(report['error'])

    def test_active_missing_record_cannot_fall_back_to_shadowed_wheel(self):
        for name, item in [('six', 'six.py'), ('torch', 'torch/__init__.py')]:
            with self.subTest(name=name):
                self.install(self.system, name, '1.0', [item])
                self.install(self.local, name, '2.0', [item], record=False)
                selected = active_distributions(distribution_groups([str(self.local), str(self.system)]))
                with self.assertRaisesRegex(RuntimeError, f'Active distribution {name}.*RECORD'):
                    protected_import_paths(selected, {name})

    def test_same_rank_packaging_duplicates_are_ambiguous(self):
        self.install(self.local, 'Packaging', '24.0', ['packaging/__init__.py'])
        self.install(self.local, 'packaging', '26.0', ['packaging/__init__.py'])
        groups = distribution_groups([str(self.local)])
        self.assertIn('Ambiguous', duplicate_report(groups)['packaging']['error'])
        with self.assertRaisesRegex(RuntimeError, 'Ambiguous distribution metadata for packaging'):
            active_distributions(groups)

    def test_metadata_cannot_authorize_an_unowned_numpy_or_pydantic_import(self):
        for name in ('numpy', 'pydantic'):
            with self.subTest(name=name):
                file = f'{name}/__init__.py'
                self.install(self.system, name, '2.0', [file])
                local_file = self.local / file
                local_file.parent.mkdir()
                local_file.touch()
                paths = [str(self.local), str(self.system)]
                selected = active_distributions(distribution_groups(paths))
                with patch.dict(sys.modules):
                    sys.modules.pop(name, None)
                    with self.assertRaisesRegex(RuntimeError, 'Distribution/import origin mismatch'):
                        verified_origins(name, selected[name], paths)
                    self.assertEqual(verified_origins(name, selected[name], paths[1:]),
                                     {name: str(self.system / file)})

    def test_torch_and_cuda_fingerprint_follow_selected_origins(self):
        self.install(self.local, 'torch', '2.11.0', ['torch/__init__.py', 'torch/lib/libtorch.so'])
        self.install(self.local, 'nvidia-cuda', '13.0', ['nvidia/cuda/libcuda.so'])
        self.install(self.system, 'torch', '1.0', ['torch/__init__.py'], record=False)
        paths = [str(self.local), str(self.system)]
        selected = active_distributions(distribution_groups(paths))
        inventory = package_inventory(selected)
        with patch.object(sys, 'path', paths), patch.dict(sys.modules):
            sys.modules.pop('torch', None)
            self.assertEqual(verified_origins('torch', selected['torch']),
                             {'torch': str(self.local / 'torch/__init__.py')})
            before = shared_library_fingerprint(inventory, distributions=selected)
            (self.local / 'nvidia/cuda/libcuda.so').write_bytes(b'drift')
            self.assertNotEqual(before, shared_library_fingerprint(inventory, distributions=selected))

    def test_modern_sqlglot_is_distinct_and_closure_is_checked(self):
        file = 'sqlglot/__init__.py'
        self.install(self.system, 'sqlglot', '25.20.2', [file])
        self.install(self.local, 'sqlglot', '26.16.4', [file])
        self.install(self.local, 'example', '1.0', ['example.py'], requires=['absent>=1'])
        paths = [str(self.local), str(self.system)]
        selected = active_distributions(distribution_groups(paths))
        with patch.dict(sys.modules):
            sys.modules.pop('sqlglot', None)
            self.assertEqual(verified_origins('sqlglot', selected['sqlglot'], paths),
                             {'sqlglot': str(self.local / file)})
        _, problems = dependency_graph(package_inventory(selected), ['example', 'sqlglot==26.16.4'])
        self.assertEqual(problems, [('example', 'absent>=1', '<missing>')])

    def test_image_fingerprint_drift_remains_fatal(self):
        captured = {'protected_packages': {}, 'protected_library_fingerprint': 'before'}
        with patch('src.training.kaggle_isolation.isolated_image', return_value={'runtime': captured}), (
            patch('src.training.modern_environment.runtime_identity',
                  return_value=dict(captured, protected_library_fingerprint='after'))
        ):
            with self.assertRaisesRegex(RuntimeError, 'CUDA fingerprint'):
                isolated_runtime({})

    def test_loaded_module_from_old_distribution_is_rejected(self):
        file = 'pydantic/__init__.py'
        self.install(self.system, 'pydantic', '1.0', [file], record=False)
        self.install(self.local, 'pydantic', '2.0', [file])
        paths = [str(self.local), str(self.system)]
        selected = active_distributions(distribution_groups(paths))
        old = SimpleNamespace(__spec__=ModuleSpec('pydantic', None, origin=str(self.system / file)))
        with patch.dict(sys.modules, {'pydantic': old}):
            with self.assertRaisesRegex(RuntimeError, 'Distribution/import origin mismatch'):
                verified_origins('pydantic', selected['pydantic'], paths)

    def test_protected_record_cannot_bridge_outside_installation(self):
        self.install(self.local, 'nvidia-cuda', '13.0', ['nvidia/libcuda.so'])
        library = self.local / 'nvidia/libcuda.so'
        library.unlink()
        external = self.system / 'libcuda.so'
        external.touch()
        library.symlink_to(external)
        selected = active_distributions(distribution_groups([str(self.local)]))
        with self.assertRaisesRegex(RuntimeError, 'outside its installation'):
            protected_import_paths(selected, {'nvidia-cuda'})
