"""Distribution precedence and wheel ownership on isolated filesystem fixtures."""
import base64
import hashlib
import os
from importlib.machinery import ModuleSpec
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.training.kaggle_distributions import (
    active_distributions, distribution_groups, duplicate_report, inspect_record_claims,
    ownership_index, verified_origins,
)
from src.training.kaggle_isolation import protected_import_paths, verify_bridge_files
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
            library = self.local / 'nvidia/cuda/libcuda.so'
            library.write_bytes(b'start')
            before = shared_library_fingerprint(inventory, distributions=selected)
            previous = library.stat()
            library.write_bytes(b'drift')
            os.utime(library, ns=(previous.st_atime_ns, previous.st_mtime_ns))
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
        captured = {'protected_packages': {}, 'protected_import_origins': {},
                    'native_runtime': {'installed_libraries': {}}}
        image = {'runtime': captured, 'image_sites': []}
        with (
            patch('src.training.kaggle_isolation.isolated_image', return_value=image),
            patch('src.training.kaggle_isolation.activate_application'),
            patch('src.training.modern_environment.package_inventory', return_value={}),
            patch('src.training.modern_environment.capture_image_baseline',
                  return_value={'runtime': dict(captured, changed=True)}),
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

    def shared_nvidia(self):
        parent = self.local / 'nvidia/__init__.py'
        parent.parent.mkdir()
        parent.write_text('raise RuntimeError("parent must not execute during verification")\n')
        checksum = base64.urlsafe_b64encode(hashlib.sha256(parent.read_bytes()).digest())
        claim = f'nvidia/__init__.py,sha256={checksum.rstrip(b"=").decode()},{parent.stat().st_size}\n'
        for name in ('nvidia-curand-cu12', 'nvidia-nvimgcodec-cu13'):
            self.install(self.local, name, '1', ['nvidia/__init__.py'])
            (self.local / f'{name.replace("-", "_")}-1.dist-info/RECORD').write_text(claim)
        self.install(self.local, 'nvidia-libnvcomp-cu13', '5.2.0.13',
                     ['nvidia/libnvcomp/__init__.py', 'nvidia/libnvcomp/lib/libnvcomp.so'])
        return active_distributions(distribution_groups([str(self.local)]))

    def test_nested_module_with_verified_shared_regular_parent(self):
        selected = self.shared_nvidia()
        with patch.object(sys, 'path', [str(self.local), *sys.path]):
            origins = verified_origins('nvidia-libnvcomp-cu13', selected['nvidia-libnvcomp-cu13'],
                                       ownership=ownership_index(selected))
            links = protected_import_paths(selected, set(selected))
            report = inspect_record_claims('nvidia/__init__.py')
        self.assertEqual(origins, {name: str(self.local / relative) for name, relative in (
            ('nvidia', 'nvidia/__init__.py'), ('nvidia.libnvcomp', 'nvidia/libnvcomp/__init__.py'),
        )})
        self.assertEqual(links[Path('nvidia/__init__.py')], self.local / 'nvidia/__init__.py')
        self.assertEqual(len(report[0]['shared_ownership']['owners']), 2)

    def test_conflicting_shared_hash_or_installed_bytes_are_rejected(self):
        selected = self.shared_nvidia()
        parent = self.local / 'nvidia/__init__.py'
        claim = self.local / 'nvidia_curand_cu12-1.dist-info/RECORD'
        original = claim.read_text()
        with patch.object(sys, 'path', [str(self.local), *sys.path]):
            claim.write_text('nvidia/__init__.py,sha256=wrong,0\n')
            with self.assertRaisesRegex(RuntimeError, 'Conflicting shared RECORD ownership'):
                protected_import_paths(selected, set(selected))
            claim.write_text(original)
            parent.write_bytes(b'unapproved bytes')
            with self.assertRaisesRegex(RuntimeError, 'Conflicting shared RECORD ownership'):
                protected_import_paths(selected, set(selected))

    def test_nested_namespace_parent_and_unrecorded_native_neighbor(self):
        child = 'nvidia/libnvcomp/__init__.py'
        library = 'nvidia/libnvcomp/lib/libnvcomp.so'
        self.install(self.local, 'nvidia-libnvcomp-cu13', '1', [child, library])
        rogue = self.local / 'nvidia/libnvcomp/lib/rogue.so'
        rogue.touch()
        selected = active_distributions(distribution_groups([str(self.local)]))
        with patch.object(sys, 'path', [str(self.local), *sys.path]):
            origins = verified_origins('nvidia-libnvcomp-cu13', selected['nvidia-libnvcomp-cu13'])
            links = protected_import_paths(selected, set(selected))
        self.assertEqual(origins, {'nvidia.libnvcomp': str(self.local / child)})
        self.assertIn(Path(library), links)
        self.assertNotIn(rogue, links.values())
        self.assertFalse(any(path.is_dir() for path in links.values()))
        bridge = self.root / 'bridge'
        for relative, source in links.items():
            path = bridge / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to(source)
        bridged = active_distributions(distribution_groups([str(self.local)]))
        # Fixture METADATA and RECORD are not listed; test the data files alone.
        claims = {bridge / path.relative_to(self.local): claim
                  for path, claim in ownership_index(bridged).items()}
        verify_bridge_files(bridge, set(selected), claims)
        (bridge / rogue.relative_to(self.local)).symlink_to(rogue)
        with self.assertRaisesRegex(RuntimeError, 'unauthorized files'):
            verify_bridge_files(bridge, set(selected), claims)

    def test_missing_nested_child_is_rejected(self):
        selected = self.shared_nvidia()
        (self.local / 'nvidia/libnvcomp/__init__.py').unlink()
        with patch.object(sys, 'path', [str(self.local), *sys.path]):
            with self.assertRaisesRegex(RuntimeError, r'origin mismatch.*nvidia\.libnvcomp'):
                protected_import_paths(selected, set(selected))

    def test_internal_symlink_to_unrecorded_native_file_is_rejected(self):
        self.install(self.local, 'nvidia-cuda', '13.0', ['nvidia/libcuda.so'])
        library = self.local / 'nvidia/libcuda.so'
        library.unlink()
        hidden = self.local / 'nvidia/libhidden.so'
        hidden.touch()
        library.symlink_to(hidden)
        selected = active_distributions(distribution_groups([str(self.local)]))
        with self.assertRaisesRegex(RuntimeError, 'symlink targets an unauthorized file'):
            protected_import_paths(selected, {'nvidia-cuda'})
