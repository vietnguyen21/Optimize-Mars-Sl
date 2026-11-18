"""Contracts for the image-owned runtime and controlled application prefix."""
from importlib import metadata
from importlib.machinery import PathFinder
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.training.kaggle_native import native_digest, mapped_gpu_libraries, verify_mapped_libraries
from src.training.modern_environment import (
    ReviewedImageFinder, application_diagnosis, verify_application_prefix, verify_loaded_image_imports,
)


class TestKaggleApplicationRuntime(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.application = self.root / 'application'
        self.image = self.root / 'image'
        for path in (self.application, self.image):
            path.mkdir()
        self.baseline = {'runtime': {'protected_packages': {'torch': '2.11.0'},
                                     'protected_import_origins': {'torch': [], 'nvidia': []}}}

    def test_sqlglot_precedence_does_not_change_image_process(self):
        for prefix, version in ((self.image, '25.20.2'), (self.application, '26.16.4')):
            (prefix / 'sqlglot').mkdir()
            (prefix / 'sqlglot/__init__.py').write_text(f'__version__ = {version!r}\n')
        code = 'import sys; sys.path[:0] = sys.argv[1:]; import sqlglot; print(sqlglot.__version__)'
        image = [sys.executable, '-I', '-B', '-c', code, str(self.image)]
        modern = [*image[:-1], str(self.application), str(self.image)]
        self.assertEqual(subprocess.check_output(modern, text=True).strip(), '26.16.4')
        self.assertEqual(subprocess.check_output(image, text=True).strip(), '25.20.2')

    def test_prefix_rejects_gpu_distribution_even_with_unrelated_import_name(self):
        info = self.application / 'nvidia_nccl_cu13-1.dist-info'
        info.mkdir()
        (info / 'METADATA').write_text('Name: nvidia-nccl-cu13\nVersion: 1\n')
        with self.assertRaisesRegex(RuntimeError, 'protected distribution'):
            verify_application_prefix(self.application, self.baseline)

    def test_prefix_rejects_file_only_shadow(self):
        for name in ('torch.py', 'nvidia', 'sitecustomize.py'):
            with self.subTest(name=name):
                path = self.application / name
                path.touch()
                with self.assertRaisesRegex(RuntimeError, 'shadows a protected import'):
                    verify_application_prefix(self.application, self.baseline)
                path.unlink()

    def test_prefix_rejects_unrecorded_code_and_gpu_libraries(self):
        directory = self.application / 'unexpected'
        directory.mkdir()
        for name in ('optional.py', 'libnccl.so.2', 'initialize.pth'):
            with self.subTest(name=name):
                path = directory / name
                path.touch()
                with self.assertRaisesRegex(RuntimeError, 'Unapproved application prefix'):
                    verify_application_prefix(self.application, self.baseline)
                path.unlink()

    def test_actually_imported_image_dependency_must_be_in_reviewed_closure(self):
        ibis = self.image / 'ibis.py'
        ibis.touch()
        lock = {'runtime': {'image_sites': [str(self.image)]}, 'packages': {}}
        with self.assertRaisesRegex(RuntimeError, 'Unexpected image dependency imported.*ibis'):
            verify_loaded_image_imports(lock, {'ibis': SimpleNamespace(__file__=str(ibis))})

    def test_excluded_image_conflict_stays_visible_and_active_conflict_is_fatal(self):
        before = {
            'ibis': {'version': '1', 'requires_dist': ['sqlglot<25.21']},
            'sqlglot': {'version': '25.20.2', 'requires_dist': []},
        }
        after = dict(before, sqlglot={'version': '26.16.4', 'requires_dist': []})
        result = application_diagnosis(before, after, ['sqlglot==26.16.4'], set(before))
        self.assertEqual(result['introduced'], [])
        self.assertEqual(result['excluded_image_conflicts'], [('ibis', 'sqlglot<25.21', '26.16.4')])
        result = application_diagnosis(before, after, ['ibis', 'sqlglot==26.16.4'], set(before))
        self.assertEqual(result['excluded_image_conflicts'], [])
        self.assertIn(('ibis', 'sqlglot<25.21', '26.16.4'), result['fatal_active'])

    def test_native_hash_and_actual_mapped_path_are_checked(self):
        library = self.image / 'libnccl.so.2'
        library.write_bytes(b'old')
        stat = library.stat()
        device = f'{os.major(stat.st_dev):x}:{os.minor(stat.st_dev):x}'
        row = f'1-2 r--p 0 {device} {stat.st_ino} {library}'
        observed = mapped_gpu_libraries(row + '\n')
        with self.assertRaisesRegex(RuntimeError, 'on-disk inode'):
            mapped_gpu_libraries(f'1-2 r--p 0 {device} {stat.st_ino + 1} {library}\n')
        expected = {str(library): native_digest(library)}
        images = {'installed_libraries': expected, 'loaded_libraries': expected}
        self.assertEqual(verify_mapped_libraries(images, observed), expected)
        stat = library.stat()
        library.write_bytes(b'new')
        os.utime(library, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with self.assertRaisesRegex(RuntimeError, 'Mapped CUDA/PyTorch libraries differ'):
            verify_mapped_libraries(images, observed)
        with self.assertRaisesRegex(RuntimeError, 'Mapped GPU library was deleted'):
            mapped_gpu_libraries(row + ' (deleted)\n')

    def test_unreviewed_image_modules_are_hidden_from_optional_discovery(self):
        reviewed = self.image / 'reviewed.py'
        reviewed.touch()
        hidden = self.image / 'outside_closure.py'
        hidden.write_text('raise RuntimeError("must never execute")')
        info = self.image / 'reviewed-1.0.dist-info'
        info.mkdir()
        (info / 'METADATA').write_text('Name: reviewed\nVersion: 1.0\n')
        finder = ReviewedImageFinder([str(self.image)], {reviewed})
        finders = [finder if entry is PathFinder else entry for entry in sys.meta_path]
        with patch.object(sys, 'path', [str(self.image), *sys.path]), patch.object(sys, 'meta_path', finders):
            self.assertIsNone(finder.find_spec('outside_closure'))
            self.assertEqual(finder.find_spec('reviewed').origin, str(reviewed))
            self.assertEqual(metadata.version('reviewed'), '1.0')
