"""Validate runnable notebook contracts; GPU training is never triggered by CPU tests."""
import ast
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / 'notebooks').glob('0[1-5]_*.ipynb'))


class TestNotebooksIntegrity(unittest.TestCase):
    def test_five_valid_notebooks_and_syntax(self):
        self.assertEqual(len(NOTEBOOKS), 5)
        for p in NOTEBOOKS:
            nb = json.loads(p.read_text())
            self.assertEqual(nb['nbformat'], 4)
            self.assertGreaterEqual(len(nb['cells']), 5)
            for c in nb['cells']:
                if c['cell_type'] == 'code':
                    ast.parse(''.join(c['source']), filename=str(p))
                    self.assertFalse(c.get('outputs'), 'Do not ship stale training results')

    def test_internal_imports_resolve(self):
        count = 0
        for p in NOTEBOOKS:
            for c in json.loads(p.read_text())['cells']:
                if c['cell_type'] != 'code':
                    continue
                for node in ast.walk(ast.parse(''.join(c['source']))):
                    if isinstance(node, ast.ImportFrom) and (node.module or '').startswith('src.'):
                        module = importlib.import_module(node.module)
                        for alias in node.names:
                            self.assertTrue(hasattr(module, alias.name), f'{node.module}:{alias.name}')
                            count += 1
        self.assertGreaterEqual(count, 15)

    def test_all_notebooks_execute_in_explicit_smoke_mode(self):
        for p in NOTEBOOKS:
            with self.subTest(notebook=p.name), tempfile.TemporaryDirectory() as d:
                env = {'PIPELINE_MODE':'smoke', 'KAGGLE_REPO_DIR':str(ROOT),
                       'KAGGLE_DATA_DIR':str(ROOT/'data/mini'), 'KAGGLE_OUTPUT_DIR':d,
                       'INSTALL_DEPS':'0', 'TRAIN_DEVICE':'cpu'}
                # Clear optional production overrides to make the test hermetic.
                clear = ('TRAIN_DATA','RETRIEVAL_EVAL_DATA','DB_DIR','CARDS_PATH','SFT_DATA','GRPO_DATA',
                         'DECISION_DATA','CALIBRATION_DATA','DECISION_TEST_DATA','EVAL_DATA',
                         'RESUME_FROM_CHECKPOINT','GENERATOR_CHECKPOINT','DECISION_CHECKPOINT',
                         'GROUNDING_CHECKPOINT','GRAMMAR_MODE')
                with patch.dict(os.environ, env):
                    for key in clear:
                        os.environ.pop(key, None)
                    namespace = {'__name__':'__main__'}
                    for c in json.loads(p.read_text())['cells']:
                        if c['cell_type'] == 'code':
                            exec(compile(''.join(c['source']), str(p), 'exec'), namespace)
                    self.assertTrue(namespace['SMOKE'])
                    self.assertTrue(Path(namespace['OUTPUT_DIR']).is_relative_to(d))

    def test_no_notebook_local_oracle_or_colab_mount(self):
        for p in NOTEBOOKS:
            source = '\n'.join(''.join(c['source']) for c in json.loads(p.read_text())['cells'] if c['cell_type']=='code')
            self.assertNotIn('gold_sql=gold_sql', source)
            self.assertNotIn('drive.mount', source)
            self.assertNotIn('class PVSQLValidator', source)
            self.assertIn('dry_run=SMOKE' if not p.name.startswith('01') else 'if SMOKE:', source)


if __name__ == '__main__':
    unittest.main()
