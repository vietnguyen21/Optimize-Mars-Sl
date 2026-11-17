"""Tiny CPU model and in-memory Hub: no pretrained models or real HF access."""
from contextlib import ExitStack
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.training.checkpoint_sync import CheckpointSync, MANIFEST, SyncError, verify
from src.training.embedding_sampling import EmbeddingSampler
from src.training.embedding_validation import isolated_validation, validation_contract
from test_checkpoint_sync import Add, Hub
from test_runtime import real_torch_available

HAS_TORCH = real_torch_available()
if HAS_TORCH:
    import numpy as np
    import torch
    from src.training import train_embedding as embedding

    class TinyEmbedding(torch.nn.Module):
        instances = []

        def __init__(self, *args, **kwargs):
            super().__init__()
            self.layer = torch.nn.Linear(4, 4)
            self.dropout = torch.nn.Dropout(0.25)
            self.queries = []
            self.instances.append(self)

        def __getitem__(self, index):
            return SimpleNamespace(auto_model=SimpleNamespace(gradient_checkpointing_enable=lambda: None))

        def features(self, texts):
            result = []
            for text in texts:
                values = [0.1] * 4
                for i, char in enumerate(text):
                    values[i % 4] += (ord(char) % 31) / 100
                result.append(values)
            return torch.tensor(result)

        def tokenize(self, texts):
            self.queries.append(texts[0])
            return {'features': self.features(texts)}

        def forward(self, inputs):
            return {'sentence_embedding': self.dropout(self.layer(inputs['features']))}

        def encode(self, texts, *, convert_to_tensor=False, **kwargs):
            self.eval()  # Match the SentenceTransformer encode mode side effect.
            with torch.no_grad():
                vectors = torch.nn.functional.normalize(self.layer(self.features(texts)), dim=-1)
            return vectors if convert_to_tensor else vectors.numpy()

        def save(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            Path(path, 'tiny-export.json').write_text('{"fixture":true}')


@unittest.skipUnless(HAS_TORCH, 'CPU PyTorch required')
class TestEmbeddingSessions(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.hub = Hub(self.root / 'cache')
        self.stack.enter_context(patch.dict(os.environ, {
            'CHECKPOINT_SYNC_ENABLED': '1', 'CHECKPOINT_HF_REPO_ID': 'team/private',
            'CHECKPOINT_EXPERIMENT_ID': 'session-fixture', 'CHECKPOINT_LOCAL_KEEP': '0',
            'CHECKPOINT_SYNC_FAILURE_POLICY': 'stop', 'CHECKPOINT_SYNC_EVERY_N_SAVES': '1',
            'CHECKPOINT_SYNC_MAX_ATTEMPTS': '1', 'CHECKPOINT_RESUME_SOURCE': 'none',
        }))
        # Restore only these entries, preserving lazy torch module imports.
        for name, module in {'sentence_transformers': SimpleNamespace(SentenceTransformer=TinyEmbedding),
                             'huggingface_hub': SimpleNamespace(CommitOperationAdd=Add)}.items():
            previous = sys.modules.get(name)
            sys.modules[name] = module
            def restore(name=name, previous=previous):
                if previous is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous
            self.stack.callback(restore)
        self.managers = []
        def backend(*args, **kwargs):
            result = CheckpointSync(*args, **kwargs, api=self.hub, download=self.hub.download, sleep=lambda _: None)
            self.managers.append(result)
            return result
        self.stack.enter_context(patch('src.training.checkpoint_sync.CheckpointSync', side_effect=backend))
        identity = dict(kind='hub', repo='Qwen/Qwen3-Embedding-0.6B', revision='tiny-pinned-base', files={})
        self.stack.enter_context(patch('src.training.model_identity.model_identity', return_value=(identity, {})))
        self.cards = self.root / 'cards.jsonl'
        self.cards.write_text(''.join(json.dumps(dict(db_id='db', table_name='t', column_name=f'c{i}')) + '\n'
                                      for i in range(4)))
        self.train = self.root / 'train.jsonl'
        self.train.write_text(''.join(json.dumps(dict(db_id='db', question=f'train-{i}',
            SQL='SELECT c0 FROM t', positive_columns=['t.c0'])) + '\n' for i in range(7)))
        self.evaluation = self.root / 'validation.jsonl'
        self.evaluation.write_text(''.join(json.dumps(dict(db_id='db', question=f'validation-{i}',
            SQL='SELECT c1 FROM t', positive_columns=['t.c1'])) + '\n' for i in range(2)))
        TinyEmbedding.instances.clear()

    def run_session(self, output='run', **kwargs):
        options = dict(cards_path=str(self.cards), bird_path=str(self.train),
            output_dir=str(self.root / output), device='cpu', seed=17, max_length=16,
            sampling_strategy='epoch_shuffle_v1', gradient_accumulation_steps=2,
            save_steps=2, global_max_steps=6, session_stop_step=3)
        options.update(kwargs)
        return embedding.run_embedding_training(**options)

    def state(self, output, step):
        return torch.load(self.root / output / f'checkpoint-{step}' / 'training_state.pt',
                          weights_only=False, map_location='cpu')

    def metrics(self, output='run'):
        return [json.loads(line) for line in (self.root / output / 'metrics.jsonl').read_text().splitlines()]

    def periodic(self, **kwargs):
        options = dict(eval_path=str(self.evaluation), eval_steps=2, evaluation_kind='internal_validation')
        options.update(kwargs)
        return self.run_session(**options)

    def test_interim_has_verified_checkpoint_without_any_final_artifact(self):
        result = self.run_session()
        self.assertEqual((result['status'], result['step']), ('session_stopped', 3))
        self.assertIsNone(result['model_path'])
        self.assertIsNone(result['index'])
        self.assertFalse((self.root / 'run' / 'final').exists())
        self.assertFalse(list((self.root / 'run').glob('offline_column_index*')))
        self.assertEqual(verify(result['checkpoint_path'])['step'], 3)
        self.assertEqual(self.managers[-1].remote_candidates()[0][0], 3)

    def test_multiple_sessions_equal_continuous_parameters_optimizer_rng_and_order(self):
        with patch.dict(os.environ, CHECKPOINT_EXPERIMENT_ID='continuous-fixture'):
            self.run_session('continuous', session_stop_step=6, gradient_accumulation_steps=8)
        continuous_queries = list(TinyEmbedding.instances[-1].queries)
        expected = self.state('continuous', 6)
        self.run_session('first', gradient_accumulation_steps=8)
        first_queries = list(TinyEmbedding.instances[-1].queries)
        # New output/session, exclusively restored from the fake remote Hub.
        result = self.run_session('second', session_stop_step=6, resume_from_checkpoint='remote',
                                  gradient_accumulation_steps=8)
        second_queries = list(TinyEmbedding.instances[-1].queries)
        actual = self.state('second', 6)
        self.assertEqual(result['step'], 6)
        self.assertEqual(first_queries + second_queries, continuous_queries)
        sampler = EmbeddingSampler(7, 17, 'epoch_shuffle_v1')
        self.assertEqual(continuous_queries,
            [f'Query: train-{sampler.index(i)} Evidence:' for i in range(48)])
        self.assertEqual(actual['rng_python'], expected['rng_python'])
        self.assertTrue(torch.equal(actual['rng_torch'], expected['rng_torch']))
        for name in actual['model']:
            self.assertTrue(torch.equal(actual['model'][name], expected['model'][name]))
        for param in actual['optimizer']['state']:
            for key in actual['optimizer']['state'][param]:
                self.assertTrue(torch.equal(actual['optimizer']['state'][param][key],
                                            expected['optimizer']['state'][param][key]))

    def test_final_only_at_global_target_and_never_overwritten(self):
        self.run_session()
        result = self.run_session(session_stop_step=6, resume_from_checkpoint='local')
        self.assertEqual(result['status'], 'completed')
        self.assertTrue((self.root / 'run' / 'final' / 'tiny-export.json').is_file())
        self.assertTrue((self.root / 'run' / 'offline_column_index.npy').is_file())
        final_bytes = (self.root / 'run' / 'final' / 'tiny-export.json').read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_session(session_stop_step=6, resume_from_checkpoint='local')
        self.assertEqual((self.root / 'run' / 'final' / 'tiny-export.json').read_bytes(), final_bytes)

    def test_save_eval_and_session_boundary_share_one_immutable_checkpoint(self):
        self.periodic(session_stop_step=2)
        path = self.root / 'run' / 'checkpoint-2'
        before = (path / MANIFEST).read_bytes()
        commits = self.hub.commits
        self.periodic(session_stop_step=2, resume_from_checkpoint='local')
        self.assertEqual((path / MANIFEST).read_bytes(), before)
        self.assertEqual(self.hub.commits, commits)
        self.assertEqual(len([r for r in self.metrics() if r.get('event') == 'retrieval_validation']), 1)

    def test_absolute_validation_schedule_and_resume_watermark(self):
        self.periodic(session_stop_step=2)
        self.periodic(session_stop_step=6, resume_from_checkpoint='local')
        rows = [r for r in self.metrics() if r.get('event') == 'retrieval_validation']
        self.assertEqual([r['step'] for r in rows], [2, 4, 6])
        for row in rows:
            self.assertEqual(row['evaluation_kind'], 'internal_validation')
            self.assertEqual(row['questions'], 2)
            self.assertEqual(row['k'], 10)
            self.assertIn('recall_at_10', row)
            self.assertIn('hit_at_10', row)
        progress = self.state('run', 6)['extra']['validation_progress']
        self.assertEqual(progress['last_evaluation_step'], 6)
        self.assertEqual(progress['last_validation']['status'], 'ok')

    def test_final_evaluates_off_interval_without_evaluating_interim_off_interval(self):
        self.periodic(global_max_steps=5, session_stop_step=3)
        self.periodic(global_max_steps=5, session_stop_step=5, resume_from_checkpoint='local')
        rows = [r for r in self.metrics() if r.get('event') == 'retrieval_validation']
        self.assertEqual([r['step'] for r in rows], [2, 4, 5])

    def test_validation_rng_draws_do_not_change_training(self):
        with patch.dict(os.environ, CHECKPOINT_EXPERIMENT_ID='noeval-fixture'):
            self.run_session('no-eval', session_stop_step=6)
        expected = self.state('no-eval', 6)
        evaluate = embedding.evaluate_retrieval
        def random_evaluation(*args, **kwargs):
            random.random()
            np.random.rand(7)
            torch.rand(11)
            self.assertFalse(torch.is_grad_enabled())
            return evaluate(*args, **kwargs)
        with patch.object(embedding, 'evaluate_retrieval', side_effect=random_evaluation):
            self.periodic(output='with-eval', session_stop_step=6)
        actual = self.state('with-eval', 6)
        for name in expected['model']:
            self.assertTrue(torch.equal(expected['model'][name], actual['model'][name]))
        self.assertTrue(torch.equal(expected['rng_torch'], actual['rng_torch']))
        self.assertEqual(expected['rng_python'], actual['rng_python'])
        self.assertTrue(np.array_equal(expected['rng_numpy'][1], actual['rng_numpy'][1]))
        self.assertEqual(TinyEmbedding.instances[-1].queries, TinyEmbedding.instances[-2].queries)
        self.assertTrue(all('validation-' not in q for q in TinyEmbedding.instances[-1].queries))

    def test_isolated_validation_preserves_optimizer_mixed_modes_and_rng_on_error(self):
        model = TinyEmbedding()
        model.train()
        model.layer.eval()
        optimizer = torch.optim.AdamW(model.parameters())
        for param in model.parameters():
            param.grad = torch.ones_like(param)
        optimizer.step()
        optimizer.zero_grad()
        before = copy.deepcopy(optimizer.state_dict())
        modes = [m.training for m in model.modules()]
        python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        with self.assertRaisesRegex(RuntimeError, 'validation failed'):
            with isolated_validation(model):
                self.assertTrue(all(not m.training for m in model.modules()))
                self.assertFalse(torch.is_grad_enabled())
                random.random(); np.random.rand(); torch.rand(1)
                raise RuntimeError('validation failed')
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertEqual(python_rng, random.getstate())
        self.assertTrue(np.array_equal(numpy_rng[1], np.random.get_state()[1]))
        self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))
        after = optimizer.state_dict()
        for p in before['state']:
            for key in before['state'][p]:
                self.assertTrue(torch.equal(before['state'][p][key], after['state'][p][key]))

    def test_overlap_rejected_before_model_load(self):
        self.evaluation.write_text(json.dumps(dict(db_id='db', question=' TRAIN-0 ',
            SQL='SELECT c1 FROM t', positive_columns=['t.c1'])) + '\n')
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.periodic()
        self.assertEqual(TinyEmbedding.instances, [])

    def test_validation_failure_saves_completed_update_without_final(self):
        with patch.object(embedding, 'evaluate_retrieval', side_effect=RuntimeError('validation OOM fixture')):
            with self.assertRaisesRegex(RuntimeError, 'validation OOM fixture'):
                self.periodic()
        path = self.root / 'run' / 'checkpoint-2'
        self.assertEqual(verify(path)['step'], 2)
        self.assertEqual(self.state('run', 2)['step'], 2)
        self.assertFalse((self.root / 'run' / 'final').exists())
        self.assertEqual(self.managers[-1].remote_candidates()[0][0], 2)
        # Explicitly resuming the completed update never restarts at zero.
        result = self.periodic(session_stop_step=3, resume_from_checkpoint='local')
        self.assertEqual(result['step'], 3)
        self.assertEqual(len(TinyEmbedding.instances[-1].queries), 2)

    def test_failed_final_validation_can_resume_export_without_retraining_or_overwriting_checkpoint(self):
        with patch.object(embedding, 'evaluate_retrieval', side_effect=RuntimeError('transient final validation')):
            with self.assertRaises(RuntimeError):
                self.periodic(global_max_steps=2, session_stop_step=2)
        path = self.root / 'run' / 'checkpoint-2'
        manifest = (path / MANIFEST).read_bytes()
        result = self.periodic(global_max_steps=2, session_stop_step=2, resume_from_checkpoint='local')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(TinyEmbedding.instances[-1].queries, [])
        self.assertEqual((path / MANIFEST).read_bytes(), manifest)
        self.assertTrue((self.root / 'run' / 'final').is_dir())

    def test_invalid_validation_metrics_are_checkpointed_as_failure(self):
        with patch.object(embedding, 'evaluate_retrieval', return_value=(
                {'recall_at_k': float('nan'), 'hit_at_k': 1, 'k': 10, 'questions': 2}, None)):
            with self.assertRaisesRegex(ValueError, 'Invalid Recall'):
                self.periodic()
        self.assertEqual(verify(self.root / 'run' / 'checkpoint-2')['step'], 2)

    def test_cuda_rng_preserved_without_using_gpu(self):
        model = TinyEmbedding()
        states = [torch.tensor([1], dtype=torch.uint8)]
        with patch.object(torch.cuda, 'is_available', return_value=True), \
             patch.object(torch.cuda, 'get_rng_state_all', return_value=states), \
             patch.object(torch.cuda, 'set_rng_state_all') as restore:
            with isolated_validation(model):
                torch.rand(1)
        restore.assert_called_once_with(states)

    def test_remote_failure_under_warn_stops_before_next_update_or_session_success(self):
        self.run_session(session_stop_step=1, save_steps=1)
        with patch.dict(os.environ, CHECKPOINT_SYNC_FAILURE_POLICY='warn'), \
             patch.object(self.hub, 'create_commit', side_effect=ConnectionError('offline')):
            with self.assertWarns(RuntimeWarning), self.assertRaisesRegex(SyncError, 'refusing continuation'):
                self.run_session(session_stop_step=4, save_steps=1, resume_from_checkpoint='local')
        self.assertEqual(len(TinyEmbedding.instances[-1].queries), 2)
        self.assertTrue((self.root / 'run' / 'checkpoint-2' / 'COMPLETE').exists())
        self.assertFalse((self.root / 'run' / 'checkpoint-3').exists())

    def test_session_boundary_forces_remote_backup_despite_sync_interval(self):
        with patch.dict(os.environ, CHECKPOINT_SYNC_EVERY_N_SAVES='100'):
            self.run_session(session_stop_step=1)
        self.assertEqual(self.managers[-1].remote_candidates()[0][0], 1)

    def test_retention_keeps_latest_verified_checkpoint_and_cache_untouched(self):
        with patch.dict(os.environ, CHECKPOINT_LOCAL_KEEP='1'):
            self.periodic(session_stop_step=3)
        self.assertFalse((self.root / 'run' / 'checkpoint-2').exists())
        self.assertTrue((self.root / 'run' / 'checkpoint-3').exists())
        self.assertTrue(list((self.root / 'cache').rglob('training_state.pt')))
        self.assertEqual(self.managers[-1].remote_candidates()[0][0], 3)

    def test_global_target_and_validation_contract_changes_reject_resume(self):
        self.periodic()
        for overrides in (dict(global_max_steps=7), dict(eval_steps=3),
                          dict(evaluation_kind='external_held_out'), dict(sampling_strategy='sequential'),
                          dict(gradient_accumulation_steps=4), dict(grounding_label_mode='strict')):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.periodic(session_stop_step=6,
                    resume_from_checkpoint=str(self.root / 'run' / 'checkpoint-3'), **overrides)
        self.evaluation.write_text(self.evaluation.read_text() + json.dumps(dict(db_id='db',
            question='extra heldout', SQL='SELECT c1 FROM t', positive_columns=['t.c1'])) + '\n')
        with self.assertRaises(ValueError):
            self.periodic(session_stop_step=6, resume_from_checkpoint=str(self.root / 'run' / 'checkpoint-3'))

    def test_boundary_before_resume_and_missing_resume_rejected(self):
        self.run_session()
        models_before = len(TinyEmbedding.instances)
        with self.assertRaisesRegex(ValueError, 'explicitly resume'):
            self.run_session(session_stop_step=4)
        self.assertEqual(len(TinyEmbedding.instances), models_before)
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            self.run_session(session_stop_step=2, resume_from_checkpoint='local')
        with self.assertRaises(FileNotFoundError):
            self.run_session('empty', resume_from_checkpoint='local')
        self.assertFalse((self.root / 'empty' / 'checkpoint-1').exists())

    def test_invalid_session_or_evaluation_configuration_rejected(self):
        for overrides in (dict(global_max_steps=None), dict(session_stop_step=7),
                          dict(global_max_steps=0), dict(eval_steps=2),
                          dict(eval_path=str(self.evaluation), eval_steps=2), dict(eval_steps=-1)):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.run_session(**overrides)
        with patch.dict(os.environ, CHECKPOINT_SYNC_ENABLED='0'):
            with self.assertRaisesRegex(ValueError, 'requires enabled'):
                self.run_session()

    def test_legacy_max_steps_exports_without_new_contract_fields(self):
        with patch.dict(os.environ, CHECKPOINT_SYNC_ENABLED='0'):
            result = self.run_session(global_max_steps=None, session_stop_step=None,
                                      sampling_strategy='sequential', max_steps=2)
        self.assertEqual(result['step'], 2)
        self.assertTrue((self.root / 'run' / 'final').exists())
        contract = self.state('run', 2)['extra']['contract']
        self.assertNotIn('global_max_steps', contract)
        self.assertNotIn('validation', contract)
        config = json.loads((self.root / 'run' / 'checkpoint-2' / MANIFEST).read_text())['configuration']
        for key in ('global_max_steps', 'eval_steps', 'evaluation_kind', 'session_stop_step'):
            self.assertNotIn(key, config)
        # Legacy max_steps remains an extendable absolute optimizer-step limit.
        with patch.dict(os.environ, CHECKPOINT_SYNC_ENABLED='0'):
            resumed = self.run_session('legacy-resume', global_max_steps=None, session_stop_step=None,
                sampling_strategy='sequential', max_steps=4,
                resume_from_checkpoint=str(self.root / 'run' / 'checkpoint-2'))
        self.assertEqual(resumed['step'], 4)
        self.assertEqual(contract, self.state('legacy-resume', 4)['extra']['contract'])

    def test_preexisting_index_protected(self):
        output = self.root / 'run'
        output.mkdir()
        index = output / 'offline_column_index.npy'
        index.write_bytes(b'preserve existing export')
        with self.assertRaises(FileExistsError):
            self.run_session()
        self.assertEqual(index.read_bytes(), b'preserve existing export')

    def test_preexisting_final_summary_protected(self):
        output = self.root / 'run'
        output.mkdir()
        summary = output / 'training_summary.json'
        summary.write_text('{"existing":"final summary"}')
        with self.assertRaises(FileExistsError):
            self.run_session()
        self.assertEqual(summary.read_text(), '{"existing":"final summary"}')

    def test_final_export_disk_failure_preserves_target_checkpoint_and_can_resume(self):
        from src.training.environment import check_checkpoint_disk
        def no_export_space(output, cache, payload, final_export):
            if payload == 0:
                raise OSError('Insufficient final export disk fixture')
            return check_checkpoint_disk(output, cache, payload, final_export)
        with patch('src.training.environment.check_checkpoint_disk', side_effect=no_export_space):
            with self.assertRaisesRegex(OSError, 'final export disk'):
                self.run_session(global_max_steps=2, session_stop_step=2)
        self.assertEqual(verify(self.root / 'run' / 'checkpoint-2')['step'], 2)
        self.assertFalse((self.root / 'run' / '.final-export.claim').exists())
        self.assertFalse((self.root / 'run' / 'final').exists())
        result = self.run_session(global_max_steps=2, session_stop_step=2, resume_from_checkpoint='local')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(TinyEmbedding.instances[-1].queries, [])

    @unittest.skipUnless(importlib.util.find_spec('sqlglot'), 'Pinned SQLGlot required')
    def test_strict_grounding_labels_work_across_sessions(self):
        with sqlite3.connect(self.root / 'db.sqlite') as database:
            database.execute('CREATE TABLE t(c0 INTEGER, c1 INTEGER, c2 INTEGER, c3 INTEGER)')
        self.periodic(grounding_label_mode='strict', db_dir=str(self.root), session_stop_step=2)
        result = self.periodic(grounding_label_mode='strict', db_dir=str(self.root),
                               session_stop_step=6, resume_from_checkpoint='local')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(self.state('run', 6)['extra']['contract']['grounding_label_mode'], 'strict')

    def test_recall_at_10_and_hit_at_10_use_correct_denominators(self):
        cards = [dict(table_name='t', column_name=f'c{i}') for i in range(12)]
        vectors = np.array([[12 - i, 0] for i in range(12)], dtype=float)
        model = SimpleNamespace(encode=lambda texts, **kwargs:
            vectors if len(texts) > 1 else np.array([[1., 0.]]))
        examples = [dict(query='q1', positive=[0, 11], negative=list(range(1, 11))),
                    dict(query='q2', positive=[11], negative=list(range(11)))]
        metrics, _ = embedding.evaluate_retrieval(model, cards, examples, top_k=10)
        self.assertAlmostEqual(metrics['recall_at_k'], 1 / 3)
        self.assertEqual(metrics['hit_at_k'], 0.5)
        self.assertEqual((metrics['k'], metrics['questions']), (10, 2))


class TestValidationConfiguration(unittest.TestCase):
    def test_disabled_legacy_contract_unchanged(self):
        self.assertIsNone(validation_contract(0, None, None))


if __name__ == '__main__':
    unittest.main()
