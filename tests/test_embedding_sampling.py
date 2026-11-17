"""CPU-only sampling and checkpoint contract regressions; no model downloads."""
import random
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from src.training.embedding_sampling import EmbeddingSampler, validate_embedding_extra


class SamplingTests(unittest.TestCase):
    def test_epochs(self):
        sampler = EmbeddingSampler(8149, 42, 'epoch_shuffle_v1')
        epochs = [[sampler.index(e * 8149 + i) for i in range(8149)] for e in range(3)]
        for order in epochs:
            self.assertEqual(sorted(order), list(range(8149)))
            self.assertEqual(len(set(order)), 8149)
        self.assertNotEqual(epochs[0], epochs[1])
        self.assertNotEqual(epochs[1], epochs[2])
        other = EmbeddingSampler(8149, 43, 'epoch_shuffle_v1')
        self.assertNotEqual(epochs[0], [other.index(i) for i in range(8149)])

    def test_resume_and_boundaries(self):
        for accumulation in (1, 2, 4, 8):
            sampler = EmbeddingSampler(8149, 42, 'epoch_shuffle_v1')
            order = [sampler.index(i) for i in range(8149 * 3)]
            for step in (0, 1, 37, 8149 // accumulation, 8149 // accumulation + 1, 2000):
                resumed = EmbeddingSampler(8149, 42, 'epoch_shuffle_v1')
                start = step * accumulation
                self.assertEqual([resumed.index(start + i) for i in range(accumulation * 3)],
                                 order[start:start + accumulation * 3])
            start = (8149 // accumulation) * accumulation
            self.assertEqual([sampler.index(start + i) for i in range(accumulation)],
                             order[start:start + accumulation])

    def test_rng_independence_and_reproducibility(self):
        before = random.getstate()
        a = EmbeddingSampler(100, 42, 'epoch_shuffle_v1')
        b = EmbeddingSampler(100, 42, 'epoch_shuffle_v1')
        self.assertEqual([a.index(i) for i in range(400)], [b.index(i) for i in range(400)])
        self.assertEqual(before, random.getstate())
        # A fixed v1 vector guards the algorithm against accidental changes.
        self.assertEqual([EmbeddingSampler(5, 42, 'epoch_shuffle_v1').index(i) for i in range(5)],
                         [3, 2, 0, 4, 1])

    def test_legacy_and_strategy_contracts(self):
        base = dict(seed=42, gradient_accumulation_steps=8)
        sequential = dict(base, sampling_strategy='sequential', sampling_version=1)
        shuffled = dict(base, sampling_strategy='epoch_shuffle_v1', sampling_version=1)
        validate_embedding_extra({'contract': base}, {'contract': sequential})
        for saved, expected in ((base, shuffled), (sequential, shuffled), (shuffled, sequential),
                                (dict(shuffled, sampling_version=2), shuffled)):
            with self.assertRaises(ValueError):
                validate_embedding_extra({'contract': saved}, {'contract': expected})
        validate_embedding_extra({'contract': shuffled}, {'contract': shuffled})
        sampler = EmbeddingSampler(8149)
        for i in (0, 8148, 8149, 50000):
            self.assertEqual(sampler.index(i), i % 8149)

    def test_checkpoint_validation_before_state_or_rng_restore(self):
        from src.training.runtime import load_checkpoint
        saved = {'step': 13, 'extra': {'contract': {'seed': 42}}}
        expected = dict(seed=42, sampling_strategy='epoch_shuffle_v1', sampling_version=1)
        model = SimpleNamespace()
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'COMPLETE').touch()
            with patch.dict('sys.modules', {'torch': SimpleNamespace(load=lambda *a, **k: saved),
                                           'numpy': SimpleNamespace()}), \
                 patch('src.training.checkpoint_sync.verify', return_value=None):
                rng = random.getstate()
                with self.assertRaisesRegex(ValueError, 'sampling'):
                    load_checkpoint(directory, model, expected_extra={'contract': expected},
                                    extra_validator=validate_embedding_extra)
                self.assertEqual(random.getstate(), rng)
                self.assertEqual(saved['step'], 13)

    def test_sync_manifest_compatibility(self):
        from src.training.checkpoint_sync import persistent_training, ACTIVE_SYNC, fingerprint
        managers = []

        @persistent_training('grounding')
        def runner(output_dir, seed=42, sampling_strategy='sequential', resume_from_checkpoint=None):
            managers.append(ACTIVE_SYNC.get())

        with tempfile.TemporaryDirectory() as directory, \
             patch('src.training.checkpoint_sync.CheckpointSync.resolve', return_value=None), \
             patch.dict('os.environ', {'CHECKPOINT_SYNC_ENABLED': '0'}):
            runner(directory)
            runner(directory, sampling_strategy='epoch_shuffle_v1')
        sequential, shuffled = managers
        legacy = {'compatibility': fingerprint({'seed': 42}), 'compatibility_version': 1}
        self.assertTrue(sequential.compatible(legacy))
        self.assertTrue(sequential.compatible(dict(legacy, compatibility_version=2)))
        self.assertFalse(shuffled.compatible(dict(legacy, compatibility_version=2)))
        self.assertFalse(shuffled.compatible(legacy))
        for source, target in ((sequential, shuffled), (shuffled, sequential)):
            self.assertFalse(target.compatible({'compatibility': source.compatibility,
                                               'compatibility_version': 2}))
        self.assertEqual(shuffled.configuration['sampling_version'], 1)


if __name__ == '__main__':
    unittest.main()
