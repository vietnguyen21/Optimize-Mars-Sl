"""Path/mode invariants plus real checkpoint roundtrips when PyTorch is installed."""
import importlib.util
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
from src.training.kaggle import KagglePaths
from src.training.runtime import (RunConfig, ExperimentLogger, load_records, latest_checkpoint,
                                  save_checkpoint, load_checkpoint, seed_everything, resolve_db)


def real_torch_available():
    try:
        import torch
        return hasattr(torch, 'optim') and hasattr(torch, 'manual_seed')
    except ImportError:
        return False


class TestRuntime(unittest.TestCase):
    def test_mode_is_explicit_and_outputs_separated(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.dict(os.environ, {'KAGGLE_OUTPUT_DIR':d,'PIPELINE_MODE':'train'}):
                train = KagglePaths.from_env()
                self.assertFalse(train.smoke)
                a = train.output('sft')
            with patch.dict(os.environ, {'KAGGLE_OUTPUT_DIR':d,'PIPELINE_MODE':'smoke'}):
                self.assertNotEqual(a, KagglePaths.from_env().output('sft'))
            with patch.dict(os.environ, {'PIPELINE_MODE':'maybe'}):
                with self.assertRaises(ValueError): KagglePaths.from_env()

    def test_load_json_and_jsonl_and_reject_empty(self):
        with tempfile.TemporaryDirectory() as d:
            for name, content in [('r.json','[{"question":"q"}]'), ('r.jsonl','{"question":"q"}\n')]:
                p=Path(d)/name; p.write_text(content)
                self.assertEqual(load_records(p), [{'question':'q'}])
            p.write_text('')
            with self.assertRaises(ValueError): load_records(p)

    def test_log_and_checkpoint_discovery_ignore_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            ExperimentLogger(d, RunConfig(d)).log(step=1, loss=0.5)
            rows=load_records(Path(d)/'metrics.jsonl')
            self.assertEqual(rows[-1]['loss'], .5)
            for step in (1, 2, 10): (Path(d)/f'checkpoint-{step}').mkdir()
            (Path(d)/'checkpoint-2'/'COMPLETE').touch()
            self.assertEqual(latest_checkpoint(d), str(Path(d)/'checkpoint-2'))

    def test_db_resolution_nested_and_no_escape(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'db').mkdir(); (Path(d)/'db'/'db.sqlite').touch()
            self.assertEqual(resolve_db(d, 'db'), str(Path(d)/'db'/'db.sqlite'))
            with self.assertRaises(FileNotFoundError): resolve_db(d,'../../missing')

    @unittest.skipUnless(real_torch_available(), 'PyTorch is not installed')
    def test_interrupted_save_does_not_block_retry(self):
        import torch
        class InterruptedModel(torch.nn.Linear):
            def save_pretrained(self, path):
                (Path(path)/'partial.weights').write_text('interrupted')
                raise OSError('simulated interruption')
        with tempfile.TemporaryDirectory() as d:
            bad=InterruptedModel(2,1)
            optimizer=torch.optim.AdamW(bad.parameters())
            with self.assertRaises(OSError): save_checkpoint(d,bad,optimizer,1)
            self.assertIsNone(latest_checkpoint(d))
            self.assertFalse((Path(d)/'checkpoint-1').exists())
            good=torch.nn.Linear(2,1)
            path=save_checkpoint(d,good,torch.optim.AdamW(good.parameters()),1)
            self.assertEqual(latest_checkpoint(d),path)
            self.assertTrue(list(Path(d).glob('.checkpoint-1-*/partial.weights')))

    @unittest.skipUnless(real_torch_available(), 'PyTorch is not installed')
    def test_checkpoint_restores_parameters_optimizer_and_rng(self):
        import torch
        with tempfile.TemporaryDirectory() as d:
            seed_everything(7)
            model=torch.nn.Linear(2,1); opt=torch.optim.AdamW(model.parameters())
            model(torch.ones(1,2)).sum().backward(); opt.step(); opt.zero_grad()
            expected={k:v.detach().clone() for k,v in model.state_dict().items()}
            p=save_checkpoint(d,model,opt,1,extra={'cursor':3})
            expected_random=random.random(); expected_torch=torch.rand(3)
            with torch.no_grad(): model.weight.fill_(99)
            state=load_checkpoint(p,model,opt)
            self.assertEqual(state['extra']['cursor'],3)
            self.assertEqual(random.random(),expected_random)
            self.assertTrue(torch.equal(torch.rand(3),expected_torch))
            for k,v in expected.items(): self.assertTrue(torch.equal(model.state_dict()[k],v))
            with self.assertRaises(FileExistsError): save_checkpoint(d,model,opt,1)
            (Path(d)/'checkpoint-9').mkdir()
            with self.assertRaises(ValueError): load_checkpoint(Path(d)/'checkpoint-9',model,opt)


if __name__=='__main__': unittest.main()
