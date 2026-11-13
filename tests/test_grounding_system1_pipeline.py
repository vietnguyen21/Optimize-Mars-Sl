"""CPU contracts for real Grounding/System-1 training and calibration."""
import importlib.util
import math
import tempfile
import unittest
from unittest.mock import patch

from src.training import train_embedding as embedding
from src.training import train_decision_model as decision
from src.validation.laya_scorer import LayaScorer


class PipelineContracts(unittest.TestCase):
    def test_real_embedding_never_calls_smoke(self):
        with patch.object(embedding, 'TORCH_AVAILABLE', False), patch.object(embedding, 'run_dry_run') as smoke:
            with self.assertRaises(ImportError):
                embedding.run_embedding_training(dry_run=False)
            smoke.assert_not_called()

    def test_real_decision_never_calls_smoke(self):
        with patch.object(decision, 'TORCH_AVAILABLE', False), patch.object(decision, 'run_dry_run') as smoke:
            with self.assertRaises(ImportError):
                decision.run_decision_model_training(data_path='labeled.jsonl', output_dir='unused',dry_run=False)
            smoke.assert_not_called()

    def test_real_scorer_requires_checkpoint(self):
        with self.assertRaises(ValueError):
            LayaScorer(dry_run=False)

    def test_question_groups_are_disjoint(self):
        records = [dict(db_id='db',question=f'q{i}',sql=f'SELECT {j}',label=j)
                   for i in range(20) for j in [0,1]]
        train,cal,test=decision.split_candidate_groups(records)
        keys=[{r['question'] for r in rows} for rows in (train,cal,test)]
        self.assertFalse(keys[0]&keys[1] or keys[0]&keys[2] or keys[1]&keys[2])
        self.assertEqual(sum(map(len,(train,cal,test))),len(records))
        self.assertEqual((train,cal,test),decision.split_candidate_groups(records))

    def test_temperature_fits_nll_without_promised_ece(self):
        logits=[10.,10.,-10.,-10.];labels=[1,0,0,1]
        temperature,ece=decision.fit_temperature(logits,labels)
        self.assertGreater(temperature,1.)
        self.assertGreater(ece,0.)
        self.assertTrue(math.isfinite(ece))

    def test_explicit_embedding_labels_keep_database_negatives(self):
        cards=[dict(db_id=db,table_name='t',column_name=c) for db in ['a','b'] for c in ['id','value']]
        examples=embedding.build_embedding_examples(cards,[dict(db_id='a',question='find id',positive_columns=['t.id'])])
        self.assertEqual(examples[0]['positive'],[0])
        self.assertEqual(examples[0]['negative'],[1])

    @unittest.skipUnless(importlib.util.find_spec('sqlglot'), 'sqlglot unavailable')
    def test_count_star_does_not_label_all_columns(self):
        cards=[dict(db_id='db',table_name='t',column_name=c) for c in ['id','amount','flag']]
        examples=embedding.build_embedding_examples(cards,[dict(db_id='db',question='Count flagged rows',
            SQL='SELECT COUNT(*) FROM t WHERE flag = 1')])
        self.assertEqual(examples[0]['positive'],[2])
        self.assertEqual(examples[0]['negative'],[0,1])

    def test_embedding_requires_recoverable_database_membership(self):
        with self.assertRaises(ValueError):
            embedding.build_embedding_examples([dict(table_name='t',column_name='id')],
                [dict(db_id='a',question='find id',positive_columns=['t.id'])])

    def test_candidate_labels_come_from_execution(self):
        records=[dict(db_id='financial_mini',question='Count clients',schema='',
                      SQL='SELECT COUNT(*) FROM clients',candidates=['SELECT COUNT(*) FROM clients',
                      'SELECT COUNT(*)+1 FROM clients','SELECT missing FROM clients'])]
        rows=decision.build_candidate_dataset(records,'data/mini',include_gold=False)
        self.assertEqual([r['label'] for r in rows],[1,0,0])
        self.assertTrue(all(r['label_source']=='execution_match' for r in rows))

    def test_candidate_label_respects_order(self):
        records=[dict(db_id='financial_mini',question='Client identifiers ascending',
            SQL='SELECT client_id FROM clients ORDER BY client_id ASC',
            candidates=['SELECT client_id FROM clients ORDER BY client_id DESC'])]
        rows=decision.build_candidate_dataset(records,'data/mini')
        self.assertEqual(rows[0]['label'],0)
        self.assertIn('CREATE TABLE',rows[0]['schema'])

    def test_candidate_builder_rejects_fixture_generator(self):
        import types
        with self.assertRaises(ValueError):
            decision.build_candidate_dataset([dict(db_id='financial_mini',question='count',
                SQL='SELECT COUNT(*) FROM clients')],'data/mini',generator=types.SimpleNamespace(dry_run=True))

    def test_candidate_builder_rejects_missing_rollouts(self):
        with self.assertRaises(ValueError):
            decision.build_candidate_dataset([dict(db_id='financial_mini',question='count',
                SQL='SELECT COUNT(*) FROM clients')],'data/mini')

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'torch unavailable')
    def test_hn_supcon_keeps_autograd(self):
        import torch
        q=torch.tensor([1.,0.],requires_grad=True)
        loss=embedding.compute_hn_supcon_loss(q,torch.tensor([.9,.1]),[torch.tensor([.8,.2])])
        loss.backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertGreater(float(q.grad.abs().sum()),0.)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'torch unavailable')
    def test_rlcd_keeps_autograd(self):
        import torch
        logits=torch.tensor([.5,-.1],requires_grad=True)
        loss=decision.compute_rlcd_loss(logits.sigmoid(),torch.tensor([1.,0.]))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())



@unittest.skipUnless(importlib.util.find_spec('torch'), 'torch unavailable')
class TinyAutogradLoops(unittest.TestCase):
    """Exercise real optimizer/resume paths with local tiny neural backbones.

    These are CPU integration tests, not claims about HF or Kaggle execution.
    """
    def test_embedding_optimizer_and_resume(self):
        import json
        import sys
        import types
        from pathlib import Path
        import torch
        import torch.nn.functional as F

        class TinySentence(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.embedding=torch.nn.Embedding(128,8)
                self.projection=torch.nn.Linear(8,8)
                self.max_seq_length=32
            def __getitem__(self,index):
                return types.SimpleNamespace(auto_model=types.SimpleNamespace(gradient_checkpointing_enable=lambda:None))
            def tokenize(self,texts):
                ids=[[ord(c)%128 for c in text[:16]] for text in texts]
                width=max(map(len,ids))
                return {'input_ids':torch.tensor([x+[0]*(width-len(x)) for x in ids])}
            def forward(self,features):
                return {'sentence_embedding':self.projection(self.embedding(features['input_ids']).mean(1))}
            def encode(self,texts,convert_to_tensor=False,**kwargs):
                with torch.no_grad():
                    x=F.normalize(self(self.tokenize(texts))['sentence_embedding'],dim=-1)
                return x if convert_to_tensor else x.numpy()
            def save(self,path):
                Path(path).mkdir(parents=True)
                torch.save(self.state_dict(),Path(path)/'tiny_weights.pt')

        module=types.ModuleType('sentence_transformers');module.SentenceTransformer=TinySentence
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules,{'sentence_transformers':module}), \
             patch('src.training.model_identity.model_identity', return_value=({'kind':'local','files':{'mock.safetensors':'0'*64}}, {})):
            root=Path(directory); cards=root/'cards.json'; data=root/'train.json'
            cards.write_text(json.dumps([dict(db_id='db',table_name='t',column_name=c) for c in ['id','value','other']]))
            data.write_text(json.dumps([dict(db_id='db',question='Find identifier',positive_columns=['t.id'])]))
            common=dict(cards_path=str(cards),bird_path=str(data),device='cpu',gradient_accumulation_steps=1,
                        max_length=32,save_steps=1,seed=9,hard_negatives=1,learning_rate=.01)
            first=embedding.run_embedding_training(output_dir=str(root/'first'),max_steps=1,**common)
            resumed=embedding.run_embedding_training(output_dir=str(root/'resumed'),max_steps=2,
                         resume_from_checkpoint=str(root/'first'/'checkpoint-1'),**common)
            full=embedding.run_embedding_training(output_dir=str(root/'full'),max_steps=2,**common)
            self.assertTrue(first['trained'] and resumed['trained'])
            self.assertTrue((root/'resumed'/'checkpoint-2'/'COMPLETE').exists())
            a=torch.load(root/'resumed'/'final'/'tiny_weights.pt',weights_only=True)
            b=torch.load(root/'full'/'final'/'tiny_weights.pt',weights_only=True)
            for key in a:
                self.assertTrue(torch.equal(a[key],b[key]),key)
            first_weights=torch.load(root/'first'/'final'/'tiny_weights.pt',weights_only=True)
            self.assertFalse(torch.equal(a['projection.weight'],first_weights['projection.weight']))

            from src.training.runtime import optimizer_update_succeeded
            attempts = []
            def skip_once(optimizer, scaler):
                attempts.append(True)
                return False if len(attempts) == 1 else optimizer_update_succeeded(optimizer, scaler)
            with patch('src.training.runtime.optimizer_update_succeeded', side_effect=skip_once):
                skipped = embedding.run_embedding_training(output_dir=str(root/'skipped'),max_steps=1,**common)
            self.assertEqual(skipped['step'],1)
            self.assertEqual(len(attempts),2)
            import os
            with patch.dict(os.environ, AMP_MAX_CONSECUTIVE_SKIPS='2'), patch('src.training.runtime.optimizer_update_succeeded',return_value=False):
                with self.assertRaisesRegex(FloatingPointError,'Repeated AMP overflow'):
                    embedding.run_embedding_training(output_dir=str(root/'always-overflow'),max_steps=1,**common)
            skipped_weights=torch.load(root/'skipped'/'final'/'tiny_weights.pt',weights_only=True)
            for key in first_weights:
                torch.testing.assert_close(first_weights[key],skipped_weights[key],rtol=0,atol=0)


    def test_decision_optimizer_and_resume(self):
        import json
        import sys
        import types
        from pathlib import Path
        import torch
        import torch._dynamo
        import torch._inductor.config

        class TinyTokenizer:
            @classmethod
            def from_pretrained(cls,*args,**kwargs):
                return cls()
            def __call__(self,text,**kwargs):
                texts=[text] if isinstance(text,str) else text
                ids=[[ord(c)%128 for c in t[:128]] for t in texts]
                width=max(map(len,ids))
                return {'input_ids':torch.tensor([x+[0]*(width-len(x)) for x in ids])}
            def save_pretrained(self,path):
                Path(path).mkdir(parents=True,exist_ok=True)
                (Path(path)/'tokenizer_config.json').write_text('{}')

        class TinyClassifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding=torch.nn.Embedding(128,8)
                self.classifier=torch.nn.Linear(8,1)
            @classmethod
            def from_pretrained(cls,source,**kwargs):
                model=cls()
                weights=Path(source)/'tiny_weights.pt'
                if weights.exists():
                    model.load_state_dict(torch.load(weights,weights_only=True))
                return model
            def forward(self,input_ids):
                return types.SimpleNamespace(logits=self.classifier(self.embedding(input_ids).mean(1)))
            def gradient_checkpointing_enable(self):
                pass
            def save_pretrained(self,path):
                Path(path).mkdir(parents=True,exist_ok=True)
                torch.save(self.state_dict(),Path(path)/'tiny_weights.pt')

        module=types.ModuleType('transformers')
        module.AutoTokenizer=TinyTokenizer;module.AutoModelForSequenceClassification=TinyClassifier
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules,{'transformers':module}), \
             patch('src.training.model_identity.model_identity', return_value=({'kind':'local','files':{'mock.safetensors':'0'*64}}, {})):
            root=Path(directory);data=root/'candidates.json'
            rows=[dict(db_id='db',question=f'Question {i}',schema='CREATE TABLE t(id INT);',
                        sql=f'SELECT {label}',label=label) for i in range(12) for label in [0,1]]
            data.write_text(json.dumps(rows))
            common=dict(data_path=str(data),device='cpu',dry_run=False,model_name_or_path='tiny',
                        gradient_accumulation_steps=1,save_steps=1,seed=9,lr_backbone=.001,lr_head=.01)
            first=decision.run_decision_model_training(output_dir=str(root/'first'),max_steps=1,**common)
            resumed=decision.run_decision_model_training(output_dir=str(root/'resumed'),max_steps=2,
                        resume_from_checkpoint=str(root/'first'/'checkpoint-1'),**common)
            full=decision.run_decision_model_training(output_dir=str(root/'full'),max_steps=2,**common)
            self.assertTrue(first['trained'] and resumed['trained'])
            self.assertTrue((root/'resumed'/'final'/'calibration_meta.json').exists())
            a=torch.load(root/'resumed'/'final'/'tiny_weights.pt',weights_only=True)
            b=torch.load(root/'full'/'final'/'tiny_weights.pt',weights_only=True)
            for key in a:
                self.assertTrue(torch.equal(a[key],b[key]),key)
            first_weights=torch.load(root/'first'/'final'/'tiny_weights.pt',weights_only=True)
            self.assertFalse(torch.equal(a['classifier.weight'],first_weights['classifier.weight']))

            from src.training.runtime import optimizer_update_succeeded
            attempts = []
            def skip_once(optimizer, scaler):
                attempts.append(True)
                return False if len(attempts) == 1 else optimizer_update_succeeded(optimizer, scaler)
            with patch('src.training.runtime.optimizer_update_succeeded', side_effect=skip_once):
                skipped = decision.run_decision_model_training(output_dir=str(root/'skipped'),max_steps=1,**common)
            self.assertEqual(skipped['step'],1)
            self.assertEqual(len(attempts),2)
            import os
            with patch.dict(os.environ, AMP_MAX_CONSECUTIVE_SKIPS='2'), patch('src.training.runtime.optimizer_update_succeeded',return_value=False):
                with self.assertRaisesRegex(FloatingPointError,'Repeated AMP overflow'):
                    decision.run_decision_model_training(output_dir=str(root/'always-overflow'),max_steps=1,**common)
            skipped_weights=torch.load(root/'skipped'/'final'/'tiny_weights.pt',weights_only=True)
            for key in first_weights:
                torch.testing.assert_close(first_weights[key],skipped_weights[key],rtol=0,atol=0)

            self.assertGreater(resumed['temperature'],0)
            scorer=LayaScorer(model_path=resumed['model_path'],device='cpu',dry_run=False)
            with patch.object(scorer,'_compute_calibrated_logit',side_effect=AssertionError('heuristic used')):
                scores=scorer.score_candidates_parallel('Question 1','CREATE TABLE t(id INT);',['SELECT 0','SELECT 1'])
            self.assertEqual(len(scores),2)
            self.assertTrue(all(0 <= score <= 1 for score in scores))


if __name__=='__main__':
    unittest.main()
