"""Offline native HF API integration; tiny random models, never benchmark results."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

AVAILABLE = all(importlib.util.find_spec(name) for name in ('torch','transformers','sentence_transformers','tokenizers'))


@unittest.skipUnless(AVAILABLE,'native Hugging Face dependencies unavailable')
class NativeGroundingSystem1(unittest.TestCase):
    def setUp(self):
        import torch
        self.old_threads=torch.get_num_threads()
        torch.set_num_threads(1)
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)

    def tearDown(self):
        import torch
        torch.set_num_threads(self.old_threads)
        self.temp.cleanup()

    def tokenizer(self,path):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.processors import TemplateProcessing
        from transformers import PreTrainedTokenizerFast
        tokens=['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','Table','Column','Constraints','Description',
                'Sample','Values','Query','Evidence','id','amount','other','t','Find','identifier',
                'SELECT','0','1','question','answer','schema','CREATE','TABLE','INT','Question']
        vocab={word:i for i,word in enumerate(tokens)}
        backend=Tokenizer(WordLevel(vocab,unk_token='[UNK]'))
        backend.pre_tokenizer=Whitespace()
        backend.post_processor=TemplateProcessing(single='[CLS] $A [SEP]',special_tokens=[('[CLS]',2),('[SEP]',3)])
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,pad_token='[PAD]',unk_token='[UNK]',
                cls_token='[CLS]',sep_token='[SEP]',mask_token='[MASK]',
                model_input_names=['input_ids','attention_mask'])
        tokenizer.save_pretrained(path)
        return tokenizer

    def test_native_sentence_transformer_train_export_resume(self):
        import torch
        from transformers import BertConfig,BertModel
        from sentence_transformers import SentenceTransformer,models
        from src.training.train_embedding import run_embedding_training
        base=self.root/'bert';base.mkdir()
        tokenizer=self.tokenizer(base)
        model=BertModel(BertConfig(vocab_size=len(tokenizer),hidden_size=32,intermediate_size=48,
                       num_hidden_layers=1,num_attention_heads=4,hidden_dropout_prob=0.,
                       attention_probs_dropout_prob=0.,max_position_embeddings=128))
        model.save_pretrained(base)
        source=self.root/'sentence'
        initial=SentenceTransformer(modules=[models.Transformer(str(base),max_seq_length=64),models.Pooling(32)])
        initial.save(str(source))
        cards=self.root/'cards.json';data=self.root/'questions.json'
        cards.write_text(json.dumps([dict(db_id='db',table_name='t',column_name=c) for c in ['id','amount','other']]))
        data.write_text(json.dumps([dict(db_id='db',question='Find identifier',positive_columns=['t.id'])]))
        heldout=self.root/'heldout.json'
        heldout.write_text(json.dumps([dict(db_id='db',question='Find amount',positive_columns=['t.amount'])]))
        common=dict(cards_path=str(cards),bird_path=str(data),eval_path=str(heldout),model_name_or_path=str(source),device='cpu',
                    max_length=64,gradient_accumulation_steps=1,hard_negatives=1,save_steps=1,learning_rate=.001)
        first=run_embedding_training(output_dir=str(self.root/'first'),max_steps=1,**common)
        resumed=run_embedding_training(output_dir=str(self.root/'resumed'),max_steps=2,
                  resume_from_checkpoint=str(self.root/'first'/'checkpoint-1'),**common)
        full=run_embedding_training(output_dir=str(self.root/'full'),max_steps=2,**common)
        self.assertTrue(first['trained'] and resumed['trained'])
        a=SentenceTransformer(resumed['model_path'],device='cpu')
        b=SentenceTransformer(full['model_path'],device='cpu')
        for key,value in a.state_dict().items():
            self.assertTrue(torch.equal(value,b.state_dict()[key]),key)
        self.assertTrue(any(not torch.equal(value,initial.state_dict()[key]) for key,value in a.state_dict().items()))
        self.assertIsNotNone(resumed['retrieval'])
        self.assertEqual(a.encode(['Find identifier']).shape,(1,32))
        self.assertTrue((self.root/'resumed'/'offline_column_index.npy').exists())

    def test_native_modernbert_train_score_resume(self):
        import torch
        from transformers import ModernBertConfig,ModernBertForSequenceClassification
        from src.training.train_decision_model import run_decision_model_training
        from src.validation.laya_scorer import LayaScorer
        source=self.root/'modernbert';source.mkdir()
        tokenizer=self.tokenizer(source)
        config=ModernBertConfig(vocab_size=len(tokenizer),hidden_size=32,intermediate_size=48,
                    num_hidden_layers=1,num_attention_heads=4,max_position_embeddings=128,
                    pad_token_id=0,bos_token_id=2,eos_token_id=3,cls_token_id=2,sep_token_id=3,
                    num_labels=1,reference_compile=False,attn_implementation='eager')
        ModernBertForSequenceClassification(config).save_pretrained(source)
        data=self.root/'candidates.json'
        data.write_text(json.dumps([dict(db_id='db',question=f'Question {i}',
             schema='CREATE TABLE t(id INT);',sql=f'SELECT {label}',label=label)
             for i in range(12) for label in [0,1]]))
        common=dict(data_path=str(data),model_name_or_path=str(source),device='cpu',dry_run=False,
                    max_length=64,gradient_accumulation_steps=1,save_steps=1,lr_backbone=.001,lr_head=.001)
        first=run_decision_model_training(output_dir=str(self.root/'first'),max_steps=1,**common)
        resumed=run_decision_model_training(output_dir=str(self.root/'resumed'),max_steps=2,
                    resume_from_checkpoint=str(self.root/'first'/'checkpoint-1'),**common)
        full=run_decision_model_training(output_dir=str(self.root/'full'),max_steps=2,**common)
        a=ModernBertForSequenceClassification.from_pretrained(resumed['model_path'])
        b=ModernBertForSequenceClassification.from_pretrained(full['model_path'])
        self.assertFalse(a.config.reference_compile)
        for key,value in a.state_dict().items():
            self.assertTrue(torch.equal(value,b.state_dict()[key]),key)
        scorer=LayaScorer(model_path=resumed['model_path'],device='cpu',dry_run=False)
        self.assertFalse(scorer.model.config.reference_compile)
        scores=scorer.score_candidates_parallel('Question 2','CREATE TABLE t(id INT);',['SELECT 0','SELECT 1'])
        self.assertEqual(len(scores),2)
        self.assertTrue(all(0<=score<=1 for score in scores))
        self.assertTrue(first['trained'])


if __name__=='__main__':
    unittest.main()
