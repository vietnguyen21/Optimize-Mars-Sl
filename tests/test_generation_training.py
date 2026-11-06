"""Regression checks for genuine generation training (tiny CPU models, no downloads)."""
import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from src.training.train_sft_generator import CompletionOnlyDataCollator, run_sft_training
from src.training.train_grpo_generator import (
    run_grpo_training, normalize_group_advantages, grpo_token_objective,
)
try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None


class ByteTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return ([1] if add_special_tokens else []) + [ord(c) % 30 + 2 for c in text]

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        return {"input_ids": torch.tensor([self.encode(text, add_special_tokens)])}

    def decode(self, ids, **kwargs):
        return ",".join(str(int(x)) for x in ids)

    def save_pretrained(self, directory):
        Path(directory, "tokenizer_config.json").write_text("{}")


if torch is not None:
    class TinyPolicy(torch.nn.Module):
        """Frozen tiny causal backbone and switchable trainable/reference heads."""
        def __init__(self):
            super().__init__()
            from types import SimpleNamespace
            self.config = SimpleNamespace(hidden_size=8)
            self.embedding = torch.nn.Embedding(64, 8)
            self.embedding.weight.requires_grad_(False)
            self.adapters = torch.nn.ModuleDict({"default": torch.nn.Linear(8, 64, bias=False),
                                                 "reference": torch.nn.Linear(8, 64, bias=False)})
            self.adapters["reference"].load_state_dict(self.adapters["default"].state_dict())
            self.set_adapter("default")

        def set_adapter(self, name):
            self.active = name
            for key, module in self.adapters.items():
                for p in module.parameters():
                    p.requires_grad_(key == "default" and name == "default")

        def load_adapter(self, source, adapter_name="reference", is_trainable=False):
            self.adapters[adapter_name].load_state_dict(self.adapters["default"].state_dict())

        def forward(self, input_ids, labels=None, **kwargs):
            from types import SimpleNamespace
            hidden = self.embedding(input_ids).cumsum(1)
            hidden = hidden / torch.arange(1, input_ids.shape[1] + 1)[None, :, None]
            logits = self.adapters[self.active](hidden)
            loss = None if labels is None else F.cross_entropy(
                logits[:, :-1].reshape(-1, 64), labels[:, 1:].reshape(-1), ignore_index=-100)
            return SimpleNamespace(logits=logits, loss=loss, hidden_states=[hidden])

        def generate(self, input_ids, max_new_tokens=3, temperature=1.0, **kwargs):
            ids = input_ids
            for _ in range(max_new_tokens):
                logits = self(ids).logits[:, -1] / temperature
                token = torch.multinomial(torch.softmax(logits, -1), 1)
                ids = torch.cat([ids, token], 1)
            return ids


class TestGenerationModes(unittest.TestCase):
    def test_real_mode_never_falls_back_to_smoke(self):
        for module, runner in [("src.training.train_sft_generator", run_sft_training),
                               ("src.training.train_grpo_generator", run_grpo_training)]:
            with patch(module + ".TORCH_AVAILABLE", False), patch(module + ".run_dry_run") as dry:
                with self.assertRaisesRegex(RuntimeError, "Real"):
                    runner(dry_run=False)
                dry.assert_not_called()

    def test_explicit_smoke_is_labeled(self):
        with patch("src.training.train_sft_generator.run_dry_run", return_value={"status": "PASSED"}):
            result = run_sft_training(dry_run=True)
        self.assertEqual(result["mode"], "smoke")
        self.assertFalse(result["trained"])

    def test_equal_rewards_have_exact_zero_advantages(self):
        self.assertEqual(normalize_group_advantages([1.5] * 4), [0.0] * 4)
        self.assertEqual(normalize_group_advantages([1.5]), [0.0])

    def test_missing_completion_boundary_fails_closed(self):
        collator = CompletionOnlyDataCollator(ByteTokenizer())
        with self.assertRaisesRegex(ValueError, "boundary"):
            collator.encode_example({"formatted_prompt": "Question only"})

    def test_structured_boundary_is_exact(self):
        collator = CompletionOnlyDataCollator(ByteTokenizer(), max_length=2048)
        item = collator.encode_example({"question": "Count rows", "schema": "t(x)", "answer": "SELECT COUNT(*) FROM t;"})
        boundary = item["pred_position"] + 1
        self.assertTrue(all(x == -100 for x in item["labels"][:boundary]))
        self.assertEqual(item["labels"][boundary:], item["input_ids"][boundary:])
        self.assertEqual(item["labels"][-1], 0)


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestTinyGenerationTraining(unittest.TestCase):
    def test_padding_masks_attention_not_eos_identity(self):
        collator = CompletionOnlyDataCollator(ByteTokenizer())
        batch = collator([
            dict(input_ids=[1, 2, 0], labels=[-100, 2, 0], pred_position=0, sql_start=1),
            dict(input_ids=[1, 3], labels=[-100, 3], pred_position=0, sql_start=1),
        ])
        self.assertEqual(batch["labels"][0, -1].item(), 0)
        self.assertEqual(batch["labels"][1, -1].item(), -100)

    def test_grpo_ratio_uses_old_policy_not_reference(self):
        new = torch.tensor([-0.4], requires_grad=True)
        loss = grpo_token_objective(new, torch.tensor([-1.5]), torch.tensor([-0.4]),
                                    1.0, beta_kl=0.0)
        loss.backward()
        self.assertEqual(new.grad.item(), 0.0)  # Positive advantage clipped by OLD-policy ratio.
        new = torch.tensor([-0.4], requires_grad=True)
        loss = grpo_token_objective(new, new.detach(), torch.tensor([-8.0]),
                                    1.0, beta_kl=0.0)
        loss.backward()
        self.assertAlmostEqual(new.grad.item(), -1.0, places=5)

    def test_jepa_detaches_targets(self):
        from src.training.jepa_module import LLM_JEPAModule
        module = LLM_JEPAModule(hidden_dim=8, loss_dropout=0.0)
        context = torch.randn(2, 8, requires_grad=True)
        target = torch.randn(2, 8, requires_grad=True)
        module(context, target)["jepa_loss"].backward()
        self.assertIsNotNone(context.grad)
        self.assertIsNone(target.grad)
        self.assertTrue(any(p.grad is not None for p in module.predictor.parameters()))

    def test_sft_updates_and_exact_checkpoint_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "sft.jsonl"
            data.write_text("\n".join(json.dumps({"question": f"Q{i}", "schema": "t(x)",
                            "think_steps": "Count.", "answer": "SELECT COUNT(*) FROM t;"}) for i in range(3)))
            torch.manual_seed(9)
            initial = TinyPolicy()
            before = copy.deepcopy(initial.state_dict())
            common = dict(data_path=str(data), tokenizer=ByteTokenizer(), device="cpu", epochs=2,
                          batch_size=1, max_length=1024, gradient_accumulation_steps=2,
                          lambda_jepa=0.7, loss_dropout=0.5, predictor_dim=4, learning_rate=0.01,
                          save_steps=50, seed=42)
            full_model = copy.deepcopy(initial)
            full = run_sft_training(model=full_model, output_dir=str(root / "full"), max_steps=3, **common)
            split = run_sft_training(model=copy.deepcopy(initial), output_dir=str(root / "split"), max_steps=1, **common)
            resumed_model = copy.deepcopy(initial)
            resumed = run_sft_training(model=resumed_model, output_dir=str(root / "split"), max_steps=3,
                                       resume_from_checkpoint=split["checkpoint_path"], **common)
            self.assertTrue(full["trained"])
            self.assertEqual(resumed["optimizer_steps"], 3)
            self.assertFalse(torch.equal(before["adapters.default.weight"], full_model.state_dict()["adapters.default.weight"]))
            for key, value in full_model.state_dict().items():
                torch.testing.assert_close(value, resumed_model.state_dict()[key], rtol=0, atol=0)
            full_state = torch.load(Path(full["checkpoint_path"]) / "training_state.pt", weights_only=False)
            resumed_state = torch.load(Path(resumed["checkpoint_path"]) / "training_state.pt", weights_only=False)
            for key, value in full_state["extra"]["predictor"].items():
                torch.testing.assert_close(value, resumed_state["extra"]["predictor"][key], rtol=0, atol=0)
            self.assertTrue((Path(resumed["checkpoint_path"]) / "COMPLETE").is_file())

    def test_sft_only_saves_no_predictor(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp, "sft.json")
            data.write_text(json.dumps([dict(question="Q", schema="t(x)", answer="SELECT x FROM t;")]))
            result = run_sft_training(data_path=str(data), output_dir=str(Path(tmp, "out")),
                model=TinyPolicy(), tokenizer=ByteTokenizer(), device="cpu", epochs=1,
                max_steps=1, lambda_jepa=0.0, max_length=1024)
            state = torch.load(Path(result["checkpoint_path"], "training_state.pt"), weights_only=False)
            self.assertIsNone(state["extra"]["predictor"])

    def test_grpo_updates_from_sampled_policy_and_resumes(self):
        try:
            import sqlglot
        except ImportError:
            self.skipTest("sqlglot required for real GRPO AST rewards")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sft = root / "sft"
            sft.mkdir()
            (sft / "adapter_config.json").write_text("{}")
            (sft / "adapter_model.bin").write_bytes(b"tiny-reference-identity")
            data = root / "grpo.json"
            data.write_text(json.dumps([dict(db_id="tiny", question=f"Q{i}",
                schema="CREATE TABLE t(x);", SQL="SELECT x FROM t;") for i in range(3)]))
            connection = sqlite3.connect(root / "tiny.sqlite")
            connection.execute("CREATE TABLE t(x)")
            connection.close()
            torch.manual_seed(9)
            initial = TinyPolicy()
            models = []
            def loader(*args, **kwargs):
                model = copy.deepcopy(initial)
                models.append(model)
                return model, ByteTokenizer()
            def reward(response, *args):
                # This test isolates policy optimization. SQL reward correctness is
                # tested separately against real SQLite; the model still samples.
                score = sum(int(x) for x in response.split(",")) % 13
                return dict(r_format=0., r_exec=0., r_mem=0., r_atomic=float(score), total_reward=float(score))
            common = dict(data_path=str(data), db_dir=str(root), sft_checkpoint=str(sft),
                device="cpu", epochs=3, max_length=512, max_new_tokens=3,
                num_rollouts=4, learning_rate=0.01, save_steps=50, seed=42)
            with patch("src.training.runtime.load_causal_model", side_effect=loader), \
                 patch("src.training.train_grpo_generator.compute_dense_rewards", side_effect=reward):
                full = run_grpo_training(output_dir=str(root / "full"), max_steps=3, **common)
                split = run_grpo_training(output_dir=str(root / "split"), max_steps=1, **common)
                resumed = run_grpo_training(output_dir=str(root / "split"), max_steps=3,
                    resume_from_checkpoint=split["checkpoint_path"], **common)
            self.assertEqual(resumed["optimizer_steps"], 3)
            self.assertTrue(full["trained"])
            self.assertFalse(torch.equal(initial.adapters["default"].weight, models[0].adapters["default"].weight))
            torch.testing.assert_close(models[0].adapters["reference"].weight,
                                       initial.adapters["reference"].weight, rtol=0, atol=0)
            for key, value in models[0].state_dict().items():
                torch.testing.assert_close(value, models[-1].state_dict()[key], rtol=0, atol=0)



try:
    import importlib.util
    HF_AVAILABLE = (torch is not None and importlib.util.find_spec("transformers") is not None
                    and importlib.util.find_spec("peft") is not None)
except ImportError:
    HF_AVAILABLE = False


@unittest.skipUnless(HF_AVAILABLE, "Transformers and PEFT required")
class TestNativeLlamaPeftTraining(unittest.TestCase):
    """Actual HF/PEFT checkpoint interfaces, tiny random local model, no downloads."""
    def _local_base(self, root):
        from tokenizers import Tokenizer, models, pre_tokenizers, processors
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
        vocab = {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, "[PAD]": 3}
        for word in ["Q", "SELECT", "COUNT", "FROM", "t", "x", "CREATE", "TABLE", "think",
                     "answer", "system", "schema", "question", "PRED", "(", ")", ";", "<", ">", "/", "*"]:
            vocab[word] = len(vocab)
        while len(vocab) < 64:
            vocab[f"word{len(vocab)}"] = len(vocab)
        core = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
        core.pre_tokenizer = pre_tokenizers.Whitespace()
        core.post_processor = processors.TemplateProcessing(single="[BOS] $A", special_tokens=[("[BOS]", 1)])
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, unk_token="[UNK]",
            bos_token="[BOS]", eos_token="[EOS]", pad_token="[PAD]",
            model_input_names=["input_ids", "attention_mask"])
        config = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=256, bos_token_id=1, eos_token_id=2, pad_token_id=3)
        torch.manual_seed(19)
        model = LlamaForCausalLM(config)
        base = root / "base"
        model.save_pretrained(base)
        tokenizer.save_pretrained(base)
        return base

    def test_native_sft_checkpoint_resume_at_epoch_boundary(self):
        from safetensors.torch import load_file
        from src.training.runtime import load_causal_model
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = self._local_base(root)
            data = root / "sft.json"
            data.write_text(json.dumps([dict(question=f"Q {i}", schema="CREATE TABLE t(x);",
                think_steps="COUNT", answer="SELECT COUNT(*) FROM t;") for i in range(3)]))
            common = dict(data_path=str(data), model_name_or_path=str(base),
                device="cpu", epochs=2, max_length=256, lora_rank=2,
                gradient_accumulation_steps=2, lambda_jepa=0.7, predictor_dim=4,
                loss_dropout=0.5, learning_rate=0.001, save_steps=50, seed=42)
            full = run_sft_training(output_dir=str(root / "full"), max_steps=3, **common)
            split = run_sft_training(output_dir=str(root / "split"), max_steps=2, **common)
            cursor = torch.load(Path(split["checkpoint_path"], "training_state.pt"), weights_only=False)["extra"]
            self.assertEqual((cursor["epoch"], cursor["batch_index"]), (1, 0))
            resumed = run_sft_training(output_dir=str(root / "split"), max_steps=3,
                resume_from_checkpoint=split["checkpoint_path"], **common)
            weights = load_file(str(Path(full["checkpoint_path"], "adapter_model.safetensors")))
            reloaded = load_file(str(Path(resumed["checkpoint_path"], "adapter_model.safetensors")))
            self.assertEqual(set(weights), set(reloaded))
            for key, value in weights.items():
                torch.testing.assert_close(value, reloaded[key], rtol=0, atol=0)
            model, tokenizer = load_causal_model(str(base), adapter_path=resumed["checkpoint_path"],
                                                training=False, device="cpu")
            self.assertTrue(hasattr(model, "peft_config"))
            with torch.no_grad():
                generated = model.generate(**tokenizer("Q", return_tensors="pt"), max_new_tokens=2)
            self.assertGreater(generated.shape[1], 1)

    def test_native_grpo_reference_export_and_resume(self):
        try:
            import sqlglot
        except ImportError:
            self.skipTest("SQLGlot required")
        from safetensors.torch import load_file
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = self._local_base(root)
            data = root / "sft.json"
            data.write_text(json.dumps([dict(question="Q", schema="CREATE TABLE t(x);",
                answer="SELECT x FROM t;")]))
            sft = run_sft_training(data_path=str(data), output_dir=str(root / "sft"),
                model_name_or_path=str(base), device="cpu", epochs=1, max_steps=1,
                max_length=256, lora_rank=2, lambda_jepa=0.0)
            grpo_data = root / "grpo.json"
            grpo_data.write_text(json.dumps([dict(db_id="tiny", question=f"Q {i}",
                schema="CREATE TABLE t(x);", SQL="SELECT x FROM t;") for i in range(3)]))
            conn = sqlite3.connect(root / "tiny.sqlite")
            conn.execute("CREATE TABLE t(x)")
            conn.close()
            def reward(response, *args):
                score = sum(map(ord, response)) % 19
                return dict(r_format=0., r_exec=0., r_mem=0., r_atomic=float(score), total_reward=float(score))
            common = dict(data_path=str(grpo_data), db_dir=str(root), sft_checkpoint=sft["checkpoint_path"],
                device="cpu", epochs=3, max_length=256, max_new_tokens=4,
                num_rollouts=4, learning_rate=0.001, save_steps=50, seed=42)
            with patch("src.training.train_grpo_generator.compute_dense_rewards", side_effect=reward):
                full = run_grpo_training(output_dir=str(root / "full"), max_steps=2, **common)
                split = run_grpo_training(output_dir=str(root / "split"), max_steps=1, **common)
                resumed = run_grpo_training(output_dir=str(root / "split"), max_steps=2,
                    resume_from_checkpoint=split["checkpoint_path"], **common)
            self.assertEqual(resumed["optimizer_steps"], 2)
            original = load_file(str(Path(sft["checkpoint_path"], "adapter_model.safetensors")))
            reference = load_file(str(Path(resumed["checkpoint_path"], "reference", "adapter_model.safetensors")))
            final = load_file(str(Path(full["checkpoint_path"], "adapter_model.safetensors")))
            restored = load_file(str(Path(resumed["checkpoint_path"], "adapter_model.safetensors")))
            for key, value in original.items():
                torch.testing.assert_close(value, reference[key], rtol=0, atol=0)
            for key, value in final.items():
                torch.testing.assert_close(value, restored[key], rtol=0, atol=0)
            self.assertTrue(any(not torch.equal(final[key], value) for key, value in original.items()))


if __name__ == "__main__":
    unittest.main()
