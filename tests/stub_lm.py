"""
Tiny stand-in for a HuggingFace tokenizer + causal LM, used only by tests and the smoke script.
It emits a fixed completion and records every prompt it receives. It has no access to gold SQL,
so it lets tests check pipeline wiring and the no-gold-leak rule without a GPU.
"""

from types import SimpleNamespace


class _Ids(list):
    @property
    def shape(self):
        return (1, len(self))


class StubTokenizer:
    def __init__(self, completion: str = "reasoning</think>\n<answer>\nSELECT 1;\n</answer>"):
        self.completion = completion
        self.prompts = []

    def __call__(self, text, return_tensors=None):
        self.prompts.append(text)
        return {"input_ids": _Ids(range(len(text.split())))}

    def decode(self, ids, skip_special_tokens=True):
        return self.completion


class StubModel:
    def generate(self, input_ids, num_return_sequences=1, **kwargs):
        seq = list(input_ids) + [0]
        return SimpleNamespace(sequences=[seq] * num_return_sequences, scores=None)
