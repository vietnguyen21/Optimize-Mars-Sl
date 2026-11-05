"""
Multi-Candidate SQL Trajectory Generator.
Samples 8 to 16 candidate rollouts per natural language query from a loaded checkpoint using
configurable temperature (0.7), top_p (0.9), and top_k (50) sampling parameters.

Integrity rule: the generator never sees gold SQL and never fabricates candidates. If no model
is loaded or generation fails, it raises instead of returning synthetic output.
"""

import json
import os
from typing import List, Optional, Tuple, Union, Any
from src.common.schemas import SQLCandidate, GenerationInput


def build_prompt(question: str, evidence: str = "", schema_ddl: str = "", plan: str = "") -> str:
    """Builds the generator prompt from inference-time inputs only (question, evidence, S' DDL, plan)."""
    parts = []
    if schema_ddl:
        parts.append(f"<schema>\n{schema_ddl.strip()}\n</schema>")
    if evidence:
        parts.append(f"<evidence>{evidence.strip()}</evidence>")
    if plan:
        parts.append(f"<plan>{plan}</plan>")
    parts.append(f"<question>{question.strip()}</question>")
    parts.append("<think>")
    return "\n".join(parts)


def load_generator(model_path: str) -> Tuple[Any, Any]:
    """
    Loads a checkpoint for inference on GPU in bf16/fp16. LoRA adapters are merged into the base
    model: unmerged adapters run at about half the speed (8.8 vs 19 tok/s measured on a T4).
    Any failure raises; there is no fallback generator.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    adapter_config = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_config):
        from peft import PeftModel
        with open(adapter_config, encoding="utf-8") as f:
            base = json.load(f)["base_model_name_or_path"]
        model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype, device_map="auto")
        model = PeftModel.from_pretrained(model, model_path).merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, device_map="auto")
    return model.eval(), tokenizer


class CandidateGenerator:
    """Samples K candidate SQL trajectories from a HuggingFace causal LM."""

    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        default_num_candidates: int = 8,
        default_temperature: float = 0.7,
        default_top_p: float = 0.9,
        default_top_k: int = 50,
        max_new_tokens: int = 512
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.default_num_candidates = default_num_candidates
        self.default_temperature = default_temperature
        self.default_top_p = default_top_p
        self.default_top_k = default_top_k
        self.max_new_tokens = max_new_tokens

    def generate(
        self,
        prompt_or_input: Union[GenerationInput, str],
        num_candidates: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        evidence: str = "",
        schema_ddl: str = "",
        plan: str = ""
    ) -> List[SQLCandidate]:
        """
        Generates K candidate trajectories (K clamped to [1, 32]).

        Args:
            prompt_or_input: GenerationInput instance or natural language question string.
            num_candidates, temperature, top_p, top_k: sampling overrides.
            evidence, schema_ddl: used only when prompt_or_input is a plain question string.

        Raises:
            RuntimeError: if no model/tokenizer is loaded or generation fails.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("CandidateGenerator has no loaded model/tokenizer; refusing to fabricate candidates.")

        if isinstance(prompt_or_input, GenerationInput):
            question = prompt_or_input.question
            evidence = prompt_or_input.evidence or ""
            schema_ddl = prompt_or_input.sub_schema.ddl_statements
            k = num_candidates or prompt_or_input.num_trajectories
            temp = temperature or prompt_or_input.temperature
            p = top_p or prompt_or_input.top_p
            k_val = top_k or prompt_or_input.top_k
        else:
            question = str(prompt_or_input)
            k = num_candidates or self.default_num_candidates
            temp = temperature or self.default_temperature
            p = top_p or self.default_top_p
            k_val = top_k or self.default_top_k

        k = max(1, min(k, 32))
        prompt = build_prompt(question, evidence, schema_ddl, plan)
        return self._generate_with_model(prompt, k, temp, p, k_val)

    def _generate_with_model(
        self,
        prompt: str,
        num_candidates: int,
        temperature: float,
        top_p: float,
        top_k: int
    ) -> List[SQLCandidate]:
        """Runs autoregressive sampling with the loaded model. Errors propagate to the caller."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[-1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=num_candidates,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            output_scores=True,
            return_dict_in_generate=True
        )

        # Sequence log-probabilities from the sampling scores (sum over generated tokens).
        if hasattr(self.model, "compute_transition_scores"):
            seq_logprobs = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            ).sum(dim=-1).tolist()
        else:
            seq_logprobs = [0.0] * len(outputs.sequences)

        candidates = []
        for idx, seq in enumerate(outputs.sequences):
            # Decode only the completion; the prompt already ends with "<think>".
            text = "<think>" + self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            reasoning = ""
            sql = text
            if "</think>" in text:
                reasoning = text.split("<think>", 1)[1].split("</think>", 1)[0].strip()
                sql = text.split("</think>", 1)[1]
            if "<answer>" in sql and "</answer>" in sql:
                sql = sql.split("<answer>", 1)[1].split("</answer>", 1)[0]
            candidates.append(SQLCandidate(
                candidate_id=idx,
                sql=sql.strip(),
                reasoning=reasoning,
                logprob=float(seq_logprobs[idx])
            ))
        return candidates
