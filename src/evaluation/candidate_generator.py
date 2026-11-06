"""Checkpoint-sampled SQL candidates with explicit, non-benchmark fixture mode."""

import math
import random
import re
from typing import List, Optional, Union, Dict, Any
from src.common.schemas import SQLCandidate, GenerationInput, CompactSubSchema


class CandidateGenerator:
    """
    Candidate trajectory generator supporting 8 to 16 rollouts per prompt.
    Real mode samples a loaded checkpoint and propagates failures.
    Explicit dry_run mode preserves legacy fixtures for offline plumbing tests.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        default_num_candidates: int = 8,
        default_temperature: float = 0.7,
        default_top_p: float = 0.9,
        default_top_k: int = 50,
        dry_run: bool = False,
        max_new_tokens: int = 256,
        max_prompt_length: int = 1024,
        constraint_mode: str = "posthoc",
    ):
        self.dry_run = dry_run
        self.max_new_tokens = max_new_tokens
        self.max_prompt_length = max_prompt_length
        self.constraint_mode = constraint_mode
        if constraint_mode not in {"posthoc", "xgrammar"}:
            raise ValueError("constraint_mode must be posthoc or xgrammar")
        self.model = model
        self.tokenizer = tokenizer
        self.default_num_candidates = default_num_candidates
        self.default_temperature = default_temperature
        self.default_top_p = default_top_p
        self.default_top_k = default_top_k

    def generate(
        self,
        prompt_or_input: Union[GenerationInput, str],
        gold_sql: Optional[str] = None,
        num_candidates: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None
    ) -> List[SQLCandidate]:
        """
        Generates K candidate trajectories (where K is typically in [8, 16]).
        
        Args:
            prompt_or_input: GenerationInput instance or natural language question string.
            gold_sql: Legacy fixture reference, allowed only with explicit dry_run=True.
            num_candidates: Number of trajectories to sample (default 8, supports up to 16+).
            temperature: Sampling temperature (default 0.7).
            top_p: Nucleus sampling probability cutoff (default 0.9).
            top_k: Top-k vocabulary truncation cutoff (default 50).
            
        Returns:
            List of SQLCandidate objects.
        """
        if gold_sql is not None and not self.dry_run:
            raise ValueError("Gold SQL is forbidden during real candidate generation")
        if isinstance(prompt_or_input, GenerationInput):
            question = prompt_or_input.question
            k = num_candidates or prompt_or_input.num_trajectories
            temp = temperature or prompt_or_input.temperature
            p = top_p or prompt_or_input.top_p
            k_val = top_k or prompt_or_input.top_k
            sub_schema = prompt_or_input.sub_schema
        else:
            question = str(prompt_or_input)
            k = num_candidates or self.default_num_candidates
            temp = temperature or self.default_temperature
            p = top_p or self.default_top_p
            k_val = top_k or self.default_top_k
            sub_schema = None

        # Clamp k to valid ablation range [1, 32], default expectation is [8, 16]
        k = max(1, min(k, 32))

        # If a live HuggingFace model and tokenizer are available, run model.generate
        if self.model is not None and self.tokenizer is not None:
            return self._generate_with_model(question, k, temp, p, k_val, sub_schema)

        if not self.dry_run:
            raise RuntimeError("Real generation requires a loaded model and tokenizer; use dry_run=True only for fixtures")
        return self._generate_ablation_trajectories(question, gold_sql, k, temp)

    def _generate_with_model(
        self,
        question: str,
        num_candidates: int,
        temperature: float,
        top_p: float,
        top_k: int,
        sub_schema: Optional[CompactSubSchema]
    ) -> List[SQLCandidate]:
        """Sample serially to keep K independent rollouts within one T4's memory."""
        import torch
        from src.training.runtime import build_prompt
        schema = sub_schema.ddl_statements if sub_schema else ""
        prompt = build_prompt(question, schema)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        if inputs["input_ids"].shape[1] > self.max_prompt_length:
            raise ValueError("Grounded prompt exceeds max_prompt_length; reduce retrieval top_k or increase the explicit budget")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        prompt_length = inputs["input_ids"].shape[1]
        self.model.eval()
        candidates = []
        for idx in range(num_candidates):
            generation_args = {}
            if self.constraint_mode == "xgrammar":
                from src.grammar.xgrammar_engine import make_hf_sql_processor
                generation_args["logits_processor"] = [make_hf_sql_processor(
                    self.tokenizer, self.model.config.vocab_size, schema
                )]
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens,
                    num_return_sequences=1, do_sample=True,
                    temperature=temperature, top_p=top_p, top_k=top_k,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    output_scores=True, return_dict_in_generate=True,
                    **generation_args,
                )
                generated = outputs.sequences[0, prompt_length:]
                text = self.tokenizer.decode(generated, skip_special_tokens=True)
                transitions = self.model.compute_transition_scores(
                    outputs.sequences, outputs.scores, normalize_logits=True
                )[0]
                # Mean sampled-token log probability, not a rank-derived constant.
                logprob = float(transitions.mean().item()) if transitions.numel() else 0.0
            reasoning_match = re.search(r"<think>(.*?)</think>", text, re.S)
            answer_match = re.search(r"<answer>(.*?)</answer>", text, re.S)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
            sql = answer_match.group(1).strip() if answer_match else ""
            candidates.append(SQLCandidate(candidate_id=idx, sql=sql,
                                           reasoning=reasoning, logprob=logprob))
            del outputs, generated, transitions
        return candidates

    def _generate_ablation_trajectories(
        self,
        question: str,
        gold_sql: Optional[str],
        num_candidates: int,
        temperature: float
    ) -> List[SQLCandidate]:
        """
        Produces diverse candidate trajectories for ablation studies and smoke tests.
        Generates realistic reasoning traces and diverse structural candidates.
        """
        candidates: List[SQLCandidate] = []
        base_sql = gold_sql.strip() if gold_sql else "SELECT 1;"

        # Candidate 0: Primary policy beam (direct execution)
        clean_base = base_sql.rstrip(";")
        candidates.append(SQLCandidate(
            candidate_id=0,
            sql=f"{clean_base};",
            reasoning=f"Identified primary entity relations for question: '{question}'. Constructed optimal relational join projection.",
            logprob=-0.05 / max(0.1, temperature)
        ))

        # Diverse structural rollout variations for candidates 1 to k-1
        variations = [
            # 1. Alias qualification variation
            lambda s: self._add_explicit_aliases(s),
            # 2. Uppercase keyword normalization
            lambda s: self._normalize_keywords(s),
            # 3. Predicate reordering or spacing
            lambda s: self._reorder_predicates(s),
            # 4. Explicit ASC in ORDER BY
            lambda s: s.replace("ORDER BY", "ORDER BY").replace(";", " ASC;") if "DESC" not in s else s,
            # 5. Comment annotated variant
            lambda s, i=1: f"{s.rstrip(';')} -- rollout beam {i};",
            # 6. Parenthesis around conditions
            lambda s: self._parenthesize_where(s),
            # 7. Explicit table prefix
            lambda s: s.replace("JOIN", "INNER JOIN") if "JOIN" in s and "LEFT" not in s else s,
            # 8. Distinct preserving projection
            lambda s: s,
        ]

        for i in range(1, num_candidates):
            var_fn = variations[(i - 1) % len(variations)]
            try:
                if var_fn == variations[4]: # comment annotation
                    cand_sql = f"{clean_base} -- trajectory_{i};"
                else:
                    cand_sql = var_fn(clean_base)
                    if not cand_sql.endswith(";"):
                        cand_sql = f"{cand_sql};"
            except Exception:
                cand_sql = f"{clean_base} -- variant_{i};"

            # Compute realistic decaying logprobs based on temperature
            base_lp = -0.15 * i
            noise = (i % 3) * 0.05
            logprob = (base_lp - noise) / max(0.1, temperature)

            reasoning = f"Step {i}: Explored candidate trajectory variant {i} under sampling temperature={temperature:.2f}."
            candidates.append(SQLCandidate(
                candidate_id=i,
                sql=cand_sql,
                reasoning=reasoning,
                logprob=round(logprob, 4)
            ))

        return candidates

    def _add_explicit_aliases(self, sql: str) -> str:
        """Adds explicit table aliases if missing."""
        if " AS " not in sql.upper() and " FROM " in sql.upper():
            return sql.replace("FROM accounts", "FROM accounts AS T1").replace("accounts.", "T1.")
        return sql

    def _normalize_keywords(self, sql: str) -> str:
        """Uppercases common keywords."""
        keywords = ["select", "from", "where", "join", "on", "and", "or", "group by", "order by", "limit"]
        res = sql
        for kw in keywords:
            res = re.sub(rf"\b{kw}\b", kw.upper(), res, flags=re.IGNORECASE)
        return res

    def _reorder_predicates(self, sql: str) -> str:
        """Swaps two AND predicates if present."""
        if " AND " in sql.upper():
            parts = re.split(r"\s+AND\s+", sql, flags=re.IGNORECASE)
            if len(parts) == 2 and "WHERE" in parts[0].upper():
                w_split = re.split(r"\s+WHERE\s+", parts[0], flags=re.IGNORECASE)
                if len(w_split) == 2:
                    return f"{w_split[0]} WHERE {parts[1].rstrip(';')} AND {w_split[1]};"
        return sql

    def _parenthesize_where(self, sql: str) -> str:
        """Wraps where clause in parentheses."""
        if "WHERE " in sql.upper() and "(" not in sql:
            parts = re.split(r"\s+WHERE\s+", sql, flags=re.IGNORECASE)
            if len(parts) == 2:
                cond = parts[1].rstrip(";")
                return f"{parts[0]} WHERE ({cond});"
        return sql


# Global convenience function matching Notebook 05 interface
def generate_candidate_rollouts(
    question: str,
    gold_sql: str,
    num_candidates: int = 8
) -> List[SQLCandidate]:
    """
    Legacy fixture helper for offline tests only; never use for evaluation.
    Real evaluation calls CandidateGenerator.generate without gold SQL.
    """
    generator = CandidateGenerator(default_num_candidates=num_candidates, dry_run=True)
    return generator.generate(question, gold_sql=gold_sql, num_candidates=num_candidates)
