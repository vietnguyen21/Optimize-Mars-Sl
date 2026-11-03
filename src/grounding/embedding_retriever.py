"""
LitE-SQL Schema Retrieval and Column Card Embedding.
Retrieves the Top-K relevant columns and tables for a given query and evidence.
Supports dense vector embeddings with cosine similarity and provides a robust,
mathematically sound TF-IDF / BM25 field-aware offline fallback.
"""

import math
import re
import os
import json
from typing import List, Dict, Tuple, Optional, Any, Set
from src.common.schemas import ColumnCard

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class LitESQLEmbeddingRetriever:
    """
    Schema retriever implementing LitE-SQL 0.6B column selection logic.
    Provides vector embeddings with cosine similarity, and an offline BM25 / TF-IDF
    field-aware engine when embedding weights are not present.
    """

    def __init__(
        self,
        cards: Optional[List[ColumnCard]] = None,
        model_name_or_path: Optional[str] = None,
        top_k: int = 25,
    ):
        self.top_k = top_k
        self.cards: List[ColumnCard] = cards or []
        self.model = None
        self.model_loaded = False
        self.card_vectors = None

        if model_name_or_path:
            self._try_load_model(model_name_or_path)

        # Build offline corpus index
        self._build_offline_index()

    def set_cards(self, cards: List[ColumnCard]) -> None:
        """Sets the column cards and rebuilds the search indices."""
        self.cards = cards
        self._build_offline_index()
        if self.model_loaded:
            self._compute_dense_embeddings()

    def _try_load_model(self, model_path: str) -> None:
        """Attempts to load a SentenceTransformer or HuggingFace dense model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_path)
            self.model_loaded = True
            if self.cards:
                self._compute_dense_embeddings()
        except Exception:
            self.model = None
            self.model_loaded = False

    def _compute_dense_embeddings(self) -> None:
        """Computes dense vector representations for all column cards."""
        if not self.model_loaded or not self.cards or not HAS_NUMPY:
            return
        texts = [card.to_text() for card in self.cards]
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        self.card_vectors = embeddings

    def _build_offline_index(self) -> None:
        """Builds term frequency, document frequency, and field weights for BM25/TF-IDF."""
        self.doc_count = len(self.cards)
        self.doc_lengths: List[float] = []
        self.doc_tokens: List[Dict[str, float]] = []
        self.doc_freq: Dict[str, int] = {}

        field_weights = {
            "column_name": 4.0,
            "table_name": 3.0,
            "column_description": 2.5,
            "sample_values": 2.0,
            "constraints": 1.0,
            "data_type": 0.5,
        }

        for card in self.cards:
            token_counts: Dict[str, float] = {}
            total_len = 0.0

            # Weight tokens by field prominence
            fields_data = {
                "column_name": self._tokenize(card.column_name),
                "table_name": self._tokenize(card.table_name),
                "column_description": self._tokenize(card.column_description),
                "sample_values": [tok for val in card.sample_values for tok in self._tokenize(str(val))],
                "constraints": [tok for c in card.constraints for tok in self._tokenize(str(c))],
                "data_type": self._tokenize(card.data_type),
            }

            for field_name, tokens in fields_data.items():
                weight = field_weights.get(field_name, 1.0)
                for t in tokens:
                    token_counts[t] = token_counts.get(t, 0.0) + weight
                    total_len += weight

            self.doc_lengths.append(total_len)
            self.doc_tokens.append(token_counts)

            # Document frequency
            unique_terms = set(token_counts.keys())
            for t in unique_terms:
                self.doc_freq[t] = self.doc_freq.get(t, 0) + 1

        self.avg_doc_len = sum(self.doc_lengths) / max(1, self.doc_count)

    def _tokenize(self, text: str) -> List[str]:
        """Splits text into lowercase alphanumeric normalized tokens."""
        if not text:
            return []
        # Replace punctuation and split camelCase / snake_case
        s1 = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text))
        tokens = re.findall(r"[A-Za-z0-9]+", s1.lower())
        res = []
        for t in tokens:
            res.append(t)
            # Add simple singular form if ends with 's' and length > 3
            if t.endswith("s") and len(t) > 3 and not t.endswith("ss"):
                res.append(t[:-1])
        return res

    def retrieve(
        self,
        question: Optional[str] = None,
        evidence: Optional[str] = "",
        top_k: Optional[int] = None,
        query: Optional[str] = None,
        **kwargs: Any
    ) -> Tuple[List[ColumnCard], Dict[str, List[str]], Dict[str, float]]:
        """
        Retrieves the Top-K column cards relevant to the question and evidence.

        Args:
            question: Natural language question.
            evidence: External domain knowledge or hints.
            top_k: Number of columns to retrieve (default self.top_k = 25).
            query: Alias for question.

        Returns:
            Tuple of:
            - ranked_cards: List of selected ColumnCard objects.
            - terminal_tables: Dict mapping table_name -> list of selected column names.
            - similarity_scores: Dict mapping "table.column" -> float score.
        """
        resolved_question = query if query is not None else (question or "")
        k = top_k or self.top_k
        if not self.cards:
            return [], {}, {}

        # 1. Use dense embedding if model is loaded and vectors are available
        if self.model_loaded and self.card_vectors is not None and HAS_NUMPY:
            return self._retrieve_dense(question=resolved_question, evidence=evidence, k=k)

        # 2. Robust offline BM25 / field-aware retrieval
        return self._retrieve_offline(question=resolved_question, evidence=evidence, k=k)

    def _retrieve_dense(
        self, question: str, evidence: Optional[str], k: int
    ) -> Tuple[List[ColumnCard], Dict[str, List[str]], Dict[str, float]]:
        """Dense embedding retrieval using cosine similarity."""
        query_text = f"Query: {question} Evidence: {evidence or ''}".strip()
        q_vec = self.model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)[0]

        # Cosine similarity (already normalized vectors)
        scores = np.dot(self.card_vectors, q_vec)
        top_indices = np.argsort(scores)[::-1][:k]

        selected_cards: List[ColumnCard] = []
        terminal_tables: Dict[str, List[str]] = {}
        score_dict: Dict[str, float] = {}

        for idx in top_indices:
            card = self.cards[idx]
            score = float(scores[idx])
            selected_cards.append(card)
            score_dict[f"{card.table_name}.{card.column_name}"] = score
            if card.table_name not in terminal_tables:
                terminal_tables[card.table_name] = []
            if card.column_name not in terminal_tables[card.table_name]:
                terminal_tables[card.table_name].append(card.column_name)

        return selected_cards, terminal_tables, score_dict

    def _retrieve_offline(
        self, question: str, evidence: Optional[str], k: int
    ) -> Tuple[List[ColumnCard], Dict[str, List[str]], Dict[str, float]]:
        """
        Field-aware BM25 and exact-phrase match retrieval engine.
        Operates completely offline without requiring model weight downloads.
        """
        query_text = f"{question} {evidence or ''}"
        query_tokens = self._tokenize(query_text)
        query_raw_lower = query_text.lower()

        # BM25 parameters
        k1 = 1.5
        b = 0.75

        scored_cards: List[Tuple[float, ColumnCard]] = []

        for idx, card in enumerate(self.cards):
            doc_len = self.doc_lengths[idx]
            doc_toks = self.doc_tokens[idx]
            bm25_score = 0.0

            for q_term in query_tokens:
                if q_term in doc_toks:
                    tf = doc_toks[q_term]
                    df = self.doc_freq.get(q_term, 1)
                    # Lucene / Robertson-Spärck Jones IDF
                    idf = math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))
                    idf = max(0.1, idf)

                    # BM25 term score
                    denom = tf + k1 * (1.0 - b + b * (doc_len / max(1.0, self.avg_doc_len)))
                    bm25_score += idf * ((tf * (k1 + 1.0)) / denom)

            # Token-level and exact name match bonuses
            bonus = 0.0
            col_name_lower = card.column_name.lower()
            tbl_name_lower = card.table_name.lower()

            # Column name token matches (e.g. 'math' in 'AvgScrMath')
            col_tokens = set(self._tokenize(card.column_name))
            matching_col_tokens = col_tokens.intersection(set(query_tokens))
            if matching_col_tokens:
                bonus += 4.5 * len(matching_col_tokens)

            # Exact full column name match
            if col_name_lower in query_raw_lower:
                bonus += 5.0

            # Table name token matches
            tbl_tokens = set(self._tokenize(card.table_name))
            matching_tbl_tokens = tbl_tokens.intersection(set(query_tokens))
            if matching_tbl_tokens:
                bonus += 1.5 * len(matching_tbl_tokens)

            # Table exact full name match
            if tbl_name_lower in query_raw_lower:
                bonus += 1.5

            # Sample value exact hit bonus
            for s in card.sample_values:
                s_lower = str(s).lower()
                if len(s_lower) >= 2 and s_lower in query_raw_lower:
                    bonus += 5.0
                    break

            # Primary key or foreign key priority if table is mentioned
            if ("PRIMARY KEY" in card.constraints or any("FOREIGN KEY" in c for c in card.constraints)) and (tbl_name_lower in query_raw_lower or matching_tbl_tokens):
                bonus += 2.0

            total_score = bm25_score + bonus
            scored_cards.append((total_score, card))

        # Sort descending by score
        scored_cards.sort(key=lambda x: x[0], reverse=True)

        selected_cards: List[ColumnCard] = []
        terminal_tables: Dict[str, List[str]] = {}
        score_dict: Dict[str, float] = {}

        # Normalize score range for clarity
        max_score = max(scored_cards[0][0], 1e-6) if scored_cards else 1.0

        for score, card in scored_cards[:k]:
            normalized_score = round(float(score / max_score), 4)
            selected_cards.append(card)
            score_dict[f"{card.table_name}.{card.column_name}"] = normalized_score
            if card.table_name not in terminal_tables:
                terminal_tables[card.table_name] = []
            if card.column_name not in terminal_tables[card.table_name]:
                terminal_tables[card.table_name].append(card.column_name)

        return selected_cards, terminal_tables, score_dict

    def export_offline_index(self, output_path: str) -> None:
        """
        Exports the precomputed index to disk for fast offline inference.
        If numpy is available and dense vectors exist, saves .npy; otherwise saves JSON.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        if HAS_NUMPY and self.card_vectors is not None:
            np.save(output_path, self.card_vectors)
        else:
            # Save lightweight JSON index
            data = {
                "doc_count": self.doc_count,
                "avg_doc_len": self.avg_doc_len,
                "doc_freq": self.doc_freq,
                "cards": [c.model_dump() for c in self.cards],
            }
            json_path = output_path if output_path.endswith(".json") else f"{output_path}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)


# Top-level alias
EmbeddingRetriever = LitESQLEmbeddingRetriever
