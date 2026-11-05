"""
Grammar-Constrained Decoding Package for Text-to-SQL SLM Architecture.
Exports BaseGrammarEngine, XGrammarPDAEngine, RegexASTFallbackEngine,
and the unified get_grammar_engine factory.
"""

from typing import Optional
from src.grammar.base import BaseGrammarEngine
from src.grammar.fallback_engine import RegexASTFallbackEngine
from src.grammar.xgrammar_engine import XGrammarPDAEngine, HAS_XGRAMMAR


def get_grammar_engine(engine_type: str = "auto", schema_ddl: str = "") -> BaseGrammarEngine:
    """
    Factory function for obtaining grammar-constrained decoding engines.

    Args:
        engine_type: "auto", "xgrammar", "pda", "fallback", or "regex".
        schema_ddl: Schema DDL statement string used to compile domain grammars.

    Returns:
        Instance of BaseGrammarEngine (XGrammarPDAEngine or RegexASTFallbackEngine).
    """
    mode = engine_type.lower()
    if mode in {"xgrammar", "pda"}:
        return XGrammarPDAEngine(schema_ddl=schema_ddl)
    elif mode in {"fallback", "regex"}:
        return RegexASTFallbackEngine(schema_ddl=schema_ddl)
    elif mode == "auto":
        # In auto mode: if xgrammar is available, use XGrammarPDAEngine;
        # otherwise, XGrammarPDAEngine still provides pure-Python PDA stack & bitmask simulation,
        # but RegexASTFallbackEngine is the primary zero-crash fallback engine.
        return XGrammarPDAEngine(schema_ddl=schema_ddl)
    else:
        raise ValueError(f"Unknown grammar engine type: {engine_type}. Expected 'auto', 'xgrammar', or 'fallback'.")


__all__ = [
    "BaseGrammarEngine",
    "XGrammarPDAEngine",
    "RegexASTFallbackEngine",
    "get_grammar_engine",
    "HAS_XGRAMMAR"
]
