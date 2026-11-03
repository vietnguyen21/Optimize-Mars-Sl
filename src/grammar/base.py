"""
Base Grammar Engine Interface for Text-to-SQL SLM Architecture.
Defines abstract contracts for grammar-constrained decoding, pushdown automata (PDA),
and zero-crash fallback engines.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any, List, Union


class BaseGrammarEngine(ABC):
    """
    Abstract base class for grammar-constrained decoding engines.
    Provides polymorphic interface for compiling SQL grammars from schema DDLs,
    filtering generation logits at runtime, and validating SQL syntax.
    """

    def __init__(self, schema_ddl: str = ""):
        self.schema_ddl: str = schema_ddl.strip()
        self.compiled_grammar: Optional[Any] = None
        if self.schema_ddl:
            self.compiled_grammar = self.compile_grammar(self.schema_ddl)

    @abstractmethod
    def compile_grammar(self, schema_ddl: str) -> Any:
        """
        Compiles the target database schema DDL into a structured grammar (BNF/EBNF or PDA).
        
        Args:
            schema_ddl: DDL statements defining tables and columns.
            
        Returns:
            Compiled grammar structure or AST definition.
        """
        pass

    @abstractmethod
    def filter_logits(self, logits: Any, context: str) -> Any:
        """
        Filters vocabulary logits given the current SQL prefix context using a bitmask
        or token admissibility rules.
        
        Args:
            logits: Logits vector or 1D/2D array of token scores.
            context: Current partial SQL prefix string.
            
        Returns:
            Masked logits where invalid token transitions have -inf or minimal score.
        """
        pass

    @abstractmethod
    def validate_sql_syntax(self, sql: str) -> Tuple[bool, Optional[str]]:
        """
        Validates complete SQL statement for syntactic and dialect correctness.
        
        Args:
            sql: SQL statement string.
            
        Returns:
            Tuple of (is_valid: bool, error_message: Optional[str]).
        """
        pass

    def validate_prefix(self, prefix_sql: str) -> Tuple[bool, Optional[str]]:
        """
        Validates whether a partial prefix can lead to a valid SQL statement.
        
        Args:
            prefix_sql: Incomplete SQL query prefix.
            
        Returns:
            Tuple of (is_valid_prefix: bool, error_message: Optional[str]).
        """
        return True, None

    def validate_or_mask(self, token_ids: List[int], prefix_sql: str) -> Tuple[bool, Any]:
        """
        Unified compatibility interface supporting prefix validation and token masking.
        Preserves compatibility with Notebook 05 and evaluation pipelines.
        
        Args:
            token_ids: List of generated token ids so far (can be empty).
            prefix_sql: Current prefix string.
            
        Returns:
            Tuple of (is_valid: bool, mask_or_detail: Any).
        """
        is_valid, err = self.validate_prefix(prefix_sql)
        return is_valid, err
