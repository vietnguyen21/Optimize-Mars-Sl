"""
Pydantic data models and interface contracts for the Text-to-SQL SLM Architecture.
Defines canonical exchange contracts across Grounding, Generation, Validation, and Benchmark modules.
"""

from typing import List, Dict, Tuple, Optional, Any, Union
import json

try:
    from pydantic import BaseModel as _PydanticBaseModel, Field, ConfigDict
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    _PydanticBaseModel = object
    Field = lambda default=None, **kwargs: default


if HAS_PYDANTIC:
    class BaseModel(_PydanticBaseModel):
        """Pydantic BaseModel with dict/dump serialization compatibility."""
        model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

        def to_dict(self) -> Dict[str, Any]:
            return self.model_dump()
else:
    class BaseModel:
        """Lightweight pure-Python fallback for BaseModel when pydantic is not installed."""
        def __init__(self, **kwargs):
            # Read annotations and default values from class hierarchy
            cls = self.__class__
            defaults = {}
            for base in reversed(cls.__mro__):
                for k, v in getattr(base, "__annotations__", {}).items():
                    if hasattr(base, k):
                        defaults[k] = getattr(base, k)
                    else:
                        defaults[k] = None

            # Apply defaults then kwargs
            for k, default_val in defaults.items():
                setattr(self, k, default_val)

            for k, v in kwargs.items():
                setattr(self, k, v)

        def dict(self) -> Dict[str, Any]:
            return self.model_dump()

        def model_dump(self) -> Dict[str, Any]:
            result = {}
            for k, v in self.__dict__.items():
                if k.startswith("_"):
                    continue
                if isinstance(v, BaseModel):
                    result[k] = v.model_dump()
                elif isinstance(v, list):
                    result[k] = [item.model_dump() if isinstance(item, BaseModel) else item for item in v]
                elif isinstance(v, dict):
                    result[k] = {
                        dk: (dv.model_dump() if isinstance(dv, BaseModel) else dv)
                        for dk, dv in v.items()
                    }
                else:
                    result[k] = v
            return result

        def to_dict(self) -> Dict[str, Any]:
            return self.model_dump()

        @classmethod
        def model_validate(cls, obj: Any) -> "BaseModel":
            if isinstance(obj, cls):
                return obj
            if isinstance(obj, dict):
                return cls(**obj)
            raise ValueError(f"Cannot validate {type(obj)} to {cls.__name__}")

        def __repr__(self) -> str:
            attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items() if not k.startswith("_"))
            return f"{self.__class__.__name__}({attrs})"

        def __eq__(self, other: Any) -> bool:
            if not isinstance(other, self.__class__):
                return False
            return self.model_dump() == other.model_dump()


class ColumnCard(BaseModel):
    """
    6-Field Column Card standard representation.
    Used by LitE-SQL 0.6B schema retriever for dense embedding & schema linking.
    """
    table_name: str
    column_name: str
    data_type: str = "TEXT"
    constraints: List[str] = []
    column_description: str = ""
    sample_values: List[str] = []

    def __init__(self, **data):
        # Support constraints as string or list
        if "constraints" in data and isinstance(data["constraints"], str):
            c_str = data["constraints"].strip()
            data["constraints"] = [c.strip() for c in c_str.split(",") if c.strip()] if c_str else []
        # Ensure sample_values are strings
        if "sample_values" in data and isinstance(data["sample_values"], list):
            data["sample_values"] = [str(v) for v in data["sample_values"] if v is not None]
        super().__init__(**data)

    def to_text(self) -> str:
        """Formats the 6-field column card into canonical text for embedding or prompts."""
        constraints_str = ", ".join(self.constraints) if self.constraints else "NONE"
        samples_str = ", ".join(f"'{v}'" for v in self.sample_values) if self.sample_values else "NONE"
        return (
            f"Table: {self.table_name} | "
            f"Column: {self.column_name} | "
            f"Type: {self.data_type} | "
            f"Constraints: {constraints_str} | "
            f"Description: {self.column_description or 'None'} | "
            f"Sample Values: [{samples_str}]"
        )


class CompactSubSchema(BaseModel):
    """
    Contract for Compact Sub-Schema output from Agent 1 (Grounding).
    Preserves all terminal tables, bridge tables, and PK-FK join edges.
    """
    db_id: str
    selected_tables: List[str]
    selected_columns: Dict[str, List[str]] # table_name -> list of column names
    foreign_keys: List[Tuple[str, str, str, str]] = [] # list of (from_tbl, from_col, to_tbl, to_col)
    ddl_statements: str = ""
    bridge_tables_added: List[str] = []
    graph_connectivity_verified: bool = True

    def get_join_clauses(self) -> List[str]:
        """Returns readable JOIN clauses from the foreign key edges."""
        clauses = []
        for t1, c1, t2, c2 in self.foreign_keys:
            clauses.append(f"{t1}.{c1} = {t2}.{c2}")
        return clauses


class TableGraphNode(BaseModel):
    """Represents a table node within the schema graph G = (V, E)."""
    table_name: str
    columns: List[str] = []
    primary_keys: List[str] = []
    weight: float = 1.0


class FKEdge(BaseModel):
    """Represents a foreign-key directed/undirected edge in the schema graph."""
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    weight: float = 1.0

    def as_tuple(self) -> Tuple[str, str, str, str]:
        return (self.from_table, self.from_column, self.to_table, self.to_column)


class GenerationInput(BaseModel):
    """Standard input payload passed to Agent 2 (SQL Generation)."""
    question: str
    evidence: Optional[str] = ""
    sub_schema: CompactSubSchema
    num_trajectories: int = 8
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    use_xgrammar: bool = True


class SQLCandidate(BaseModel):
    """Represents an individual SQL generation rollout candidate."""
    candidate_id: int
    sql: str
    reasoning: str = ""
    logprob: float = 0.0


class ValidationResult(BaseModel):
    """Verification and assessment report produced by Agent 3 (Validation)."""
    candidate_id: int
    is_valid_syntax: bool
    pv_sql_passed: bool
    probe_passed: bool
    system1_score: float
    security_safe: bool
    final_verdict: str # "ACCEPT", "REJECT", "ABSTAIN"
    rejection_reason: Optional[str] = None
