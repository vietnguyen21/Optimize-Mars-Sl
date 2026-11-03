"""
6-Field Column Card Extraction and Formatting for LitE-SQL Schema Grounding.
Extracts schema metadata, constraints, descriptions, and representative sample values
from SQLite databases or schema dictionaries.
"""

import sqlite3
import re
import os
import json
from typing import List, Dict, Optional, Any, Union
from src.common.schemas import ColumnCard


class ColumnCardExtractor:
    """Extracts 6-field ColumnCard models from SQLite databases and schema metadata."""

    def __init__(self, sample_limit: int = 3, max_sample_len: int = 64):
        self.sample_limit = sample_limit
        self.max_sample_len = max_sample_len

    def extract_from_sqlite(
        self,
        db_path: str,
        data_dict: Optional[Dict[str, Any]] = None,
        table_descriptions: Optional[Dict[str, str]] = None,
    ) -> List[ColumnCard]:
        """
        Extracts complete 6-field ColumnCard objects from a live SQLite database.

        Args:
            db_path: Path to the SQLite database file.
            data_dict: Optional dictionary providing column descriptions.
                       Format: {table_name: {column_name: "description"}}
                               or {"table.column": "description"}.
            table_descriptions: Optional dictionary providing table-level descriptions.

        Returns:
            List of populated ColumnCard instances.
        """
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"SQLite database not found at: {db_path}")

        cards: List[ColumnCard] = []
        conn = sqlite3.connect(f"file:{os.path.abspath(db_path)}?mode=ro", uri=True)
        cursor = conn.cursor()

        try:
            # 1. Fetch user-defined tables
            cursor.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            )
            tables = cursor.fetchall()

            for table_name, create_sql in tables:
                # 2. Extract column metadata via PRAGMA table_info
                cursor.execute(f'PRAGMA table_info("{table_name}");')
                cols = cursor.fetchall()
                # col schema: (cid, name, type, notnull, dflt_value, pk)

                # 3. Extract foreign key metadata via PRAGMA foreign_key_list
                cursor.execute(f'PRAGMA foreign_key_list("{table_name}");')
                fks = cursor.fetchall()
                # fk schema: (id, seq, table, from, to, on_update, on_delete, match)
                fk_map: Dict[str, str] = {}
                for fk in fks:
                    from_col = fk[3]
                    to_table = fk[2]
                    to_col = fk[4]
                    fk_map[from_col] = f"FOREIGN KEY REFERENCES {to_table}({to_col})"

                for col in cols:
                    col_name = col[1]
                    col_type = col[2].upper() if col[2] else "TEXT"
                    not_null = bool(col[3])
                    dflt_val = col[4]
                    is_pk = bool(col[5])

                    constraints: List[str] = []
                    if is_pk:
                        constraints.append("PRIMARY KEY")
                    if not_null:
                        constraints.append("NOT NULL")
                    if dflt_val is not None:
                        constraints.append(f"DEFAULT {dflt_val}")
                    if col_name in fk_map:
                        constraints.append(fk_map[col_name])

                    # 4. Extract sample values directly from table
                    sample_values: List[str] = []
                    try:
                        cursor.execute(
                            f'SELECT DISTINCT "{col_name}" FROM "{table_name}" '
                            f'WHERE "{col_name}" IS NOT NULL LIMIT {self.sample_limit};'
                        )
                        raw_samples = cursor.fetchall()
                        for (val,) in raw_samples:
                            val_str = str(val).strip()
                            if len(val_str) > self.max_sample_len:
                                val_str = val_str[: self.max_sample_len] + "..."
                            sample_values.append(val_str)
                    except Exception:
                        sample_values = []

                    # 5. Extract or infer column description
                    description = self._resolve_column_description(
                        table_name=table_name,
                        column_name=col_name,
                        data_type=col_type,
                        constraints=constraints,
                        sample_values=sample_values,
                        data_dict=data_dict,
                    )

                    card = ColumnCard(
                        table_name=table_name,
                        column_name=col_name,
                        data_type=col_type,
                        constraints=constraints,
                        column_description=description,
                        sample_values=sample_values,
                    )
                    cards.append(card)

        finally:
            conn.close()

        return cards

    def extract_from_schema_dict(self, schema_dict: Dict[str, Any]) -> List[ColumnCard]:
        """
        Extracts ColumnCard objects from a structured schema dictionary.
        Supports standard formats:
        {
            "tables": {
                "schools": {
                    "columns": {
                        "CDSCode": {
                            "type": "TEXT",
                            "constraints": ["PRIMARY KEY"],
                            "description": "California school code",
                            "sample_values": ["01611760130062"]
                        }
                    }
                }
            }
        }
        or list of table dictionaries.
        """
        cards: List[ColumnCard] = []

        tables_data = schema_dict.get("tables", schema_dict)
        if isinstance(tables_data, list):
            # List of table objects
            for tbl in tables_data:
                table_name = tbl.get("table_name", tbl.get("name", "table"))
                for col in tbl.get("columns", []):
                    if isinstance(col, dict):
                        cards.append(
                            ColumnCard(
                                table_name=table_name,
                                column_name=col.get("name", col.get("column_name", "")),
                                data_type=col.get("type", col.get("data_type", "TEXT")),
                                constraints=col.get("constraints", []),
                                column_description=col.get("description", col.get("column_description", "")),
                                sample_values=col.get("sample_values", []),
                            )
                        )
        elif isinstance(tables_data, dict):
            for table_name, table_info in tables_data.items():
                cols_data = table_info.get("columns", table_info) if isinstance(table_info, dict) else {}
                if isinstance(cols_data, dict):
                    for col_name, col_info in cols_data.items():
                        if isinstance(col_info, dict):
                            cards.append(
                                ColumnCard(
                                    table_name=table_name,
                                    column_name=col_name,
                                    data_type=col_info.get("type", col_info.get("data_type", "TEXT")),
                                    constraints=col_info.get("constraints", []),
                                    column_description=col_info.get("description", col_info.get("column_description", "")),
                                    sample_values=col_info.get("sample_values", []),
                                )
                            )
                        elif isinstance(col_info, str):
                            cards.append(
                                ColumnCard(
                                    table_name=table_name,
                                    column_name=col_name,
                                    data_type=col_info,
                                    constraints=[],
                                    column_description=f"{col_name} attribute in {table_name}",
                                    sample_values=[],
                                )
                            )

        return cards

    def _resolve_column_description(
        self,
        table_name: str,
        column_name: str,
        data_type: str,
        constraints: List[str],
        sample_values: List[str],
        data_dict: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Resolves description from data dictionary or synthesizes an informative description."""
        if data_dict:
            # Check table.column format
            composite_key = f"{table_name}.{column_name}"
            if composite_key in data_dict:
                desc = data_dict[composite_key]
                if isinstance(desc, str) and desc.strip():
                    return desc.strip()
                if isinstance(desc, dict) and "description" in desc:
                    return str(desc["description"]).strip()

            # Check nested table -> column format
            if table_name in data_dict and isinstance(data_dict[table_name], dict):
                tbl_cols = data_dict[table_name].get("columns", data_dict[table_name])
                if column_name in tbl_cols:
                    col_info = tbl_cols[column_name]
                    if isinstance(col_info, str) and col_info.strip():
                        return col_info.strip()
                    if isinstance(col_info, dict) and "description" in col_info:
                        return str(col_info["description"]).strip()

        # Natural heuristic description synthesis
        # e.g., CDSCode -> School unique code
        norm_col = column_name.lower().replace("_", " ")
        if "id" in norm_col or "pk" in norm_col or "code" in norm_col:
            desc = f"Unique identifier {column_name} for {table_name}"
        elif "date" in norm_col or "time" in norm_col or "year" in norm_col:
            desc = f"Temporal/date attribute {column_name} in {table_name}"
        elif "name" in norm_col or "title" in norm_col:
            desc = f"Name or title attribute {column_name} in {table_name}"
        elif "amount" in norm_col or "price" in norm_col or "balance" in norm_col or "score" in norm_col:
            desc = f"Numeric measure or metric {column_name} in {table_name}"
        elif "count" in norm_col or "num" in norm_col:
            desc = f"Count/quantity {column_name} in {table_name}"
        else:
            desc = f"Column {column_name} of type {data_type} in table {table_name}"

        return desc

    @staticmethod
    def save_cards_to_jsonl(cards: List[ColumnCard], output_path: str) -> None:
        """Writes ColumnCard objects to a JSONL file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for card in cards:
                f.write(json.dumps(card.model_dump(), ensure_ascii=False) + "\n")

    @staticmethod
    def load_cards_from_jsonl(input_path: str) -> List[ColumnCard]:
        """Loads ColumnCard objects from a JSONL file."""
        cards: List[ColumnCard] = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    cards.append(ColumnCard(**data))
        return cards


# Convenient top-level functions
def extract_column_cards_from_sqlite(
    db_path: str,
    data_dict: Optional[Dict[str, Any]] = None,
    num_samples: int = 3,
    db_id: Optional[str] = None,
    **kwargs: Any
) -> List[ColumnCard]:
    extractor = ColumnCardExtractor(sample_limit=num_samples)
    return extractor.extract_from_sqlite(db_path=db_path, data_dict=data_dict)


def extract_column_cards_from_schema_dict(schema_dict: Dict[str, Any]) -> List[ColumnCard]:
    extractor = ColumnCardExtractor()
    return extractor.extract_from_schema_dict(schema_dict=schema_dict)
