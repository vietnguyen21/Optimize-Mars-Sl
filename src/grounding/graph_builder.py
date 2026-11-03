"""
Schema Graph Builder for Text-to-SQL Grounding.
Constructs directed and undirected PK-FK schema graphs G = (V, E)
from SQLite databases, DDL statements, or schema dictionaries using NetworkX.
Provides robust offline fallback when networkx is not installed.
"""

import os
import sqlite3
import pickle
import heapq
from typing import List, Dict, Set, Tuple, Optional, Any, Iterator
from src.common.schemas import TableGraphNode, FKEdge

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class PurePythonGraph:
    """Lightweight pure-Python undirected graph with NetworkX-compatible API."""

    def __init__(self, **attr):
        self.graph = dict(attr)
        self._node: Dict[str, Dict[str, Any]] = {}
        self._adj: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def add_node(self, node_for_adding: str, **attr) -> None:
        if node_for_adding not in self._node:
            self._node[node_for_adding] = {}
            self._adj[node_for_adding] = {}
        self._node[node_for_adding].update(attr)

    def add_edge(self, u_of_edge: str, v_of_edge: str, **attr) -> None:
        u, v = u_of_edge, v_of_edge
        self.add_node(u)
        self.add_node(v)
        datadict = self._adj[u].get(v, {})
        datadict.update(attr)
        self._adj[u][v] = datadict
        self._adj[v][u] = datadict

    @property
    def nodes(self) -> Dict[str, Dict[str, Any]]:
        return self._node

    @property
    def edges(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        edge_dict = {}
        seen = set()
        for u, neighbors in self._adj.items():
            for v, attrs in neighbors.items():
                if (v, u) not in seen:
                    seen.add((u, v))
                    edge_dict[(u, v)] = attrs
        return edge_dict

    def has_node(self, n: str) -> bool:
        return n in self._node

    def has_edge(self, u: str, v: str) -> bool:
        return u in self._adj and v in self._adj[u]

    def neighbors(self, n: str) -> Iterator[str]:
        if n not in self._adj:
            return iter([])
        return iter(self._adj[n].keys())

    def degree(self, n: str) -> int:
        return len(self._adj.get(n, {}))

    def get_edge_data(self, u: str, v: str, default: Any = None) -> Any:
        try:
            return self._adj[u][v]
        except KeyError:
            return default

    def subgraph(self, nodes: List[str]) -> "PurePythonGraph":
        sub = PurePythonGraph()
        node_set = set(nodes)
        for n in node_set:
            if n in self._node:
                sub.add_node(n, **self._node[n])
        for u in node_set:
            if u in self._adj:
                for v, attr in self._adj[u].items():
                    if v in node_set:
                        sub.add_edge(u, v, **attr)
        return sub

    def copy(self) -> "PurePythonGraph":
        cp = PurePythonGraph(**self.graph)
        for n, attr in self._node.items():
            cp.add_node(n, **dict(attr))
        for (u, v), attr in self.edges.items():
            cp.add_edge(u, v, **dict(attr))
        return cp

    def get_nodes(self) -> List[str]:
        """Returns list of node names."""
        return list(self._node.keys())

    def get_edges(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Returns list of (u, v, data) edge tuples."""
        return [(u, v, data) for (u, v), data in self.edges.items()]

    def __iter__(self):
        return iter(self._node)

    def __len__(self):
        return len(self._node)

    def __contains__(self, n):
        return n in self._node


if HAS_NETWORKX:
    BaseGraphClass = nx.Graph
else:
    BaseGraphClass = PurePythonGraph


class SchemaGraph(BaseGraphClass):
    """
    SchemaGraph representing database tables as nodes and PK-FK relations as edges.
    Extends NetworkX Graph when available, with pure-Python fallback.
    """

    def __init__(self, db_id: str = "", **attr):
        super().__init__(**attr)
        self.db_id = db_id
        self.foreign_keys: List[FKEdge] = []

    def get_nodes(self) -> List[str]:
        """Returns list of node names."""
        if hasattr(self, "_node") and isinstance(self._node, dict):
            return list(self._node.keys())
        if hasattr(self, "nodes"):
            if isinstance(self.nodes, dict):
                return list(self.nodes.keys())
            if callable(self.nodes):
                return list(self.nodes())
            return list(self.nodes)
        return []

    def get_edges(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Returns list of (u, v, data) edge tuples."""
        if hasattr(self, "edges"):
            if isinstance(self.edges, dict):
                return [(u, v, data) for (u, v), data in self.edges.items()]
            if callable(self.edges):
                return [(u, v, data) for u, v, data in self.edges(data=True)]
            try:
                return [(u, v, data) for u, v, data in self.edges(data=True)]
            except TypeError:
                return [(u, v, self.get_edge_data(u, v, {})) for u, v in self.edges]
        return []

    def add_table_node(
        self,
        table_name: str,
        columns: Optional[List[str]] = None,
        primary_keys: Optional[List[str]] = None,
        weight: float = 1.0,
    ) -> None:
        """Adds a table node with column attributes and weight."""
        cols = columns or []
        pks = primary_keys or []
        self.add_node(
            table_name,
            columns=cols,
            primary_keys=pks,
            weight=weight,
            table_name=table_name,
        )

    def add_fk_edge(
        self,
        from_table: str,
        from_col: str,
        to_table: str,
        to_col: str,
        weight: float = 1.0,
    ) -> None:
        """Adds a foreign key relation edge between two tables."""
        fk = FKEdge(
            from_table=from_table,
            from_column=from_col,
            to_table=to_table,
            to_column=to_col,
            weight=weight,
        )
        self.foreign_keys.append(fk)

        existing_edge = self.get_edge_data(from_table, to_table, default={})
        fk_list = list(existing_edge.get("fk_list", []))
        fk_list.append((from_table, from_col, to_table, to_col))

        # Update or add edge
        current_weight = existing_edge.get("weight", weight)
        min_weight = min(current_weight, weight)

        self.add_edge(
            from_table,
            to_table,
            weight=min_weight,
            fk_list=fk_list,
            relation="pk_fk",
        )

    def get_shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Finds shortest path between two table nodes considering edge weights."""
        if source == target:
            return [source]
        if not self.has_node(source) or not self.has_node(target):
            return None

        # Dijkstra algorithm
        dist: Dict[str, float] = {source: 0.0}
        prev: Dict[str, Optional[str]] = {source: None}
        pq = [(0.0, source)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, float("inf")):
                continue
            if u == target:
                break

            for v in self.neighbors(u):
                edge_data = self.get_edge_data(u, v, default={})
                # Edge weight + node traversal weight
                v_node_data = self.nodes.get(v, {})
                node_weight = v_node_data.get("weight", 1.0)
                edge_weight = edge_data.get("weight", 1.0)
                cost = edge_weight + node_weight

                new_dist = d + cost
                if new_dist < dist.get(v, float("inf")):
                    dist[v] = new_dist
                    prev[v] = u
                    heapq.heappush(pq, (new_dist, v))

        if target not in prev:
            return None

        path = []
        curr: Optional[str] = target
        while curr is not None:
            path.append(curr)
            curr = prev[curr]
        path.reverse()
        return path

    def get_connected_components(self) -> List[Set[str]]:
        """Returns connected components of the schema graph."""
        visited: Set[str] = set()
        components: List[Set[str]] = []

        for node in self.nodes:
            if node not in visited:
                comp: Set[str] = set()
                queue = [node]
                visited.add(node)
                while queue:
                    curr = queue.pop(0)
                    comp.add(curr)
                    for neighbor in self.neighbors(curr):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                components.append(comp)

        return components

    def to_networkx(self):
        """Returns standard NetworkX Graph representation."""
        if HAS_NETWORKX and isinstance(self, nx.Graph):
            return self
        if not HAS_NETWORKX:
            return self
        g = nx.Graph(db_id=self.db_id)
        for n, attr in self.nodes.items():
            g.add_node(n, **attr)
        for (u, v), attr in self.edges.items():
            g.add_edge(u, v, **attr)
        return g


class SchemaGraphBuilder:
    """Builds SchemaGraph instances from SQLite databases, DDLs, and metadata."""

    @classmethod
    def build_from_sqlite(
        cls,
        db_path: str,
        db_id: Optional[str] = None,
        infer_implicit_fks: bool = True,
    ) -> SchemaGraph:
        """
        Builds a SchemaGraph from an SQLite database.
        Extracts table columns, primary keys, and foreign key relations.
        Optionally infers implicit foreign keys if no explicit constraints are defined.
        """
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file not found: {db_path}")

        resolved_db_id = db_id or os.path.splitext(os.path.basename(db_path))[0]
        graph = SchemaGraph(db_id=resolved_db_id)

        conn = sqlite3.connect(f"file:{os.path.abspath(db_path)}?mode=ro", uri=True)
        cursor = conn.cursor()

        table_columns: Dict[str, List[str]] = {}
        table_pks: Dict[str, List[str]] = {}
        explicit_fks_found = 0

        try:
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            )
            tables = [row[0] for row in cursor.fetchall()]

            # 1. Add all table nodes
            for tbl in tables:
                cursor.execute(f'PRAGMA table_info("{tbl}");')
                cols_info = cursor.fetchall()

                cols = [c[1] for c in cols_info]
                pks = [c[1] for c in cols_info if c[5] > 0]

                table_columns[tbl] = cols
                table_pks[tbl] = pks
                graph.add_table_node(table_name=tbl, columns=cols, primary_keys=pks, weight=1.0)

            # 2. Extract explicit foreign keys
            for tbl in tables:
                cursor.execute(f'PRAGMA foreign_key_list("{tbl}");')
                fks = cursor.fetchall()
                for fk in fks:
                    to_table = fk[2]
                    from_col = fk[3]
                    to_col = fk[4]
                    if to_table in tables:
                        graph.add_fk_edge(
                            from_table=tbl,
                            from_col=from_col,
                            to_table=to_table,
                            to_col=to_col,
                            weight=1.0,
                        )
                        explicit_fks_found += 1

            # 3. Handle Edge Case E01: Infer implicit FKs if needed or if requested
            if infer_implicit_fks and (explicit_fks_found == 0 or len(graph.edges) < len(tables) - 1):
                cls._infer_implicit_foreign_keys(graph, table_columns, table_pks)

        finally:
            conn.close()

        return graph

    @classmethod
    def _infer_implicit_foreign_keys(
        cls,
        graph: SchemaGraph,
        table_columns: Dict[str, List[str]],
        table_pks: Dict[str, List[str]],
    ) -> None:
        """
        Infers implicit foreign keys between tables based on naming conventions.
        Handles edge cases where DDL lacks explicit FOREIGN KEY constraints.
        Applies higher penalty weight (e.g. 2.0) to implicit relations.
        """
        tables = list(table_columns.keys())
        for i in range(len(tables)):
            for j in range(i + 1, len(tables)):
                t1, t2 = tables[i], tables[j]
                if graph.has_edge(t1, t2):
                    continue

                cols1 = table_columns[t1]
                cols2 = table_columns[t2]
                pks1 = set(table_pks.get(t1, []))
                pks2 = set(table_pks.get(t2, []))

                for c1 in cols1:
                    c1_clean = c1.lower().replace("_", "").replace("-", "")
                    for c2 in cols2:
                        c2_clean = c2.lower().replace("_", "").replace("-", "")

                        is_candidate = False
                        # Exact name match on key-like column
                        if (c1_clean == c2_clean) and (c1 in pks1 or c2 in pks2 or "id" in c1_clean or "code" in c1_clean):
                            is_candidate = True
                        # Table name prefix pattern (e.g. school_id in frpm matching id in schools)
                        elif c1_clean == f"{t2.lower().rstrip('s')}id" or c2_clean == f"{t1.lower().rstrip('s')}id":
                            is_candidate = True

                        if is_candidate:
                            # Add edge with higher penalty weight
                            graph.add_fk_edge(
                                from_table=t1,
                                from_col=c1,
                                to_table=t2,
                                to_col=c2,
                                weight=2.0,
                            )
                            break

    @classmethod
    def build_from_schema_dict(cls, schema_dict: Dict[str, Any], db_id: str = "") -> SchemaGraph:
        """Builds SchemaGraph from dictionary containing tables, columns, and foreign keys."""
        resolved_db_id = db_id or schema_dict.get("db_id", "schema_graph")
        graph = SchemaGraph(db_id=resolved_db_id)

        tables = schema_dict.get("tables", {})
        if isinstance(tables, dict):
            for tbl_name, tbl_info in tables.items():
                cols = tbl_info.get("columns", []) if isinstance(tbl_info, dict) else []
                pks = tbl_info.get("primary_keys", []) if isinstance(tbl_info, dict) else []
                graph.add_table_node(table_name=tbl_name, columns=cols, primary_keys=pks)

        # Process foreign keys
        fks = schema_dict.get("foreign_keys", [])
        for fk in fks:
            if isinstance(fk, (list, tuple)) and len(fk) >= 4:
                graph.add_fk_edge(from_table=fk[0], from_col=fk[1], to_table=fk[2], to_col=fk[3])
            elif isinstance(fk, dict):
                graph.add_fk_edge(
                    from_table=fk["from_table"],
                    from_col=fk["from_column"],
                    to_table=fk["to_table"],
                    to_col=fk["to_column"],
                    weight=fk.get("weight", 1.0),
                )

        return graph

    @staticmethod
    def save_graph_pickle(graph: SchemaGraph, output_path: str) -> None:
        """Serializes SchemaGraph to a pickle file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        # Store serializable dict representation
        data = {
            "db_id": graph.db_id,
            "nodes": dict(graph.nodes),
            "edges": dict(graph.edges),
            "foreign_keys": [fk.model_dump() for fk in graph.foreign_keys],
        }
        with open(output_path, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def load_graph_pickle(input_path: str) -> SchemaGraph:
        """Loads SchemaGraph from a pickle file."""
        with open(input_path, "rb") as f:
            data = pickle.load(f)

        graph = SchemaGraph(db_id=data.get("db_id", ""))
        for n, attr in data.get("nodes", {}).items():
            graph.add_node(n, **attr)
        for (u, v), attr in data.get("edges", {}).items():
            graph.add_edge(u, v, **attr)
        for fk_data in data.get("foreign_keys", []):
            graph.foreign_keys.append(FKEdge(**fk_data))
        return graph


def build_graph_from_sqlite(db_path: str, db_id: Optional[str] = None) -> SchemaGraph:
    return SchemaGraphBuilder.build_from_sqlite(db_path=db_path, db_id=db_id)
