"""
Node-Weighted Steiner Tree (NWST) Approximation for Text-to-SQL Schema Grounding.
Identifies minimal connecting subgraphs across database tables given query terminal nodes.
Guarantees 100% preservation of intermediate bridge tables necessary for multi-hop JOINs.
"""

import os
import sqlite3
from typing import List, Dict, Set, Tuple, Optional, Any
from src.common.schemas import CompactSubSchema, FKEdge
from src.grounding.graph_builder import SchemaGraph


class DisjointSet:
    """Disjoint Set Union (DSU) / Union-Find data structure for Kruskal's MST."""

    def __init__(self, elements: List[str]):
        self.parent = {el: el for el in elements}
        self.rank = {el: 0 for el in elements}

    def find(self, item: str) -> str:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, set1: str, set2: str) -> bool:
        root1 = self.find(set1)
        root2 = self.find(set2)
        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            else:
                self.parent[root1] = root2
                if self.rank[root1] == self.rank[root2]:
                    self.rank[root2] += 1
            return True
        return False


class NodeWeightedSteinerTree:
    """
    Node-Weighted Steiner Tree (NWST) solver using the Kou-Markowsky-Berman (KMB)
    metric closure and MST approximation with node-weight penalties.
    """

    def __init__(self, schema_graph: SchemaGraph):
        self.graph = schema_graph

    def find_steiner_tree(
        self, terminal_tables: List[str]
    ) -> Tuple[List[str], List[str], List[Tuple[str, str, str, str]], bool]:
        """
        Finds the minimal connecting Steiner tree over the given terminal tables.

        Args:
            terminal_tables: List of table names directly referenced by question or retriever.

        Returns:
            Tuple of:
            - selected_tables (terminals + bridge tables)
            - bridge_tables_added (intermediate tables not in original terminal set)
            - foreign_key_edges (list of (from_tbl, from_col, to_tbl, to_col))
            - connectivity_verified (True if all terminals connected in one tree)
        """
        # Deduplicate and filter terminals existing in graph
        terminals = list(dict.fromkeys([t for t in terminal_tables if self.graph.has_node(t)]))

        if not terminals:
            return [], [], [], True

        if len(terminals) == 1:
            return terminals, [], [], True

        # Group terminals by connected components
        components = self.graph.get_connected_components()
        terminals_by_comp: List[List[str]] = []
        for comp in components:
            comp_terms = [t for t in terminals if t in comp]
            if comp_terms:
                terminals_by_comp.append(comp_terms)

        all_selected_tables: Set[str] = set()
        all_bridge_tables: Set[str] = set()
        all_fk_edges: Set[Tuple[str, str, str, str]] = set()
        connectivity_verified = (len(terminals_by_comp) <= 1)

        # Solve Steiner Tree for each connected component
        for comp_terms in terminals_by_comp:
            sel_nodes, bridges, fk_edges = self._solve_component_steiner_tree(comp_terms)
            all_selected_tables.update(sel_nodes)
            all_bridge_tables.update(bridges)
            all_fk_edges.update(fk_edges)

        return (
            sorted(list(all_selected_tables)),
            sorted(list(all_bridge_tables)),
            sorted(list(all_fk_edges)),
            connectivity_verified,
        )

    def _solve_component_steiner_tree(
        self, terminals: List[str]
    ) -> Tuple[Set[str], Set[str], Set[Tuple[str, str, str, str]]]:
        """
        Solves Steiner Tree for a connected set of terminals using metric closure.
        """
        if len(terminals) == 1:
            return set(terminals), set(), set()

        # Step 1: Compute metric closure between all terminal pairs
        metric_edges: List[Tuple[float, str, str, List[str]]] = []
        for i in range(len(terminals)):
            for j in range(i + 1, len(terminals)):
                u, v = terminals[i], terminals[j]
                path = self.graph.get_shortest_path(u, v)
                if path is not None:
                    # Calculate path cost (edges + intermediate nodes)
                    cost = 0.0
                    for k in range(len(path) - 1):
                        p1, p2 = path[k], path[k + 1]
                        e_data = self.graph.get_edge_data(p1, p2, default={})
                        cost += e_data.get("weight", 1.0)
                    for k in range(1, len(path) - 1):
                        inter_node = path[k]
                        n_data = self.graph.nodes.get(inter_node, {})
                        cost += n_data.get("weight", 1.0)

                    metric_edges.append((cost, u, v, path))

        # Step 2: Minimum Spanning Tree on metric closure (Kruskal)
        metric_edges.sort(key=lambda x: x[0])
        dsu = DisjointSet(terminals)
        mst_paths: List[List[str]] = []
        edges_count = 0

        for cost, u, v, path in metric_edges:
            if dsu.union(u, v):
                mst_paths.append(path)
                edges_count += 1
                if edges_count == len(terminals) - 1:
                    break

        # Step 3: Expand MST edges back into graph paths
        tree_nodes: Set[str] = set()
        tree_edges: Set[Tuple[str, str]] = set()

        for path in mst_paths:
            for node in path:
                tree_nodes.add(node)
            for k in range(len(path) - 1):
                p1, p2 = path[k], path[k + 1]
                edge_pair = (min(p1, p2), max(p1, p2))
                tree_edges.add(edge_pair)

        # Step 4: Prune non-terminal leaves
        # Iteratively prune any node with degree == 1 that is NOT in terminals
        terminals_set = set(terminals)
        changed = True
        while changed:
            changed = False
            adj: Dict[str, Set[str]] = {n: set() for n in tree_nodes}
            for u, v in tree_edges:
                adj[u].add(v)
                adj[v].add(u)

            for node in list(tree_nodes):
                if node not in terminals_set and len(adj[node]) <= 1:
                    # Prune this leaf
                    tree_nodes.remove(node)
                    tree_edges = {e for e in tree_edges if node not in e}
                    changed = True

        # Step 5: Identify bridge tables and collect foreign key edges
        bridge_tables = tree_nodes - terminals_set
        fk_edges_result: Set[Tuple[str, str, str, str]] = set()

        for u, v in tree_edges:
            edge_data = self.graph.get_edge_data(u, v, default={})
            fk_list = edge_data.get("fk_list", [])
            if fk_list:
                for fk in fk_list:
                    fk_edges_result.add(tuple(fk))
            else:
                # Check stored foreign keys
                for fk in self.graph.foreign_keys:
                    if (fk.from_table == u and fk.to_table == v) or (fk.from_table == v and fk.to_table == u):
                        fk_edges_result.add(fk.as_tuple())

        return tree_nodes, bridge_tables, fk_edges_result

    def build_compact_subschema(
        self,
        db_id: str,
        selected_columns: Dict[str, List[str]],
        db_path: Optional[str] = None,
    ) -> CompactSubSchema:
        """
        Builds a complete, validated CompactSubSchema object.
        Preserves all terminal tables, bridge tables, and required join keys.
        """
        terminal_tables = list(selected_columns.keys())
        selected_tables, bridge_tables, fk_edges, verified = self.find_steiner_tree(terminal_tables)

        # For bridge tables, populate columns with primary keys and foreign keys for joining
        final_selected_columns: Dict[str, List[str]] = {}
        for tbl in selected_tables:
            cols = list(selected_columns.get(tbl, []))
            # Include PK and FK columns so the SQL generator can join properly
            node_data = self.graph.nodes.get(tbl, {})
            pks = node_data.get("primary_keys", [])
            for pk in pks:
                if pk not in cols:
                    cols.append(pk)

            # Add FK columns relevant to the selected tree
            for from_t, from_c, to_t, to_c in fk_edges:
                if from_t == tbl and from_c not in cols:
                    cols.append(from_c)
                if to_t == tbl and to_c not in cols:
                    cols.append(to_c)

            final_selected_columns[tbl] = cols

        # Generate DDL statements for selected tables
        ddl_statements = self._generate_subschema_ddl(
            selected_tables=selected_tables,
            selected_columns=final_selected_columns,
            fk_edges=fk_edges,
            db_path=db_path,
        )

        return CompactSubSchema(
            db_id=db_id,
            selected_tables=selected_tables,
            selected_columns=final_selected_columns,
            foreign_keys=fk_edges,
            ddl_statements=ddl_statements,
            bridge_tables_added=bridge_tables,
            graph_connectivity_verified=verified,
        )

    def _generate_subschema_ddl(
        self,
        selected_tables: List[str],
        selected_columns: Dict[str, List[str]],
        fk_edges: List[Tuple[str, str, str, str]],
        db_path: Optional[str] = None,
    ) -> str:
        """Generates clean, compact DDL statements for the selected sub-schema."""
        ddl_parts: List[str] = []

        # If live SQLite db exists, extract authentic CREATE TABLE statements
        if db_path and os.path.exists(db_path):
            try:
                conn = sqlite3.connect(f"file:{os.path.abspath(db_path)}?mode=ro", uri=True)
                cursor = conn.cursor()
                for tbl in selected_tables:
                    cursor.execute(
                        f"SELECT sql FROM sqlite_master WHERE type='table' AND name = ?;", (tbl,)
                    )
                    row = cursor.fetchone()
                    if row and row[0]:
                        ddl_parts.append(row[0].strip() + ";")
                conn.close()
            except Exception:
                pass

        # Fallback to generated DDL if db_path not provided or failed
        if not ddl_parts:
            for tbl in selected_tables:
                cols = selected_columns.get(tbl, [])
                node_data = self.graph.nodes.get(tbl, {})
                pks = node_data.get("primary_keys", [])

                col_defs = []
                for c in cols:
                    if c in pks:
                        col_defs.append(f"  {c} INTEGER PRIMARY KEY")
                    else:
                        col_defs.append(f"  {c} TEXT")

                # Add FK constraints
                for from_t, from_c, to_t, to_c in fk_edges:
                    if from_t == tbl and to_t in selected_tables:
                        col_defs.append(f"  FOREIGN KEY ({from_c}) REFERENCES {to_t} ({to_c})")

                ddl_body = ",\n".join(col_defs)
                ddl_parts.append(f"CREATE TABLE {tbl} (\n{ddl_body}\n);")

        return "\n\n".join(ddl_parts)


def extract_compact_subschema(
    schema_graph: SchemaGraph,
    selected_columns: Dict[str, List[str]],
    db_id: str,
    db_path: Optional[str] = None,
) -> CompactSubSchema:
    """Convenience function to extract a CompactSubSchema using NodeWeightedSteinerTree."""
    solver = NodeWeightedSteinerTree(schema_graph)
    return solver.build_compact_subschema(
        db_id=db_id,
        selected_columns=selected_columns,
        db_path=db_path,
    )
