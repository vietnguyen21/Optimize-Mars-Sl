"""
Empirical Stress Test Suite for Milestone M1 (Data Pipeline & Schema Grounding).
Adversarially probes:
1. Synthetic Graph Topologies (linear chains, cycles, diamonds, disconnected components, boundary schemas).
2. Mini BIRD SQLite Database & Query Correctness (all 24 queries across both databases).
3. ColumnCardExtractor & LitESQLEmbeddingRetriever robustness under hostile inputs.
"""

import os
import sys
import json
import sqlite3
import unittest
import tempfile
import time
from typing import List, Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.common.schemas import ColumnCard, CompactSubSchema, TableGraphNode, FKEdge
from src.grounding.column_card import (
    ColumnCardExtractor,
    extract_column_cards_from_sqlite,
    extract_column_cards_from_schema_dict,
)
from src.grounding.graph_builder import (
    SchemaGraph,
    SchemaGraphBuilder,
    build_graph_from_sqlite,
    PurePythonGraph,
)
from src.grounding.steiner_tree import NodeWeightedSteinerTree, extract_compact_subschema
from src.grounding.embedding_retriever import LitESQLEmbeddingRetriever


class TestSteinerTreeComplexGraphs(unittest.TestCase):
    """Adversarial stress testing on graph algorithms and topologies."""

    def test_linear_chain_bridge_recovery(self):
        """
        5-node linear chain: A - B - C - D - E.
        Terminals: (A, E).
        Verification: All intermediate bridge tables (B, C, D) must be recovered (100% recovery).
        """
        graph = SchemaGraph(db_id="linear_chain_db")
        nodes = ["tbl_a", "tbl_b", "tbl_c", "tbl_d", "tbl_e"]
        for n in nodes:
            graph.add_table_node(n, columns=[f"{n}_id", "data"], primary_keys=[f"{n}_id"])

        # Add edges along the chain
        edges = [
            ("tbl_a", "tbl_b"),
            ("tbl_b", "tbl_c"),
            ("tbl_c", "tbl_d"),
            ("tbl_d", "tbl_e"),
        ]
        for u, v in edges:
            graph.add_fk_edge(
                from_table=u,
                from_col=f"{v}_id",
                to_table=v,
                to_col=f"{v}_id",
                weight=1.0,
            )

        solver = NodeWeightedSteinerTree(graph)

        # 1. Terminals (A, E) -> Intermediate: (B, C, D)
        sel, bridges, fk_edges, verified = solver.find_steiner_tree(["tbl_a", "tbl_e"])
        self.assertTrue(verified, "Linear chain should be fully connected")
        self.assertEqual(sel, sorted(nodes), "All 5 nodes must be in selected tables")
        self.assertEqual(bridges, sorted(["tbl_b", "tbl_c", "tbl_d"]), "Bridge tables must be B, C, D")
        self.assertEqual(len(fk_edges), 4, "Should have 4 join edges in linear chain MST")

        # 2. Terminals (B, D) -> Intermediate: (C)
        sel_sub, bridges_sub, _, ver_sub = solver.find_steiner_tree(["tbl_b", "tbl_d"])
        self.assertTrue(ver_sub)
        self.assertEqual(sel_sub, ["tbl_b", "tbl_c", "tbl_d"])
        self.assertEqual(bridges_sub, ["tbl_c"])

        # 3. Terminals (A, C, E) -> Intermediate: (B, D)
        sel_tri, bridges_tri, _, ver_tri = solver.find_steiner_tree(["tbl_a", "tbl_c", "tbl_e"])
        self.assertTrue(ver_tri)
        self.assertEqual(sel_tri, sorted(nodes))
        self.assertEqual(bridges_tri, sorted(["tbl_b", "tbl_d"]))

    def test_cycle_topology_and_weights(self):
        """
        Cycle graph with unequal path weights.
        Triangle: A - B (weight 1), B - C (weight 1), A - C (weight 10).
        Terminals: (A, C).
        Verification: Algorithm must choose low-cost path A-B-C instead of high-cost direct edge A-C.
        Bridge table B must be recovered.
        """
        graph = SchemaGraph(db_id="cycle_db")
        for n in ["A", "B", "C"]:
            graph.add_table_node(n, columns=[f"{n}_id"], primary_keys=[f"{n}_id"], weight=1.0)

        graph.add_fk_edge("A", "b_id", "B", "b_id", weight=1.0)
        graph.add_fk_edge("B", "c_id", "C", "c_id", weight=1.0)
        graph.add_fk_edge("A", "c_id", "C", "c_id", weight=10.0)

        solver = NodeWeightedSteinerTree(graph)
        sel, bridges, fk_edges, verified = solver.find_steiner_tree(["A", "C"])

        self.assertTrue(verified)
        self.assertIn("B", sel, "Node B must be included via the lower cost path")
        self.assertIn("B", bridges, "Node B must be identified as bridge table")
        self.assertEqual(sorted(sel), ["A", "B", "C"])

    def test_multipath_diamond_topology(self):
        r"""
        Diamond graph:
             B (cheap: weight 1)
           /   \
          A     D
           \   /
             C (expensive: weight 5)
        Terminals: (A, D).
        Verification: NWST selects path A - B - D and avoids path A - C - D.
        """
        graph = SchemaGraph(db_id="diamond_db")
        for n in ["A", "B", "C", "D"]:
            graph.add_table_node(n, columns=[f"{n}_id"], primary_keys=[f"{n}_id"], weight=1.0)

        # Cheap path through B
        graph.add_fk_edge("A", "b_id", "B", "b_id", weight=1.0)
        graph.add_fk_edge("B", "d_id", "D", "d_id", weight=1.0)

        # Expensive path through C
        graph.add_fk_edge("A", "c_id", "C", "c_id", weight=5.0)
        graph.add_fk_edge("C", "d_id", "D", "d_id", weight=5.0)

        solver = NodeWeightedSteinerTree(graph)
        sel, bridges, fk_edges, verified = solver.find_steiner_tree(["A", "D"])

        self.assertTrue(verified)
        self.assertIn("B", sel, "B must be selected (cheaper)")
        self.assertNotIn("C", sel, "C must NOT be selected (more expensive)")
        self.assertEqual(bridges, ["B"])

    def test_disconnected_components_graceful_handling(self):
        """
        Graph with two disconnected components:
        Component 1: A - B - C
        Component 2: X - Y
        Terminals: (A, X).
        Verification: Handles gracefully without throwing uncaught exceptions,
        returns connectivity_verified = False, includes both terminals, no fake bridges.
        """
        graph = SchemaGraph(db_id="disconnected_db")
        # Component 1
        for n in ["A", "B", "C"]:
            graph.add_table_node(n, columns=[f"{n}_id"], primary_keys=[f"{n}_id"])
        graph.add_fk_edge("A", "b_id", "B", "b_id")
        graph.add_fk_edge("B", "c_id", "C", "c_id")

        # Component 2
        for n in ["X", "Y"]:
            graph.add_table_node(n, columns=[f"{n}_id"], primary_keys=[f"{n}_id"])
        graph.add_fk_edge("X", "y_id", "Y", "y_id")

        solver = NodeWeightedSteinerTree(graph)

        # Terminals in different components
        sel, bridges, fk_edges, verified = solver.find_steiner_tree(["A", "X"])
        self.assertFalse(verified, "Disconnected terminals must report connectivity_verified=False")
        self.assertEqual(sorted(sel), ["A", "X"])
        self.assertEqual(bridges, [])
        self.assertEqual(fk_edges, [])

        # Terminals (A, C, X) -> Component 1 has terminals A and C (needs bridge B), Component 2 has X
        sel2, bridges2, fk2, verified2 = solver.find_steiner_tree(["A", "C", "X"])
        self.assertFalse(verified2)
        self.assertEqual(sorted(sel2), ["A", "B", "C", "X"])
        self.assertEqual(bridges2, ["B"], "Bridge B in component 1 must be recovered despite X being disconnected")

    def test_boundary_empty_and_single_node_schemas(self):
        """
        Boundary conditions:
        - Completely empty graph
        - Non-existent terminal nodes
        - Single table schema
        - Single terminal queried
        - Graph with tables but zero foreign keys (fully disconnected)
        """
        # 1. Empty graph
        empty_graph = SchemaGraph(db_id="empty_db")
        solver_empty = NodeWeightedSteinerTree(empty_graph)
        sel, bridges, fks, ver = solver_empty.find_steiner_tree([])
        self.assertEqual((sel, bridges, fks, ver), ([], [], [], True))

        # Querying non-existent nodes
        sel_ne, bridges_ne, fks_ne, ver_ne = solver_empty.find_steiner_tree(["non_existent_table"])
        self.assertEqual((sel_ne, bridges_ne, fks_ne, ver_ne), ([], [], [], True))

        # 2. Single table graph
        single_graph = SchemaGraph(db_id="single_db")
        single_graph.add_table_node("solo_table", columns=["id", "val"], primary_keys=["id"])
        solver_single = NodeWeightedSteinerTree(single_graph)
        sel_s, bridges_s, fks_s, ver_s = solver_single.find_steiner_tree(["solo_table"])
        self.assertEqual(sel_s, ["solo_table"])
        self.assertEqual(bridges_s, [])
        self.assertEqual(fks_s, [])
        self.assertTrue(ver_s)

        # 3. Multi-table graph with ZERO foreign keys
        isolated_graph = SchemaGraph(db_id="no_fk_db")
        for t in ["t1", "t2", "t3"]:
            isolated_graph.add_table_node(t, columns=["id"], primary_keys=["id"])
        solver_iso = NodeWeightedSteinerTree(isolated_graph)
        sel_iso, bridges_iso, fks_iso, ver_iso = solver_iso.find_steiner_tree(["t1", "t2"])
        self.assertFalse(ver_iso, "Tables with no FK relations cannot be connected")
        self.assertEqual(sorted(sel_iso), ["t1", "t2"])
        self.assertEqual(bridges_iso, [])
        self.assertEqual(fks_iso, [])

    def test_build_compact_subschema_output_contract(self):
        """Tests that build_compact_subschema generates valid CompactSubSchema contract."""
        graph = SchemaGraph(db_id="test_contract_db")
        graph.add_table_node("customers", columns=["cid", "cname"], primary_keys=["cid"])
        graph.add_table_node("orders", columns=["oid", "cid", "order_date"], primary_keys=["oid"])
        graph.add_table_node("items", columns=["iid", "oid", "price"], primary_keys=["iid"])
        graph.add_fk_edge("orders", "cid", "customers", "cid")
        graph.add_fk_edge("items", "oid", "orders", "oid")

        solver = NodeWeightedSteinerTree(graph)
        subschema = solver.build_compact_subschema(
            db_id="test_contract_db",
            selected_columns={"customers": ["cname"], "items": ["price"]},
        )

        self.assertIsInstance(subschema, CompactSubSchema)
        self.assertTrue(subschema.graph_connectivity_verified)
        self.assertEqual(sorted(subschema.selected_tables), ["customers", "items", "orders"])
        self.assertEqual(subschema.bridge_tables_added, ["orders"])
        self.assertIn("cid", subschema.selected_columns["customers"])
        self.assertIn("oid", subschema.selected_columns["orders"])
        self.assertIn("cid", subschema.selected_columns["orders"])
        self.assertTrue(len(subschema.ddl_statements) > 0)
        self.assertTrue("CREATE TABLE" in subschema.ddl_statements)

    def test_star_topology_central_hub_recovery(self):
        """
        Star topology: Hub table H connected to spokes S1, S2, S3, S4.
        Terminals: (S1, S4).
        Verification: Hub H must be recovered as bridge table.
        Spokes S2 and S3 must NOT be selected.
        """
        graph = SchemaGraph(db_id="star_db")
        graph.add_table_node("hub", columns=["hid"], primary_keys=["hid"])
        for i in range(1, 5):
            spoke = f"spoke_{i}"
            graph.add_table_node(spoke, columns=[f"{spoke}_id", "hid"], primary_keys=[f"{spoke}_id"])
            graph.add_fk_edge(spoke, "hid", "hub", "hid", weight=1.0)

        solver = NodeWeightedSteinerTree(graph)
        sel, bridges, fks, ver = solver.find_steiner_tree(["spoke_1", "spoke_4"])

        self.assertTrue(ver)
        self.assertEqual(sorted(sel), ["hub", "spoke_1", "spoke_4"])
        self.assertEqual(bridges, ["hub"])
        self.assertNotIn("spoke_2", sel)
        self.assertNotIn("spoke_3", sel)

    def test_deep_chain_10_nodes_recovery(self):
        """
        10-node chain: T0 - T1 - T2 - T3 - T4 - T5 - T6 - T7 - T8 - T9.
        Terminals: (T0, T9).
        Verification: 100% recovery of all 8 intermediate bridge tables (T1..T8).
        """
        graph = SchemaGraph(db_id="deep_chain_db")
        for i in range(10):
            graph.add_table_node(f"t_{i}", columns=[f"id_{i}"], primary_keys=[f"id_{i}"])
        for i in range(9):
            graph.add_fk_edge(f"t_{i}", f"id_{i+1}", f"t_{i+1}", f"id_{i+1}", weight=1.0)

        solver = NodeWeightedSteinerTree(graph)
        sel, bridges, fks, ver = solver.find_steiner_tree(["t_0", "t_9"])

        self.assertTrue(ver)
        self.assertEqual(len(sel), 10)
        expected_bridges = [f"t_{i}" for i in range(1, 9)]
        self.assertEqual(sorted(bridges), sorted(expected_bridges))

    def test_multi_fk_same_table_pair(self):
        """
        Two tables with multiple foreign keys between them:
        orders -> addresses on billing_address_id
        orders -> addresses on shipping_address_id
        Verification: Both FK relationships are properly retained.
        """
        graph = SchemaGraph(db_id="multi_fk_db")
        graph.add_table_node("orders", columns=["oid", "billing_id", "shipping_id"], primary_keys=["oid"])
        graph.add_table_node("addresses", columns=["aid", "street"], primary_keys=["aid"])
        graph.add_fk_edge("orders", "billing_id", "addresses", "aid")
        graph.add_fk_edge("orders", "shipping_id", "addresses", "aid")

        solver = NodeWeightedSteinerTree(graph)
        sel, bridges, fks, ver = solver.find_steiner_tree(["orders", "addresses"])

        self.assertTrue(ver)
        self.assertEqual(len(fks), 2)
        from_cols = {fk[1] for fk in fks}
        self.assertIn("billing_id", from_cols)
        self.assertIn("shipping_id", from_cols)

    def test_duplicate_terminals(self):
        """Verify duplicate terminal inputs are cleanly deduplicated without error."""
        graph = SchemaGraph(db_id="dedup_db")
        graph.add_table_node("A", columns=["id"])
        graph.add_table_node("B", columns=["id"])
        graph.add_fk_edge("A", "id", "B", "id")

        solver = NodeWeightedSteinerTree(graph)
        sel, bridges, fks, ver = solver.find_steiner_tree(["A", "A", "B", "B", "A"])
        self.assertEqual(sorted(sel), ["A", "B"])
        self.assertEqual(bridges, [])
        self.assertTrue(ver)


class TestMiniBirdDatabaseQueries(unittest.TestCase):
    """Executes all 24 queries in data/mini/mini_bird.jsonl against SQLite databases."""

    @classmethod
    def setUpClass(cls):
        cls.bird_path = os.path.abspath("data/mini/mini_bird.jsonl")
        cls.synsql_path = os.path.abspath("data/mini/mini_synsql.jsonl")
        cls.fin_db_path = os.path.abspath("data/mini/financial_mini.sqlite")
        cls.sch_db_path = os.path.abspath("data/mini/california_schools.sqlite")

        # Verify files exist
        assert os.path.exists(cls.bird_path), f"Missing {cls.bird_path}"
        assert os.path.exists(cls.synsql_path), f"Missing {cls.synsql_path}"
        assert os.path.exists(cls.fin_db_path), f"Missing {cls.fin_db_path}"
        assert os.path.exists(cls.sch_db_path), f"Missing {cls.sch_db_path}"

        with open(cls.bird_path, "r", encoding="utf-8") as f:
            cls.bird_queries = [json.loads(line) for line in f if line.strip()]

    def test_mini_bird_query_count_and_schema(self):
        """Verify there are exactly 24 queries with required fields."""
        self.assertEqual(len(self.bird_queries), 24, "Must contain exactly 24 BIRD queries")
        required_keys = {"question_id", "db_id", "question", "evidence", "SQL", "difficulty"}

        fin_count = sum(1 for q in self.bird_queries if q["db_id"] == "financial_mini")
        sch_count = sum(1 for q in self.bird_queries if q["db_id"] == "california_schools")

        self.assertEqual(fin_count, 12, "Must have 12 financial_mini queries")
        self.assertEqual(sch_count, 12, "Must have 12 california_schools queries")

        for q in self.bird_queries:
            missing = required_keys - set(q.keys())
            self.assertEqual(len(missing), 0, f"Query {q.get('question_id')} missing keys: {missing}")

    def test_execute_all_24_queries_on_sqlite(self):
        """
        Empirically executes all 24 queries against their respective SQLite databases.
        Verifies:
        1. No syntax or schema errors (OperationalError, DatabaseError).
        2. Queries execute in < 50ms each.
        3. Queries return non-empty result sets.
        4. Valid columns returned.
        """
        fin_conn = sqlite3.connect(f"file:{self.fin_db_path}?mode=ro", uri=True)
        sch_conn = sqlite3.connect(f"file:{self.sch_db_path}?mode=ro", uri=True)

        failures = []
        execution_times = []

        try:
            for q in self.bird_queries:
                qid = q["question_id"]
                db_id = q["db_id"]
                sql = q["SQL"]
                conn = fin_conn if db_id == "financial_mini" else sch_conn

                start_t = time.perf_counter()
                try:
                    cursor = conn.cursor()
                    cursor.execute(sql)
                    rows = cursor.fetchall()
                    elapsed_ms = (time.perf_counter() - start_t) * 1000
                    execution_times.append(elapsed_ms)

                    # Result verification
                    if not rows:
                        failures.append(f"Q{qid} ({db_id}): Query returned EMPTY result set: {sql}")
                    if cursor.description is None or len(cursor.description) == 0:
                        failures.append(f"Q{qid} ({db_id}): No cursor description/columns returned")

                except Exception as e:
                    failures.append(f"Q{qid} ({db_id}) ERROR: {type(e).__name__}: {str(e)} | SQL: {sql}")

        finally:
            fin_conn.close()
            sch_conn.close()

        self.assertEqual(
            len(failures),
            0,
            f"Encountered {len(failures)} query failures out of 24:\n" + "\n".join(failures),
        )
        avg_time = sum(execution_times) / len(execution_times)
        self.assertLess(avg_time, 10.0, f"Average query execution too slow: {avg_time:.2f}ms")

    def test_mini_synsql_reasoning_format_and_execution(self):
        """Verify mini_synsql.jsonl has 20 valid multi-turn thinking examples and executable SQL."""
        with open(self.synsql_path, "r", encoding="utf-8") as f:
            syn_samples = [json.loads(line) for line in f if line.strip()]

        self.assertEqual(len(syn_samples), 20, "Must contain exactly 20 SynSQL samples")
        required_keys = {"id", "db_id", "question", "schema", "think_steps", "answer", "formatted_prompt"}

        fin_conn = sqlite3.connect(f"file:{self.fin_db_path}?mode=ro", uri=True)
        sch_conn = sqlite3.connect(f"file:{self.sch_db_path}?mode=ro", uri=True)

        try:
            for s in syn_samples:
                missing = required_keys - set(s.keys())
                self.assertEqual(len(missing), 0, f"SynSQL sample {s.get('id')} missing keys: {missing}")

                fp = s["formatted_prompt"]
                self.assertIn("<prompt>", fp, "Must contain <prompt> tag")
                self.assertIn("</prompt>", fp, "Must contain </prompt> tag")
                self.assertIn("<think>", fp, "Must contain <think> tag")
                self.assertIn("</think>", fp, "Must contain </think> tag")
                self.assertIn("<answer>", fp, "Must contain <answer> tag")
                self.assertIn("</answer>", fp, "Must contain </answer> tag")

                # Verify answer query executes on target DB
                db_id = s["db_id"]
                sql = s["answer"]
                conn = fin_conn if db_id == "financial_mini" else sch_conn
                cursor = conn.cursor()
                cursor.execute(sql)
                rows = cursor.fetchall()
                self.assertTrue(len(rows) > 0, f"SynSQL query {s['id']} returned 0 rows: {sql}")
        finally:
            fin_conn.close()
            sch_conn.close()


class TestColumnCardExtractorStress(unittest.TestCase):
    """Adversarial stress testing on ColumnCardExtractor with hostile schemas."""

    def setUp(self):
        self.temp_db_file = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
        self.temp_db_path = self.temp_db_file.name
        self.temp_db_file.close()

    def tearDown(self):
        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)

    def test_hostile_column_names_and_nulls(self):
        """
        Creates an SQLite database containing:
        - Spaces in table names: "order items"
        - Reserved keywords as table/column: "select", "order", "group"
        - Spaces, hyphens, dots, and unicode in column names: "First Name", "user-id", "mã_đơn"
        - Column with 100% NULL values
        - Column with very long strings (> 100 chars)
        - Table with 0 rows
        """
        conn = sqlite3.connect(self.temp_db_path)
        cur = conn.cursor()

        # Table 1: Hostile table and column names
        cur.execute("""
            CREATE TABLE "order items" (
                "order_id" INTEGER PRIMARY KEY,
                "First Name" TEXT NOT NULL,
                "user-id" TEXT DEFAULT 'N/A',
                "select" TEXT,
                "mã_đơn" TEXT,
                "all_null_col" TEXT,
                "very_long_val" TEXT
            );
        """)

        # Table 2: Empty table with 0 rows
        cur.execute("""
            CREATE TABLE "empty_table" (
                "id" INTEGER PRIMARY KEY,
                "note" TEXT
            );
        """)

        # Insert hostile data into Table 1
        cur.execute("""
            INSERT INTO "order items" (
                "order_id", "First Name", "user-id", "select", "mã_đơn", "all_null_col", "very_long_val"
            ) VALUES (
                1, 'John Doe', 'USR-001', 'OPTION_A', 'DON_HANG_123', NULL,
                'This is an exceptionally long column sample text designed to trigger length truncation boundaries in the ColumnCardExtractor module.'
            ), (
                2, 'Jane Smith', 'USR-002', 'OPTION_B', 'DON_HANG_456', NULL,
                'Another extremely long string with unicode characters: tiếng Việt có dấu và ký tự lạ !@#$%^&*().'
            );
        """)
        conn.commit()
        conn.close()

        # Extract column cards
        extractor = ColumnCardExtractor(sample_limit=3, max_sample_len=30)
        cards = extractor.extract_from_sqlite(self.temp_db_path)

        self.assertTrue(len(cards) >= 9, f"Should extract at least 9 column cards, got {len(cards)}")

        cards_by_name = {f"{c.table_name}.{c.column_name}": c for c in cards}

        # Check special column names
        self.assertIn('order items.First Name', cards_by_name)
        self.assertIn('order items.user-id', cards_by_name)
        self.assertIn('order items.select', cards_by_name)
        self.assertIn('order items.mã_đơn', cards_by_name)
        self.assertIn('order items.all_null_col', cards_by_name)
        self.assertIn('empty_table.id', cards_by_name)

        # Check NULL column: sample_values must be empty list
        null_card = cards_by_name['order items.all_null_col']
        self.assertEqual(null_card.sample_values, [], "100% NULL column must have empty sample_values")

        # Check empty table: sample_values must be empty list
        empty_card = cards_by_name['empty_table.note']
        self.assertEqual(empty_card.sample_values, [], "Empty table must have empty sample_values")

        # Check truncation on long text
        long_card = cards_by_name['order items.very_long_val']
        self.assertTrue(len(long_card.sample_values) > 0)
        for s in long_card.sample_values:
            self.assertTrue(len(s) <= 33, f"Sample value should be truncated to ~33 chars, got {len(s)}: {s}")
            self.assertTrue(s.endswith("..."), f"Truncated sample must end with '...', got {s}")

        # Check to_text formatting doesn't crash
        for c in cards:
            text = c.to_text()
            self.assertIsInstance(text, str)
            self.assertIn(c.table_name, text)
            self.assertIn(c.column_name, text)

    def test_schema_dict_extraction_robustness(self):
        """Tests extraction from dictionary with missing or malformed entries."""
        extractor = ColumnCardExtractor()

        # 1. Empty dict
        self.assertEqual(extractor.extract_from_schema_dict({}), [])

        # 2. Malformed / minimal schema dict
        schema_dict = {
            "tables": {
                "t1": {
                    "columns": {
                        "c1": "TEXT", # string format
                        "c2": {"type": "INTEGER", "constraints": ["PRIMARY KEY"]},
                    }
                }
            }
        }
        cards = extractor.extract_from_schema_dict(schema_dict)
        self.assertEqual(len(cards), 2)
        c1 = next(c for c in cards if c.column_name == "c1")
        self.assertEqual(c1.data_type, "TEXT")

    def test_implicit_foreign_key_inference_from_sqlite(self):
        """
        Creates SQLite DB without any explicit FOREIGN KEY constraints.
        Tables: orders(id, user_id, amount), users(id, name).
        Verification: SchemaGraphBuilder with infer_implicit_fks=True infers relation
        and Steiner Tree successfully connects orders and users.
        """
        conn = sqlite3.connect(self.temp_db_path)
        cur = conn.cursor()
        cur.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);")
        cur.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL);")
        cur.execute("INSERT INTO users VALUES (1, 'Alice');")
        cur.execute("INSERT INTO orders VALUES (101, 1, 50.0);")
        conn.commit()
        conn.close()

        graph = SchemaGraphBuilder.build_from_sqlite(self.temp_db_path, infer_implicit_fks=True)
        self.assertTrue(graph.has_edge("orders", "users"), "Should infer implicit FK edge between orders and users")
        edge_data = graph.get_edge_data("orders", "users")
        self.assertEqual(edge_data.get("weight"), 2.0, "Inferred edge should carry penalty weight 2.0")

        solver = NodeWeightedSteinerTree(graph)
        sel, bridges, fks, ver = solver.find_steiner_tree(["orders", "users"])
        self.assertTrue(ver, "Should verify connectivity through inferred FK edge")
        self.assertEqual(sorted(sel), ["orders", "users"])

    def test_data_dict_custom_descriptions(self):
        """Tests that external data_dict correctly overrides synthesized column descriptions."""
        conn = sqlite3.connect(self.temp_db_path)
        cur = conn.cursor()
        cur.execute("CREATE TABLE products (pid INTEGER PRIMARY KEY, sku TEXT, price REAL);")
        conn.commit()
        conn.close()

        data_dict = {
            "products.sku": "Stock Keeping Unit unique barcode",
            "products": {"price": "Retail price in USD without VAT"}
        }

        extractor = ColumnCardExtractor()
        cards = extractor.extract_from_sqlite(self.temp_db_path, data_dict=data_dict)
        card_map = {c.column_name: c for c in cards}

        self.assertEqual(card_map["sku"].column_description, "Stock Keeping Unit unique barcode")
        self.assertEqual(card_map["price"].column_description, "Retail price in USD without VAT")


class TestLitESQLEmbeddingRetrieverStress(unittest.TestCase):
    """Adversarial stress testing on LitESQLEmbeddingRetriever."""

    def setUp(self):
        # Build standard test cards from data/mini
        cards_path = os.path.abspath("data/mini/column_cards_mini.jsonl")
        self.cards = ColumnCardExtractor.load_cards_from_jsonl(cards_path)

    def test_empty_retriever(self):
        """Retriever with 0 cards returns empty tuple gracefully."""
        retriever = LitESQLEmbeddingRetriever(cards=[])
        res, tables, scores = retriever.retrieve("find schools with high SAT score")
        self.assertEqual(res, [])
        self.assertEqual(tables, {})
        self.assertEqual(scores, {})

    def test_empty_and_hostile_query_strings(self):
        """Retriever handles empty strings, whitespace, None, and pure symbols."""
        retriever = LitESQLEmbeddingRetriever(cards=self.cards, top_k=5)

        hostile_inputs = [
            "",
            "   \t\n   ",
            "!@#$%^&*()_+=-{}[]:;'<>?,./",
            "1234567890",
            "SELECT * FROM schools WHERE 1=1;",
        ]

        for q in hostile_inputs:
            cards, tables, scores = retriever.retrieve(q)
            self.assertIsInstance(cards, list)
            self.assertIsInstance(tables, dict)
            self.assertIsInstance(scores, dict)
            self.assertTrue(len(cards) <= 5)

    def test_multi_word_query_and_ranking(self):
        """Retriever ranks relevant tables on multi-word natural queries."""
        retriever = LitESQLEmbeddingRetriever(cards=self.cards, top_k=10)

        # Multi-word query targeted at financial
        q_fin = "What is the average transaction amount for classic cards in Prague district?"
        cards_fin, tables_fin, scores_fin = retriever.retrieve(q_fin)

        self.assertTrue(len(cards_fin) > 0)
        # Should retrieve financial tables: transactions, cards, or districts
        fin_tables = {"transactions", "cards", "districts", "accounts", "clients"}
        found_fin = set(tables_fin.keys()).intersection(fin_tables)
        self.assertTrue(len(found_fin) >= 2, f"Expected financial tables, got: {tables_fin.keys()}")

        # Multi-word query targeted at schools
        q_sch = "Find the high school name and average SAT math score in Alameda county"
        cards_sch, tables_sch, scores_sch = retriever.retrieve(q_sch)
        self.assertTrue(len(cards_sch) > 0)
        sch_tables = {"schools", "satscores", "frpm"}
        found_sch = set(tables_sch.keys()).intersection(sch_tables)
        self.assertTrue(len(found_sch) >= 2, f"Expected school tables, got: {tables_sch.keys()}")

    def test_out_of_vocabulary_query(self):
        """Query with unknown words should not crash and still return Top-K cards."""
        retriever = LitESQLEmbeddingRetriever(cards=self.cards, top_k=3)
        cards, tables, scores = retriever.retrieve("xyzzy qwerty plugh zork")
        self.assertEqual(len(cards), 3)

    def test_extremely_long_query_performance(self):
        """Query with 10,000 characters completes in < 50ms without recursion or memory issues."""
        retriever = LitESQLEmbeddingRetriever(cards=self.cards, top_k=5)
        long_query = "school score student " * 500  # 10,500 chars

        start_t = time.perf_counter()
        cards, tables, scores = retriever.retrieve(long_query)
        elapsed_ms = (time.perf_counter() - start_t) * 1000

        self.assertTrue(len(cards) > 0)
        self.assertLess(elapsed_ms, 50.0, f"Long query took too long: {elapsed_ms:.2f}ms")

    def test_export_offline_index_json(self):
        """Tests that export_offline_index produces valid index file."""
        retriever = LitESQLEmbeddingRetriever(cards=self.cards[:5])
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            temp_path = tf.name

        try:
            retriever.export_offline_index(temp_path)
            self.assertTrue(os.path.exists(temp_path))
            with open(temp_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("doc_count", data)
            self.assertIn("cards", data)
            self.assertEqual(len(data["cards"]), 5)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_large_k_greater_than_total_cards(self):
        """When top_k is larger than available cards, returns all cards without IndexError."""
        retriever = LitESQLEmbeddingRetriever(cards=self.cards[:7], top_k=50)
        cards, tables, scores = retriever.retrieve("find schools and students")
        self.assertEqual(len(cards), 7, "Should return all available 7 cards")

    def test_vietnamese_unicode_query(self):
        """Retriever handles non-ASCII unicode natural language query safely."""
        retriever = LitESQLEmbeddingRetriever(cards=self.cards, top_k=5)
        cards, tables, scores = retriever.retrieve("Tìm điểm thi SAT môn Toán trung bình cao nhất")
        self.assertTrue(len(cards) > 0)
        self.assertTrue(len(scores) > 0)
        for s in scores.values():
            self.assertTrue(0.0 <= s <= 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
