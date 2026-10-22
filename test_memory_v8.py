import os
import tempfile
import unittest

from jkg import EmbeddingConfigurationError, HybridMemory


class MemoryV8Tests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmpdir.name, "memory.sqlite3")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_namespaces_are_physically_isolated(self):
        alpha = HybridMemory(self.path, embedding_provider="hash", namespace="alpha")
        beta = HybridMemory(self.path, embedding_provider="hash", namespace="beta")
        self.assertNotEqual(alpha.db_path, beta.db_path)
        alpha.remember_project("Alpha launch rule", project="alpha", canonical_key="alpha:rule")
        self.assertEqual(beta.list_semantic(), [])
        alpha.conn.close()
        beta.conn.close()

    def test_every_memory_layer_has_a_persisted_vector_index(self):
        memory = HybridMemory(self.path, embedding_provider="hash")
        if not memory.has_vec:
            self.skipTest("sqlite-vec unavailable")
        memory.remember_profile("My name is Alice")
        memory.remember_project("The product uses SQLite", project="jkg", canonical_key="jkg:db")
        memory.remember_session("s1", "Discussed the launch roadmap", summary="Launch roadmap")
        memory.index_skill("review", "Review production code", ["review code"])
        for table in ("memory_profile_vec", "memory_semantic_vec", "memory_session_vec", "memory_skill_vec"):
            self.assertGreater(memory.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0], 0)
        memory.conn.close()

    def test_retrieval_changes_salience_not_truth_confidence(self):
        memory = HybridMemory(self.path, embedding_provider="hash")
        memory.remember("Alice loves concise answers")
        row = memory.conn.execute("SELECT id, confidence, salience FROM facts LIMIT 1").fetchone()
        memory._clark_confidence_update([{"fact_id": row["id"]}], boost=0.1)
        updated = memory.conn.execute("SELECT confidence, salience FROM facts WHERE id=?", (row["id"],)).fetchone()
        self.assertEqual(updated["confidence"], row["confidence"])
        self.assertGreater(updated["salience"], row["salience"])
        memory.conn.close()

    def test_embedding_registry_requires_explicit_reindex(self):
        memory = HybridMemory(self.path, embedding_provider="hash")
        memory.remember_project("Persist this vector", project="jkg", canonical_key="jkg:vector")
        memory.conn.execute(
            "UPDATE memory_meta SET value='old-provider:old-model:384' WHERE key='embedding_signature'"
        )
        memory.conn.commit()
        memory.conn.close()
        with self.assertRaises(EmbeddingConfigurationError):
            HybridMemory(self.path, embedding_provider="hash")
        migrated = HybridMemory(self.path, embedding_provider="hash", reindex_on_mismatch=True)
        self.assertEqual(
            migrated.conn.execute("SELECT COUNT(*) FROM memory_semantic_vec").fetchone()[0], 1
        )
        migrated.conn.close()

    def test_feedback_aliases_consolidation_and_context_budget(self):
        memory = HybridMemory(self.path, embedding_provider="hash")
        memory.remember("Alice loves concise answers")
        fact = memory.conn.execute("SELECT id FROM facts LIMIT 1").fetchone()
        self.assertEqual(memory.feedback("factual", fact["id"], "useful")["status"], "recorded")
        memory.add_entity_alias("alice", "Al")
        self.assertEqual(memory._resolve_entity("Al"), memory._resolve_entity("alice"))
        self.assertTrue(memory.consolidate(dry_run=True)["dry_run"])
        self.assertLessEqual(len(memory.session_start_context(owner="alice", max_chars=1000)), 1200)
        memory.conn.close()

    def test_clark_astar_returns_most_reliable_optimal_path(self):
        memory = HybridMemory(self.path, embedding_provider="hash")
        a = memory._ensure_entity("alpha")
        b = memory._ensure_entity("beta")
        goal = memory._ensure_entity("goal")
        memory.conn.executemany(
            "INSERT INTO relations(subject_id,predicate,object_id,confidence) VALUES(?,?,?,?)",
            [
                (a, "direct", goal, 0.5),
                (a, "via", b, 0.9),
                (b, "via", goal, 0.9),
            ],
        )
        memory.conn.commit()
        paths = memory._clark_astar_entities({a: 0.0}, max_nodes=10, max_depth=3)
        self.assertEqual(paths[goal]["path"], ["alpha", "beta", "goal"])
        self.assertAlmostEqual(paths[goal]["path_reliability"], 0.81, places=6)
        memory.conn.close()


if __name__ == "__main__":
    unittest.main()
