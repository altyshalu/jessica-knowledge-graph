import os
import tempfile
import unittest

from jkg import HybridMemory


class SemanticMemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "memory.sqlite3")
        self.hm = HybridMemory(db_path=self.db_path)

    def tearDown(self):
        self.hm.conn.close()
        self.tmpdir.cleanup()

    def test_project_tool_rule_entries_are_persisted(self):
        self.hm.remember_project(
            "Jessica Knowledge Graph stores durable project truths about the Nomads AI Lab stack.",
            project="jessica-knowledge-graph",
            canonical_key="project:jkg:mission",
        )
        self.hm.remember_tool(
            "Hermes is the autonomous runtime used by Jessica for orchestration and messaging.",
            tool_name="Hermes",
            canonical_key="tool:hermes:runtime",
        )
        self.hm.remember_rule(
            "Never overwrite user-owned project files without explicit approval or a clear migration reason.",
            rule_name="file-safety",
            canonical_key="rule:file-safety",
        )

        semantic_entries = self.hm.list_semantic()
        self.assertEqual(len(semantic_entries), 3)
        self.assertEqual({entry["kind"] for entry in semantic_entries}, {"project", "tool", "rule"})

    def test_semantic_entries_are_upserted_by_canonical_key(self):
        self.hm.remember_project(
            "Jessica uses CLARK retrieval for durable semantic recall.",
            project="jkg",
            canonical_key="project:jkg:retrieval",
        )
        self.hm.remember_project(
            "Jessica uses CLARK retrieval with durable semantic recall and ranking.",
            project="jkg",
            canonical_key="project:jkg:retrieval",
        )

        semantic_entries = self.hm.list_semantic(kind="project")
        self.assertEqual(len(semantic_entries), 1)
        self.assertIn("durable semantic recall", semantic_entries[0]["statement"].lower())

    def test_query_surfaces_semantic_memory(self):
        self.hm.remember_tool(
            "Hermes is the runtime behind Jessica and should be used for autonomous agent orchestration.",
            tool_name="Hermes",
            canonical_key="tool:hermes:orchestrator",
        )

        result = self.hm.query("What runtime powers Jessica?", layers=["semantic"])
        self.assertTrue(result["results"])
        self.assertEqual(result["results"][0]["layer"], "semantic")
        self.assertIn("Hermes", result["results"][0]["statement"])

    def test_session_context_includes_semantic_memory(self):
        self.hm.remember_project(
            "The project focuses on stable memory for project knowledge, tools, and rules.",
            project="jkg",
            canonical_key="project:jkg:focus",
        )
        self.hm.remember_rule(
            "Prioritize durable truths over transient chat fragments.",
            rule_name="durable-truths",
            canonical_key="rule:durable-truths",
        )

        context = self.hm.session_start_context(owner="jkg")
        self.assertIn("[Semantic Memory]", context)
        self.assertIn("stable memory", context.lower())
        self.assertIn("durable truths", context.lower())

    def test_verified_deletion_scrubs_all_memory_layers(self):
        self.hm.remember("Alice works with Bob")
        self.hm.remember_profile("Alice prefers concise answers")
        self.hm.remember_project(
            "Alice owns the launch checklist.",
            project="jkg",
            canonical_key="project:jkg:alice-checklist",
        )
        self.hm.remember_session("session-alice", "Alice reviewed the launch plan")
        self.hm.index_skill("alice-review", description="Review work assigned to Alice")

        result = self.hm.gdpr_delete("alice", request_id="test-001", verified=True)

        self.assertEqual(result["status"], "deleted")
        self.assertFalse(self.hm.query("Who is Alice?")["results"])
        for table, columns in {
            "episodes": ("title", "body"),
            "memory_profile": ("key", "value"),
            "memory_sessions": ("title", "summary", "full_text"),
            "memory_skills": ("name", "description", "triggers"),
            "memory_semantic": ("subject", "canonical_key", "statement"),
        }.items():
            predicate = " OR ".join(f"{column} LIKE ?" for column in columns)
            row = self.hm.conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {predicate}",
                tuple("%Alice%" for _ in columns),
            ).fetchone()
            self.assertEqual(row[0], 0, table)


if __name__ == "__main__":
    unittest.main()
