import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

import numpy as np

from jkg import HybridMemory, EmbeddingConfigurationError
from jkg.embeddings import EmbeddingEngine


class LocalMemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = str(Path(self.tmp.name) / 'memory.sqlite3')
        self.llm = patch('jkg.memory._llm', side_effect=RuntimeError('Offline test'))
        self.llm.start()
        self.addCleanup(self.llm.stop)
        self.memory = HybridMemory(self.path, embedding_provider='hash')
        self.addCleanup(self.memory.conn.close)

    def test_readme_preference_survives_reopen_and_is_recalled(self):
        result = self.memory.remember_profile('Alice prefers concise answers')
        self.assertEqual(result['stored'], 1)
        other = HybridMemory(self.path, embedding_provider='hash')
        self.addCleanup(other.conn.close)
        results = other.query('How does Alice prefer to communicate?', layers=['profile'])['results']
        self.assertIn('concise answers', results[0]['value'])

    def test_readme_employment_has_graph_and_date(self):
        result = self.memory.remember('Alice works at Acme since 2023')
        self.assertEqual(result['facts'], 1)
        self.assertEqual(result['relations'], 1)
        row = self.memory.conn.execute('SELECT predicate, object_text, valid_from FROM facts').fetchone()
        self.assertEqual(tuple(row), ('works_at', 'acme', '2023-01-01'))
        self.assertIn('acme', json.dumps(self.memory.query('Where does Alice work?', layers=['factual'])))

    def test_russian_employment_and_profile(self):
        self.assertEqual(self.memory.remember('Алина работает в компании Acme с 2023 года')['facts'], 1)
        self.assertEqual(self.memory.remember_profile('Я предпочитаю короткие ответы')['stored'], 1)
        results = self.memory.query('Какие ответы я предпочитаю?', layers=['profile'])['results']
        self.assertIn('короткие ответы', results[0]['value'])

    def test_profile_updates_do_not_overwrite_another_person(self):
        for text in ['Alice prefers concise answers', 'Bob prefers detailed answers',
                     'Alice prefers detailed explanations in responses']:
            self.memory.remember_profile(text)
        rows = self.memory.conn.execute('SELECT key, value FROM memory_profile ORDER BY key').fetchall()
        self.assertEqual(len(rows), 2)
        self.assertNotIn('concise', rows[0]['value'])
        self.assertIn('Bob', rows[1]['value'])

    def test_first_person_identity_and_preferences(self):
        for text in ['My name is Altynai', 'I am a software engineer', 'I prefer concise answers']:
            self.assertEqual(self.memory.remember_profile(text)['stored'], 1)
        keys = {r[0] for r in self.memory.conn.execute('SELECT key FROM memory_profile')}
        self.assertEqual(keys, {'name', 'role', 'communication_style'})

    def test_unrecognized_profile_is_preserved_and_reported(self):
        text = 'The way I choose tools depends on the task at hand'
        result = self.memory.remember_profile(text)
        self.assertEqual(result['status'], 'no_profile_facts')
        self.assertEqual(result['extraction_status'], 'raw_only')
        row = self.memory.conn.execute('SELECT body FROM episodes WHERE id=?', (result['episode_id'],)).fetchone()
        self.assertEqual(row[0], text)

    def test_questions_hypotheticals_negated_jobs_are_not_asserted(self):
        for text in ['Does Alice work at Acme?', 'Alice works at Acme?', 'If Alice works at Acme',
                     'Maybe Alice works at Acme', 'Alice does not work at Acme',
                     'Alice said Bob works at Acme', 'Алина не работает в Acme']:
            with self.subTest(text=text):
                self.assertEqual(self.memory.remember(text)['facts'], 0)

    def test_negative_preference_is_not_positive(self):
        self.memory.remember('Alice does not like coffee')
        row = self.memory.conn.execute('SELECT predicate, object_text FROM facts').fetchone()
        self.assertEqual(tuple(row), ('dislikes', 'coffee'))

    def test_multiple_sentences_keep_separate_subjects(self):
        result = self.memory.remember('Alice works at Acme since 2023. Bob lives in Paris since 2024.')
        self.assertEqual(result['facts'], 2)
        rows = self.memory.conn.execute('SELECT e.name, f.object_text FROM facts f JOIN entities e ON e.id=f.subject_id ORDER BY e.name').fetchall()
        self.assertEqual([tuple(r) for r in rows], [('alice', 'acme'), ('bob', 'paris')])

    def test_same_year_employment_interval_stays_closed(self):
        self.memory.remember('Alice worked at Acme from 2023 to 2023')
        row = self.memory.conn.execute('SELECT valid_from, valid_until FROM facts').fetchone()
        self.assertEqual(tuple(row), ('2023-01-01', '2023-12-31'))
        self.assertFalse(self.memory.query('Where does Alice work now?', layers=['factual'])['results'])

    def test_preference_correction_excludes_invalidated_fact(self):
        self.memory.remember('Teston hates JavaScript')
        self.memory.remember('Teston absolutely loves JavaScript')
        results = self.memory.query('How does Teston feel about JavaScript?', layers=['factual'])['results']
        self.assertTrue(results)
        self.assertIn('loves javascript', results[0]['value'])
        self.assertNotIn('hates', json.dumps(results))

    def test_duplicate_fact_does_not_duplicate_relation(self):
        text = 'Alice works at Acme since 2023'
        self.memory.remember(text)
        self.memory.remember(text)
        self.assertEqual(self.memory.conn.execute('SELECT COUNT(*) FROM facts').fetchone()[0], 1)
        self.assertEqual(self.memory.conn.execute('SELECT COUNT(*) FROM relations').fetchone()[0], 1)

    def test_reindex_failure_keeps_old_vectors_and_signature(self):
        self.memory.remember_project('The project uses SQLite', project='jkg')
        before = self.memory.conn.execute('SELECT embedding FROM memory_semantic_vec').fetchone()[0]
        signature = self.memory.conn.execute("SELECT value FROM memory_meta WHERE key='embedding_signature'").fetchone()[0]
        with patch.object(self.memory, '_encode', side_effect=EmbeddingConfigurationError('Unavailable')):
            with self.assertRaises(EmbeddingConfigurationError):
                self.memory.reindex_embeddings(force=True)
        self.assertEqual(self.memory.conn.execute('SELECT embedding FROM memory_semantic_vec').fetchone()[0], before)
        self.assertEqual(self.memory.conn.execute("SELECT value FROM memory_meta WHERE key='embedding_signature'").fetchone()[0], signature)

    def test_failed_embedding_rolls_back_fact_profile_and_project_writes(self):
        operations = [
            lambda: self.memory.remember('Alice works at Acme since 2023'),
            lambda: self.memory.remember_profile('Alice prefers concise answers'),
            lambda: self.memory.remember_project('Use SQLite', project='jkg'),
        ]
        for operation in operations:
            with patch.object(self.memory, '_encode', side_effect=EmbeddingConfigurationError('Unavailable')):
                with self.assertRaises(EmbeddingConfigurationError):
                    operation()
            for table in ['episodes', 'facts', 'relations', 'memory_profile', 'memory_semantic']:
                self.assertEqual(self.memory.conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0], 0, table)

    def test_updates_refresh_vectors_instead_of_leaving_stale_index(self):
        cases = [
            ('memory_profile_vec', lambda text: self.memory.remember_profile(text),
             'Alice prefers short answers', 'Alice prefers detailed responses'),
            ('memory_semantic_vec', lambda text: self.memory.remember_project(text, project='jkg', canonical_key='db'),
             'Use SQLite for storage', 'Use PostgreSQL for storage'),
        ]
        for table, write, first, second in cases:
            with self.subTest(table=table):
                write(first)
                before = self.memory.conn.execute(f'SELECT embedding FROM {table}').fetchone()[0]
                write(second)
                rows = self.memory.conn.execute(f'SELECT embedding FROM {table}').fetchall()
                self.assertEqual(len(rows), 1)
                self.assertNotEqual(rows[0][0], before)

    def test_hash_mode_is_explicit_in_query(self):
        result = self.memory.query('anything')
        self.assertEqual(result['embedding_provider'], 'hash')
        self.assertFalse(result['semantic_search'])

    def test_malformed_llm_payload_uses_local_extraction(self):
        for payload in ['not json', '[]', '{"facts": [null]}']:
            with self.subTest(payload=payload), patch('jkg.memory._llm', return_value=payload):
                result = self.memory.remember('Alice works at Acme since 2023')
                self.assertEqual(result['facts'] + result['duplicates'], 1)
                self.assertEqual(self.memory.remember_profile('I prefer concise answers')['stored'], 1)

    def test_unrecognized_text_is_not_silently_truncated(self):
        text = 'Unstructured source material. ' * 250 + 'UNIQUE_END_MARKER'
        result = self.memory.remember(text)
        row = self.memory.conn.execute('SELECT body FROM episodes WHERE id=?', (result['episode_id'],)).fetchone()
        self.assertEqual(row[0], text)

    def test_distance_is_converted_to_cosine(self):
        self.assertAlmostEqual(self.memory._distance_similarity(1), 0.5)
        self.assertAlmostEqual(self.memory._distance_similarity(2 ** 0.5), 0)


class EmbeddingBackendTests(unittest.TestCase):
    def test_default_is_local_and_model_failure_does_not_select_hash(self):
        with patch.dict(os.environ, {'JKG_EMBEDDING_PROVIDER': '', 'GEMINI_API_KEY': ''}):
            with patch('jkg.embeddings._local_model', side_effect=EmbeddingConfigurationError('No model')):
                with self.assertRaises(EmbeddingConfigurationError):
                    EmbeddingEngine()

    def test_remote_failure_does_not_return_fake_vectors(self):
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-placeholder'}):
            engine = EmbeddingEngine('gemini', 'gemini-embedding-001', 3)
            for payload in [{'error': {'message': 'failed'}}, {'embedding': {'values': [0, 0, 0]}},
                            {'embedding': {'values': [1, 2]}}, {'embedding': {'values': [1, float('nan'), 2]}}]:
                response = Mock()
                response.json.return_value = payload
                with patch('requests.post', return_value=response):
                    with self.subTest(payload=payload), self.assertRaises(EmbeddingConfigurationError):
                        engine.encode('a memory')

    def test_remote_vectors_are_requested_at_exact_dimension_and_normalized(self):
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-placeholder'}):
            engine = EmbeddingEngine('gemini', 'gemini-embedding-001', 3)
            response = Mock()
            response.json.return_value = {'embedding': {'values': [3, 4, 0]}}
            with patch('requests.post', return_value=response) as request:
                np.testing.assert_allclose(engine.encode('hello'), [0.6, 0.8, 0], atol=1e-6)
                self.assertEqual(request.call_args.kwargs['json']['outputDimensionality'], 3)


@unittest.skipUnless(os.environ.get('JKG_TEST_SEMANTIC') == '1', 'Set JKG_TEST_SEMANTIC=1 to run the downloaded local model')
class RealLocalSemanticTests(unittest.TestCase):
    def test_multilingual_retrieval_without_shared_keywords(self):
        with tempfile.TemporaryDirectory() as directory, patch('jkg.memory._llm', side_effect=RuntimeError('Offline')):
            memory = HybridMemory(str(Path(directory) / 'semantic.sqlite3'), embedding_provider='sentence-transformers')
            self.addCleanup(memory.conn.close)
            entries = {
                'brief': 'Keep replies short and avoid lengthy explanations.',
                'tests': 'Execute automated checks before publishing a software release.',
                'backup': 'Create a backup copy before modifying important files.',
                'food': 'The cafeteria serves vegetarian meals at noon.',
            }
            for key, statement in entries.items():
                memory.remember_rule(statement, rule_name=key, canonical_key=key)
            cases = [('Какой длины должны быть ответы?', 'brief'),
                     ('Что делать перед изменением важных файлов?', 'backup'),
                     ('Как проверить программу перед выпуском?', 'tests'),
                     ('Concise communication guidelines', 'brief')]
            for query, expected in cases:
                with self.subTest(query=query):
                    results = memory.query(query, layers=['semantic'])['results']
                    self.assertTrue(results)
                    self.assertEqual(results[0]['canonical_key'], expected)
            self.assertTrue(memory.query('settings')['semantic_search'])


if __name__ == '__main__':
    unittest.main()
