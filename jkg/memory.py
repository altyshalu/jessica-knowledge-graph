#!/usr/bin/env python3
"""
JKG 8 — semantic, profile, factual, episodic, and procedural memory.
One isolated SQLite database per namespace; remote providers are optional.
"""
import os, sys, json, sqlite3, re, hashlib, math, time, heapq
from datetime import datetime
from collections import defaultdict, deque

try:
    import requests
except Exception:
    requests = None
import numpy as np
from .config import (
    EmbeddingConfigurationError,
    MemoryConfig,
    namespace_db_path,
    normalize_namespace,
)
from .lifecycle import ContextBudget, FeedbackSignal
from .embeddings import EmbeddingEngine
from .local_extraction import extract_assertions, profile_payload, statements

# ═══════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════

def _load_env():
    """Load .env from current dir, ~/.jkg/, or JKG_ENV_PATH."""
    env_paths = [
        os.environ.get("JKG_ENV_PATH", ""),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"),
        os.path.expanduser("~/.jkg/.env"),
        os.path.expanduser("~/.hermes/.env"),
        ".env",
    ]
    for env_path in env_paths:
        if env_path and os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, _, v = line.partition("=")
                        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            break
_load_env()

DB_PATH = os.path.expanduser(os.environ.get(
    "JKG_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory_v3.db")
))

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
API_BASE = "https://api.deepseek.com"
LLM_TIMEOUT_SECONDS = float(os.environ.get("JKG_LLM_TIMEOUT_SECONDS", "20"))
EMBEDDING_TIMEOUT_SECONDS = float(os.environ.get("JKG_EMBEDDING_TIMEOUT_SECONDS", "8"))
SQLITE_BUSY_TIMEOUT_MS = int(os.environ.get("JKG_SQLITE_BUSY_TIMEOUT_MS", "15000"))
SEMANTIC_EMBED_LIMIT = int(os.environ.get("JKG_SEMANTIC_EMBED_LIMIT", "12"))

def _llm(prompt: str, system: str = "You are a precise knowledge extraction engine.") -> str:
    if requests is None:
        raise RuntimeError("requests is not installed; remote LLM extraction is unavailable")
    if not API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured")
    r = requests.post(
        f"{API_BASE}/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"model": "deepseek-chat", "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ], "temperature": 0.1},
        timeout=LLM_TIMEOUT_SECONDS,
    )
    try:
        r.raise_for_status()
        payload = r.json()
        return payload["choices"][0]["message"]["content"]
    except Exception as exc:
        body = (r.text or "")[:300]
        raise RuntimeError(f"LLM request failed: {exc}; body={body}") from exc


def _title_case_name(text: str) -> str:
    return " ".join(part.capitalize() for part in re.split(r"\s+", text.strip()) if part)


def _slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Zа-яА-Я0-9]+", "-", str(text or "").strip().lower())
    return text.strip("-") or "item"


def _fallback_profile_payload(text: str) -> dict:
    return profile_payload(text)


def _fallback_semantic_payload(
    statement: str,
    *,
    kind: str,
    scope: str,
    subject: str = "",
    canonical_key: str = "",
    tags: list | None = None,
    confidence: float = 0.9,
) -> dict:
    normalized_kind = (kind or "project").strip().lower()
    inferred_subject = (subject or normalized_kind).strip()
    inferred_tags = [str(tag).strip().lower() for tag in (tags or []) if str(tag).strip()]
    topic_tokens = [
        token for token in re.findall(r"[a-zA-Zа-яА-Я0-9_]+", statement.lower())
        if token not in {"the", "and", "for", "with", "this", "that", "what", "how", "our", "your"}
    ]
    topic = "-".join(topic_tokens[:6]) or hashlib.md5(statement.encode()).hexdigest()[:8]
    key = canonical_key or f"{normalized_kind}:{_slugify(scope)}:{_slugify(inferred_subject)}:{topic}"
    return {
        "entries": [{
            "kind": normalized_kind,
            "scope": scope,
            "subject": inferred_subject,
            "canonical_key": key,
            "statement": statement.strip(),
            "tags": inferred_tags,
            "confidence": float(confidence),
            "details": {},
        }]
    }


def _fallback_extract_temporal_payload(text: str) -> dict:
    return {"reference_time": None, "facts": extract_assertions(text)}


def _legacy_extract_entities_payload(text: str) -> dict:
    raw = text.strip()
    entities = []
    entity_seen = set()
    facts = []
    relations = []

    def add_entity(name: str, etype: str = "entity"):
        clean = name.strip(" .")
        if not clean:
            return
        key = clean.lower()
        if key not in entity_seen:
            entities.append({"name": clean, "type": etype})
            entity_seen.add(key)

    def add_fact(subject: str, predicate: str, obj: str, subj_type: str = "entity", obj_type: str = None):
        subject = subject.strip(" .")
        obj = obj.strip(" .")
        if not subject or not obj:
            return
        add_entity(subject, subj_type)
        if obj_type:
            add_entity(obj, obj_type)
        facts.append({"subject": subject, "predicate": predicate, "object": obj})

    def add_relation(subject: str, predicate: str, obj: str, subj_type: str = "entity", obj_type: str = "entity"):
        subject = subject.strip(" .")
        obj = obj.strip(" .")
        if not subject or not obj:
            return
        add_entity(subject, subj_type)
        add_entity(obj, obj_type)
        relations.append({"subject": subject, "predicate": predicate, "object": obj})

    favorite_match = re.search(r"^([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*)'s favorite color is\s+(.+?)[\.!]?$", raw)
    if favorite_match:
        subject, obj = favorite_match.groups()
        add_fact(subject, "favorite", obj, obj_type="concept")

    least_favorite_match = re.search(r"^([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*)'s least favorite color is\s+(.+?)[\.!]?$", raw)
    if least_favorite_match:
        subject, obj = least_favorite_match.groups()
        add_fact(subject, "least_favorite", obj, obj_type="concept")

    love_match = re.search(r"^([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*)\s+(?:absolutely\s+)?loves\s+(.+?)[\.!]?$", raw)
    if love_match:
        subject, obj = love_match.groups()
        add_fact(subject, "loves", obj, obj_type="concept")

    hate_match = re.search(r"^([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*)\s+hates\s+(.+?)[\.!]?$", raw)
    if hate_match:
        subject, obj = hate_match.groups()
        add_fact(subject, "hates", obj, obj_type="concept")

    lived_match = re.search(r"^([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*)\s+lived in\s+(.+?)(?:\s+in\s+\d{4})?[\.!]?$", raw)
    if lived_match:
        subject, obj = lived_match.groups()
        add_fact(subject, "lived_in", obj, obj_type="location")

    moved_match = re.search(r"^([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*)\s+moved to\s+(.+?)(?:\s+in\s+\d{4})?[\.!]?$", raw)
    if moved_match:
        subject, obj = moved_match.groups()
        add_fact(subject, "lives_in", obj, obj_type="location")

    built_match = re.search(r"^([Tt]he\s+.+?)\s+built\s+(?:a\s+new\s+)?(.+?)[\.!]?$", raw)
    if built_match:
        subject, obj = built_match.groups()
        add_fact(subject, "builds", obj, obj_type="technology")

    works_with_match = re.search(r"^([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*)\s+works with\s+([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*)[\.!]?$", raw)
    if works_with_match:
        subject, obj = works_with_match.groups()
        add_relation(subject, "works_with", obj, obj_type="person")

    coffee_match = re.search(r"^(I)\s+drank\s+(.+?)[\.!]?$", raw, re.IGNORECASE)
    if coffee_match:
        subject, obj = coffee_match.groups()
        add_fact(subject, "drank", obj, subj_type="person", obj_type="concept")

    is_match = re.search(r"^([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*)\s+is\s+(?:an?\s+)?(.+?)[\.!]?$", raw)
    if is_match:
        subject, obj = is_match.groups()
        add_fact(subject, "is", obj, obj_type="concept")

    founded_by_match = re.search(r"^([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*)\s+is\s+.+?founded by\s+([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*)[\.!]?$", raw)
    if founded_by_match:
        subject, obj = founded_by_match.groups()
        add_relation(subject, "founded_by", obj)

    return {"entities": entities, "facts": facts, "relations": relations}


def _fallback_extract_entities_payload(text: str) -> dict:
    result = {"entities": [], "facts": [], "relations": []}
    for sentence in statements(text):
        assertions = extract_assertions(sentence)
        if assertions:
            for fact in assertions:
                subject, obj, predicate = fact["subject"], fact["object"], fact["predicate"]
                result["entities"].extend([
                    {"name": subject, "type": "person"},
                    {"name": obj, "type": fact["object_type"]},
                ])
                triple = {"subject": subject, "predicate": predicate, "object": obj}
                result["facts"].append(triple)
                result["relations"].append(triple.copy())
        else:
            # Retain older, narrow patterns without guessing unsupported clauses.
            if re.search(r"\b(?:not|never|не|said|says|that|сказал|сказала|что)\b", sentence, re.I):
                continue
            legacy = _legacy_extract_entities_payload(sentence)
            for key in result:
                result[key].extend(legacy[key])
    result["entities"] = list({e["name"].lower(): e for e in result["entities"]}.values())
    return result


# ═══════════════════════════════════════════════════
# HYBRID MEMORY — Graph + Embeddings, one DB
# ═══════════════════════════════════════════════════

class HybridMemory:
    """Unified five-layer memory backed by one SQLite database."""

    def __init__(self, db_path=None, embedding_model=None,
                 namespace: str = None, reindex_on_mismatch: bool = False,
                 embedding_provider: str = None, embedding_dimensions: int = None):
        self.namespace = normalize_namespace(namespace)
        db_path = namespace_db_path(db_path if db_path is not None else os.environ.get("JKG_DB_PATH", DB_PATH), self.namespace)
        self._embedding_engine = EmbeddingEngine(embedding_provider, embedding_model, embedding_dimensions)
        self.config = MemoryConfig(
            namespace=self.namespace, db_path=db_path,
            embedding_provider=self._embedding_engine.provider,
            embedding_model=self._embedding_engine.model,
            embedding_dimensions=self._embedding_engine.dimensions,
        )
        # Handle special SQLite paths like ":memory:"
        if db_path not in (":memory:", "") and os.path.dirname(db_path):
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(
            db_path,
            timeout=max(SQLITE_BUSY_TIMEOUT_MS / 1000.0, 1.0),
            check_same_thread=False,
        )
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA temp_store=MEMORY")

        # Load sqlite-vec extension
        try:
            import sqlite_vec
            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
        except Exception as e:
            print(f"WARNING: sqlite-vec not loaded: {e}", file=sys.stderr)

        self._init_db()

        self._validate_embedding_registry(reindex_on_mismatch=reindex_on_mismatch)

    @property
    def embedder(self):
        return self._embedding_engine

    def encode(self, texts, **kwargs):
        return self._embedding_engine.encode(texts, **kwargs)

    @staticmethod
    def _distance_similarity(distance):
        # vec0 uses Euclidean distance by default. For unit vectors:
        # ||a-b||² = 2 - 2*cos(a,b). 1-distance is not cosine similarity.
        return max(0.0, min(1.0, 1.0 - float(distance) ** 2 / 2.0))

    # SCHEMA — one DB, all tables
    # ═══════════════════════════════════════════════

    def _init_db(self):
        """Create ALL tables in one transaction."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS memory_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS memory_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_layer TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                signal TEXT NOT NULL,
                note TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_feedback_memory
                ON memory_feedback(memory_layer, memory_id);
            CREATE TABLE IF NOT EXISTS memory_access (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_layer TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                accessed_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_access_memory
                ON memory_access(memory_layer, memory_id, accessed_at);
            CREATE TABLE IF NOT EXISTS entity_aliases (
                alias TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                source TEXT DEFAULT 'manual',
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY(entity_id) REFERENCES entities(id)
            );

            -- GRAPH tables
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT DEFAULT 'entity',
                summary TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object_text TEXT NOT NULL,
                source TEXT DEFAULT 'manual',
                episode_id INTEGER,
                fact_hash TEXT,
                confidence REAL DEFAULT 0.5,
                utility_score REAL DEFAULT 0.5,
                forget_status TEXT DEFAULT 'active',
                access_count INTEGER DEFAULT 0,
                last_accessed_at TEXT,
                valid_from TEXT,
                valid_until TEXT,
                superseded_by INTEGER,
                invalidated_by_fact_id INTEGER,
                ingestion_time TEXT DEFAULT (datetime('now')),
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY(subject_id) REFERENCES entities(id),
                FOREIGN KEY(episode_id) REFERENCES episodes(id),
                FOREIGN KEY(superseded_by) REFERENCES facts(id)
            );
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT UNIQUE NOT NULL,
                title TEXT DEFAULT '',
                body TEXT NOT NULL,
                source_type TEXT DEFAULT 'manual',
                reference_time TEXT,
                emotion TEXT DEFAULT 'neutral',
                emotion_intensity REAL DEFAULT 0.0,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS schema_evolution (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                pattern_description TEXT NOT NULL,
                proposed_schema TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                evidence_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'proposed',
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS forget_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_id INTEGER,
                subject_id TEXT,
                predicate TEXT,
                object_text TEXT,
                utility_score REAL,
                action TEXT,
                reason TEXT,
                performed_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS deletion_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id TEXT,
                reason TEXT,
                gdpr_request_id TEXT,
                verified_by TEXT,
                performed_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object_id TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                source TEXT DEFAULT 'manual',
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject_id);
            CREATE INDEX IF NOT EXISTS idx_facts_predicate ON facts(predicate);
            CREATE INDEX IF NOT EXISTS idx_relations_subject ON relations(subject_id);
            CREATE INDEX IF NOT EXISTS idx_relations_object ON relations(object_id);
            CREATE INDEX IF NOT EXISTS idx_relations_both ON relations(subject_id, object_id);

            -- BM25 text search
            CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
                entity_name, predicate, object_text, content='facts_fts_content'
            );
            -- Shadow table for FTS5 content sync
            CREATE TABLE IF NOT EXISTS facts_fts_content (
                id INTEGER PRIMARY KEY,
                entity_name TEXT,
                predicate TEXT,
                object_text TEXT
            );

            -- JKG 7.1: UNIFIED MEMORY FABRIC — multi-layer memory
            -- PROFILE Layer: user identity, preferences, environment facts
            CREATE TABLE IF NOT EXISTS memory_profile (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL UNIQUE,
                value TEXT NOT NULL,
                category TEXT DEFAULT 'preference',
                confidence REAL DEFAULT 1.0,
                source TEXT DEFAULT 'explicit',
                is_active INTEGER DEFAULT 1,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_profile_key ON memory_profile(key);
            CREATE INDEX IF NOT EXISTS idx_profile_category ON memory_profile(category);

            -- EPISODIC Layer: session transcripts + summaries
            CREATE TABLE IF NOT EXISTS memory_sessions (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                source TEXT NOT NULL,
                title TEXT DEFAULT '',
                summary TEXT DEFAULT '',
                full_text TEXT DEFAULT '',
                emotion TEXT DEFAULT 'neutral',
                emotion_intensity REAL DEFAULT 0.0,
                importance REAL DEFAULT 0.5,
                fact_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_source ON memory_sessions(source);
            CREATE INDEX IF NOT EXISTS idx_sessions_created ON memory_sessions(created_at);

            -- PROCEDURAL Layer: skills indexed for retrieval
            CREATE TABLE IF NOT EXISTS memory_skills (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                triggers TEXT DEFAULT '[]',
                category TEXT DEFAULT 'general',
                version TEXT DEFAULT '1.0.0',
                is_active INTEGER DEFAULT 1,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_skills_name ON memory_skills(name);
            CREATE INDEX IF NOT EXISTS idx_skills_category ON memory_skills(category);

            -- SEMANTIC Layer: durable project/tool/rule knowledge
            CREATE TABLE IF NOT EXISTS memory_semantic (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                semantic_id TEXT NOT NULL UNIQUE,
                kind TEXT NOT NULL,
                scope TEXT DEFAULT 'global',
                subject TEXT DEFAULT '',
                canonical_key TEXT NOT NULL UNIQUE,
                statement TEXT NOT NULL,
                details_json TEXT DEFAULT '{}',
                tags_json TEXT DEFAULT '[]',
                confidence REAL DEFAULT 0.9,
                source TEXT DEFAULT 'manual',
                status TEXT DEFAULT 'active',
                access_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                last_reinforced_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_semantic_kind ON memory_semantic(kind);
            CREATE INDEX IF NOT EXISTS idx_semantic_scope ON memory_semantic(scope);
            CREATE INDEX IF NOT EXISTS idx_semantic_subject ON memory_semantic(subject);
        """)
        fact_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(facts)")}
        if "salience" not in fact_columns:
            self.conn.execute("ALTER TABLE facts ADD COLUMN salience REAL DEFAULT 0.5")
        self.conn.commit()

        # Create vec0 table if sqlite-vec is available
        try:
            self.conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS facts_vec USING vec0(
                    fact_id INTEGER PRIMARY KEY,
                    embedding float[{self.config.embedding_dimensions}]
                )
            """)
            self.has_vec = True
        except Exception:
            self.has_vec = False
            print("WARNING: vec0 not available, vector search disabled", file=sys.stderr)
        if self.has_vec:
            for table, id_column in (
                ("memory_semantic_vec", "semantic_row_id"),
                ("memory_profile_vec", "profile_row_id"),
                ("memory_session_vec", "session_row_id"),
                ("memory_skill_vec", "skill_row_id"),
            ):
                try:
                    self.conn.execute(f"""
                        CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0(
                            {id_column} INTEGER PRIMARY KEY,
                            embedding float[{self.config.embedding_dimensions}]
                        )
                    """)
                except Exception:
                    pass
        self.conn.commit()

    def _validate_embedding_registry(self, reindex_on_mismatch: bool = False):
        """Refuse to mix vectors produced by incompatible embedding configs."""
        signature = self.config.embedding_signature
        row = self.conn.execute(
            "SELECT value FROM memory_meta WHERE key='embedding_signature'"
        ).fetchone()
        if row is None:
            self.conn.executemany(
                "INSERT OR REPLACE INTO memory_meta(key, value, updated_at) VALUES(?,?,datetime('now'))",
                (
                    ("embedding_signature", signature),
                    ("namespace", self.namespace),
                    ("schema_version", "8"),
                ),
            )
            self.conn.commit()
            return
        if row["value"] == signature:
            return
        if reindex_on_mismatch:
            self.reindex_embeddings(force=True)
            return
        message = (
            "Embedding configuration changed from "
            f"{row['value']!r} to {signature!r}. Reopen with "
            "reindex_on_mismatch=True or use a new namespace/database."
        )
        self.conn.close()
        raise EmbeddingConfigurationError(message)

    def _replace_vector(self, table: str, id_column: str, row_id: int, text: str):
        if not self.has_vec:
            return
        vector = np.asarray(self._encode(text), dtype=np.float32)
        # vec0 does not implement SQLite's INSERT OR REPLACE conflict policy.
        # Replace explicitly and keep the previous vector if insertion fails.
        self.conn.execute("SAVEPOINT vector_replace")
        try:
            self.conn.execute(f"DELETE FROM {table} WHERE {id_column}=?", (row_id,))
            self.conn.execute(
                f"INSERT INTO {table}({id_column}, embedding) VALUES(?,?)",
                (row_id, vector.tobytes()),
            )
        except Exception:
            self.conn.execute("ROLLBACK TO vector_replace")
            self.conn.execute("RELEASE vector_replace")
            raise
        else:
            self.conn.execute("RELEASE vector_replace")

    def reindex_embeddings(self, force: bool = False) -> dict:
        """Rebuild every vector layer and update the embedding registry."""
        if not force:
            raise ValueError("Reindex is destructive; pass force=True after making a backup.")
        if not self.has_vec:
            raise RuntimeError("sqlite-vec is required to rebuild vector indexes")

        self.conn.execute("SAVEPOINT embedding_reindex")
        try:
            tables = {
                "facts_vec": "fact_id",
                "memory_semantic_vec": "semantic_row_id",
                "memory_profile_vec": "profile_row_id",
                "memory_session_vec": "session_row_id",
                "memory_skill_vec": "skill_row_id",
            }
            # vec0 fixes dimensions when a table is created. Recreate tables so a
            # provider dimension change is a real migration rather than a corrupt mix.
            for table in tables:
                self.conn.execute(f"DROP TABLE IF EXISTS {table}")
            for table, id_column in tables.items():
                self.conn.execute(f"""
                    CREATE VIRTUAL TABLE {table} USING vec0(
                        {id_column} INTEGER PRIMARY KEY,
                        embedding float[{self.config.embedding_dimensions}]
                    )
                """)

            counts = {name: 0 for name in ("facts", "semantic", "profile", "sessions", "skills")}
            for row in self.conn.execute(
                "SELECT f.id, e.name, f.predicate, f.object_text FROM facts f JOIN entities e ON e.id=f.subject_id"
            ).fetchall():
                self._replace_vector("facts_vec", "fact_id", row["id"],
                                     f"{row['name']} {row['predicate']} {row['object_text']}")
                counts["facts"] += 1
            for row in self.conn.execute("SELECT * FROM memory_semantic").fetchall():
                self._replace_vector("memory_semantic_vec", "semantic_row_id", row["id"],
                                     f"{row['kind']} {row['scope']} {row['subject']} {row['statement']} {row['tags_json']}")
                counts["semantic"] += 1
            for row in self.conn.execute("SELECT * FROM memory_profile").fetchall():
                self._replace_vector("memory_profile_vec", "profile_row_id", row["id"],
                                     f"{row['key']} {row['value']} {row['category']}")
                counts["profile"] += 1
            for row in self.conn.execute("SELECT rowid AS vector_id, * FROM memory_sessions").fetchall():
                self._replace_vector("memory_session_vec", "session_row_id", row["vector_id"],
                                     f"{row['title']} {row['summary']} {row['full_text']}")
                counts["sessions"] += 1
            for row in self.conn.execute("SELECT rowid AS vector_id, * FROM memory_skills").fetchall():
                self._replace_vector("memory_skill_vec", "skill_row_id", row["vector_id"],
                                     f"{row['name']} {row['description']} {row['triggers']} {row['category']}")
                counts["skills"] += 1

            self.conn.execute(
                "INSERT OR REPLACE INTO memory_meta(key,value,updated_at) VALUES('embedding_signature',?,datetime('now'))",
                (self.config.embedding_signature,),
            )
        except Exception:
            self.conn.execute("ROLLBACK TO embedding_reindex")
            self.conn.execute("RELEASE embedding_reindex")
            raise
        else:
            self.conn.execute("RELEASE embedding_reindex")
        return {"status": "ok", "embedding_signature": self.config.embedding_signature, "indexed": counts}

    # ═══════════════════════════════════════════════
    # UTILS
    # ═══════════════════════════════════════════════

    def _eid(self, name: str) -> str:
        return hashlib.md5(name.lower().strip().encode()).hexdigest()[:12]

    def _norm(self, s: str) -> str:
        return s.lower().strip()

    _SEARCH_STOPWORDS = {
        "the", "and", "for", "with", "from", "that", "this", "what", "when",
        "where", "who", "why", "how", "tell", "about", "please", "would", "could",
        "are", "was", "were", "have", "has", "had", "you", "your", "me", "my",
        "a", "an", "of", "to", "in", "on", "is", "it", "do", "did",
        "что", "как", "где", "когда", "кто", "почему", "про", "это", "для", "или",
        "его", "ее", "её", "мне", "мой", "моя", "мои", "расскажи", "покажи",
    }

    def _search_terms(self, text: str) -> list:
        return [
            token for token in re.findall(r"[a-zA-Zа-яА-Я0-9_]+", self._norm(text))
            if len(token) > 2 and token not in self._SEARCH_STOPWORDS
        ]

    def _ensure_entity(self, name: str, etype: str = "entity") -> str:
        name = self._norm(name)
        eid = self._eid(name)
        self.conn.execute("INSERT OR IGNORE INTO entities(id, name, type) VALUES(?,?,?)",
                          (eid, name, etype))
        return eid

    def _get_name(self, eid: str) -> str:
        row = self.conn.execute("SELECT name FROM entities WHERE id=?", (eid,)).fetchone()
        return row["name"] if row else eid

    def _resolve_entity(self, name: str):
        """Fuzzy entity lookup by name (exact → LIKE)."""
        name = self._norm(name)
        exact = self.conn.execute("SELECT id FROM entities WHERE name=?", (name,)).fetchone()
        if exact: return exact["id"]
        alias = self.conn.execute(
            "SELECT entity_id FROM entity_aliases WHERE alias=?", (name,)
        ).fetchone()
        if alias:
            return alias["entity_id"]
        fuzzy = self.conn.execute(
            "SELECT id, name FROM entities WHERE name LIKE ? LIMIT 1",
            (f"%{name}%",)).fetchone()
        return fuzzy["id"] if fuzzy else None

    def _encode(self, text: str):
        return self.embedder.encode(text, normalize_embeddings=True)

    _EMPLOYMENT_PREDICATES = {
        "works_at", "worked_at", "employed_at", "position", "работает_в",
        "works_for", "job", "role"
    }

    _RESIDENCE_PREDICATES = {
        "lives_in", "live_in", "lived_in", "resides_in", "based_in",
        "located_in", "home_city", "hometown", "from", "located_at"
    }

    _RESIDENCE_TRANSITION_PREDICATES = {
        "moved_to", "relocated_to", "moved", "relocated", "settled_in",
        "переехал_в", "переехала_в"
    }

    _POSITIVE_PREFERENCE_PREDICATES = {
        "likes", "like", "loves", "love", "enjoys", "enjoy", "prefers",
        "prefer", "favorite", "favourite", "supports", "wants", "values"
    }

    _NEGATIVE_PREFERENCE_PREDICATES = {
        "hates", "hate", "dislikes", "dislike", "avoids", "avoid",
        "detests", "detest", "opposes", "rejects", "least_favorite",
        "least_favourite"
    }

    def _same_object(self, left: str, right: str) -> bool:
        left_norm = self._norm(left)
        right_norm = self._norm(right)
        return (
            left_norm == right_norm or
            left_norm in right_norm or
            right_norm in left_norm
        )

    def _predicate_family(self, predicate: str) -> str:
        pred = self._norm(predicate)
        if pred in self._EMPLOYMENT_PREDICATES or pred in self._CURRENT_ACTIVITY_PREDICATES:
            return "employment"
        if pred in self._RESIDENCE_PREDICATES or pred in self._RESIDENCE_TRANSITION_PREDICATES:
            return "residence"
        if pred in self._POSITIVE_PREFERENCE_PREDICATES or pred in self._NEGATIVE_PREFERENCE_PREDICATES:
            return "preference"
        return pred

    def _predicate_polarity(self, predicate: str):
        pred = self._norm(predicate)
        if pred in self._POSITIVE_PREFERENCE_PREDICATES:
            return "positive"
        if pred in self._NEGATIVE_PREFERENCE_PREDICATES:
            return "negative"
        return None

    def _facts_conflict(self, old_predicate: str, old_object: str,
                        new_predicate: str, new_object: str) -> bool:
        old_pred = self._norm(old_predicate)
        new_pred = self._norm(new_predicate)
        old_obj = self._norm(old_object)
        new_obj = self._norm(new_object)

        old_family = self._predicate_family(old_pred)
        new_family = self._predicate_family(new_pred)

        if old_pred == new_pred:
            if self._same_object(old_obj, new_obj):
                return False
            return old_family in {"employment", "residence"}

        if old_family != new_family:
            return False

        if old_family in {"employment", "residence"}:
            return not self._same_object(old_obj, new_obj)

        if old_family == "preference" and self._same_object(old_obj, new_obj):
            old_polarity = self._predicate_polarity(old_pred)
            new_polarity = self._predicate_polarity(new_pred)
            return bool(old_polarity and new_polarity and old_polarity != new_polarity)

        return False

    def _find_conflicting_fact_ids(self, subject_name: str, predicate: str,
                                   new_obj: str) -> list:
        sid = self._eid(subject_name)
        active_facts = self.conn.execute(
            "SELECT id, predicate, object_text FROM facts "
            "WHERE subject_id=? AND valid_until IS NULL AND superseded_by IS NULL",
            (sid,)
        ).fetchall()

        conflicts = []
        for old in active_facts:
            if self._facts_conflict(old["predicate"], old["object_text"], predicate, new_obj):
                conflicts.append(old["id"])
        return conflicts

    def _parse_partial_date(self, value: str, default_end: bool = False):
        if not value:
            return None
        value = str(value).strip()
        patterns = [
            (r"^(\d{4})-(\d{2})-(\d{2})$", lambda y, m, d: datetime(int(y), int(m), int(d))),
            (r"^(\d{4})-(\d{2})$", lambda y, m: datetime(int(y), int(m), 28 if default_end else 1)),
            (r"^(\d{4})$", lambda y: datetime(int(y), 12 if default_end else 1, 31 if default_end else 1)),
        ]
        for pattern, builder in patterns:
            match = re.match(pattern, value)
            if not match:
                continue
            try:
                return builder(*match.groups())
            except ValueError:
                return None
        return None

    def _extract_query_timeframe(self, query_text: str) -> dict:
        norm_q = self._norm(query_text)
        years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", norm_q)]
        if years:
            start_year = min(years)
            end_year = max(years)
            return {
                "start": datetime(start_year, 1, 1),
                "end": datetime(end_year, 12, 31),
                "mode": "bounded",
                "label": f"{start_year}-{end_year}" if start_year != end_year else str(start_year)
            }

        if any(word in norm_q for word in ["сейчас", "теперь", "now", "currently", "current", "today"]):
            now = datetime.now()
            return {"start": now, "end": now, "mode": "current", "label": "current"}

        return {"start": None, "end": None, "mode": "any", "label": None}

    def _fact_matches_timeframe(self, valid_from: str, valid_until: str, timeframe: dict) -> bool:
        if timeframe.get("mode") == "any":
            return True

        fact_start = self._parse_partial_date(valid_from, default_end=False)
        fact_end = self._parse_partial_date(valid_until, default_end=True)
        query_start = timeframe.get("start")
        query_end = timeframe.get("end")

        if query_start is None or query_end is None:
            return True

        if fact_end and fact_end < query_start:
            return False
        if fact_start and fact_start > query_end:
            return False
        return True

    def _get_fact_metadata(self, entity_name: str, predicate: str, value: str):
        sid = self._resolve_entity(entity_name)
        if not sid:
            return None
        return self.conn.execute(
            "SELECT id, valid_from, valid_until, confidence, superseded_by, created_at "
            "FROM facts WHERE subject_id=? AND predicate=? AND object_text=? "
            "ORDER BY created_at DESC, id DESC LIMIT 1",
            (sid, predicate, value)
        ).fetchone()

    # ═══════════════════════════════════════════════
    # JKG 4.0: TEMPORAL EXTRACTION
    # ═══════════════════════════════════════════════

    def _extract_temporal(self, text: str) -> dict:
        """LLM extracts temporal info: reference_time, valid_from, valid_until."""
        prompt = f"""Extract temporal information from this text. Output ONLY valid JSON.

Text: {text[:2000]}

Format:
{{"reference_time": "YYYY-MM-DD or null",
 "facts": [{{"subject": "entity", "predicate": "...", "object": "...",
            "valid_from": "YYYY-MM-DD or null", "valid_until": "YYYY-MM-DD or null"}}]
}}

CRITICAL RULES:
- reference_time: when the described events happened (not when text was written)
- valid_from: when a fact became true (use exact dates when given)
- valid_until: when a fact STOPPED being true (null ONLY if still true NOW)

EXAMPLES:
- "Alice worked at Google from 2020 to 2023" →
  works_at: valid_from=2020-01-01, valid_until=2023-12-31
  NOTE: "from X to Y" means ENDED at Y — set valid_until!
- "Alice switched to Meta in 2023" →
  works_at: valid_from=2023-01-01, valid_until=null (current job until next transition)
- "Alice quit Meta and builds CopilotOS since May 2025" →
  works_at Meta: valid_until=2025-05-01 (ended when she quit)
  builds CopilotOS: valid_from=2025-05-01, valid_until=null (current)
- "Alice now works at StartupX" →
  valid_from=<extract from context or use reference_time>, valid_until=null

FOR EVERY fact that describes a job/position/role change:
- The OLD fact gets valid_until = start date of the NEW fact
- The NEW fact gets valid_from = its start date, valid_until = null
- "quitting / leaving / resigning" = the old fact ENDS

JSON:"""
        try:
            result = _llm(prompt, "Extract temporal info. Output JSON only.")
            data = json.loads(result)
            if not isinstance(data, dict) or not isinstance(data.get("facts"), list):
                raise ValueError("Invalid temporal extraction payload")
            if any(not isinstance(f, dict) for f in data["facts"]):
                raise ValueError("Invalid temporal fact")
            return data
        except Exception:
            return _fallback_extract_temporal_payload(text)

    # ═══════════════════════════════════════════════
    # JKG 4.0: FACT DEDUP + INVALIDATION
    # ═══════════════════════════════════════════════

    def _compute_fact_hash(self, subject: str, predicate: str, obj: str) -> str:
        return hashlib.md5(
            f"{self._norm(subject)}|{self._norm(predicate)}|{self._norm(obj)}".encode()
        ).hexdigest()[:16]

    def _invalidate_conflicting(self, subject_name: str, predicate: str, 
                                  new_obj: str, new_valid_from: str = None):
        """Invalidate active facts that conflict with a new assertion.

        Used by CLI/debug flows. Normal inserts call _find_conflicting_fact_ids()
        first, then _invalidate_with_edge() after the new fact is inserted.
        """
        valid_until = new_valid_from or datetime.now().strftime("%Y-%m-%d")
        conflict_ids = self._find_conflicting_fact_ids(subject_name, predicate, new_obj)
        for fact_id in conflict_ids:
            self.conn.execute(
                "UPDATE facts SET valid_until=?, forget_status='invalidated' WHERE id=?",
                (valid_until, fact_id)
            )
        return len(conflict_ids)

    # ═══════════════════════════════════════════════
    # JKG 7.1: TEMPORAL LINKING — connect temporal facts to their targets
    # ═══════════════════════════════════════════════

    # Predicate groups: when one predicate in a group appears, it should
    # set valid_until on others in the same group.
    _TEMPORAL_LINK_RULES = {
        # end_year → works_at, employed_at, position
        "end_year": ["works_at", "employed_at", "position", "worked_at", "работает_в"],
        # resigned_from → works_at (same target)
        "resigned_from": ["works_at", "employed_at"],
        # уволилась_из → works_at (same target)
        "уволилась_из": ["works_at", "employed_at"],
        "уволился_из": ["works_at", "employed_at"],
        # left → works_at
        "left": ["works_at", "employed_at"],
        # closed / ended → relevant facts
        "ended": ["works_at", "employed_at", "started"],
    }

    # When a transition happens (quit Y, build X), the old fact gets valid_until
    # and the new fact gets valid_from.
    _TRANSITION_PREDICATES = {"resigned_from", "уволилась_из", "уволился_из", "left", "quit",
                               "перешла_из", "перешёл_из", "moved_from"}

    _CURRENT_ACTIVITY_PREDICATES = {"builds", "founded", "develops", "creates", "runs",
                                     "строит", "основала", "основал", "разрабатывает",
                                     "запустила", "запустил", "работает_над"}

    def _link_temporal(self, data: dict) -> dict:
        """Post-process extracted facts: link temporal facts to their targets.

        Examples:
        - end_year=2023 → works_at=google gets valid_until=2023-12-31
        - resigned_from=meta → works_at=meta gets valid_until=<now>
        - builds=copilotos → works_at=<old> gets valid_until if builds is current activity
        """
        facts = data.get("facts", [])
        if not facts:
            return data

        # Build lookup: (subject, predicate) → list of facts
        by_subj_pred = {}
        for i, f in enumerate(facts):
            subj = self._norm(f.get("subject", ""))
            pred = self._norm(f.get("predicate", ""))
            key = (subj, pred)
            if key not in by_subj_pred:
                by_subj_pred[key] = []
            by_subj_pred[key].append(i)

        # Build lookup: (subject, object) → list of fact indices (for works_at targeting)
        by_subj_obj = {}
        for i, f in enumerate(facts):
            subj = self._norm(f.get("subject", ""))
            obj = self._norm(str(f.get("object", f.get("value", ""))))
            key = (subj, obj)
            if key not in by_subj_obj:
                by_subj_obj[key] = []
            by_subj_obj[key].append(i)

        changes = []

        # Rule 1: end_year → set valid_until on works_at for same subject
        for (subj, pred), indices in list(by_subj_pred.items()):
            if pred in self._TEMPORAL_LINK_RULES:
                target_preds = self._TEMPORAL_LINK_RULES[pred]
                year_value = facts[indices[0]].get("object",
                                         facts[indices[0]].get("value", ""))
                if not year_value:
                    continue

                # Extract year number
                year_match = re.search(r'(\d{4})', str(year_value))
                if not year_match:
                    continue
                year = year_match.group(1)
                valid_until = f"{year}-12-31"

                # Find works_at facts for same subject and set valid_until
                for tpred in target_preds:
                    tp_key = (subj, tpred)
                    if tp_key in by_subj_pred:
                        for ti in by_subj_pred[tp_key]:
                            tf = facts[ti]
                            if not tf.get("valid_until"):
                                changes.append((ti, "valid_until", valid_until))

        # Rule 2: resigned_from X → set valid_until on works_at X for same subject
        for (subj, pred), indices in list(by_subj_pred.items()):
            if pred in self._TRANSITION_PREDICATES:
                target_obj = self._norm(str(facts[indices[0]].get("object",
                                                facts[indices[0]].get("value", ""))))
                # Find works_at=<target_obj> for same subject
                for tpred in ["works_at", "employed_at", "работает_в", "position"]:
                    wk_key = (subj, target_obj)
                    if wk_key in by_subj_obj:
                        for wi in by_subj_obj[wk_key]:
                            wf = facts[wi]
                            wf_pred = self._norm(wf.get("predicate", ""))
                            if wf_pred in tpred.split() or wf_pred == tpred:
                                if not wf.get("valid_until"):
                                    # Use valid_from of the transition fact as valid_until
                                    vf = facts[indices[0]].get("valid_from")
                                    changes.append(
                                        (wi, "valid_until",
                                         vf or datetime.now().strftime("%Y-%m-%d")))

        # Rule 3: builds/founded X → set valid_until on all works_at for same subject
        for (subj, pred), indices in list(by_subj_pred.items()):
            if pred in self._CURRENT_ACTIVITY_PREDICATES:
                new_vf = facts[indices[0]].get("valid_from")
                for tpred in ["works_at", "employed_at", "работает_в"]:
                    tp_key = (subj, tpred)
                    if tp_key in by_subj_pred:
                        for ti in by_subj_pred[tp_key]:
                            tf = facts[ti]
                            if not tf.get("valid_until"):
                                changes.append(
                                    (ti, "valid_until",
                                     new_vf or datetime.now().strftime("%Y-%m-%d")))

        # Apply changes
        for idx, field, value in changes:
            facts[idx][field] = value

        return data

    # ═══════════════════════════════════════════════
    # JKG 4.0: ENTITY RESOLUTION (embedding + LLM)
    # ═══════════════════════════════════════════════

    def resolve_entity(self, name: str, etype: str = "entity") -> str:
        """Resolve entity: find existing or create. Returns entity ID.
        
        Pipeline: exact match → embedding search → LLM resolution → create new.
        """
        name = self._norm(name)

        # Step 1: Exact match
        exact = self.conn.execute("SELECT id FROM entities WHERE name=?", (name,)).fetchone()
        if exact: return exact["id"]

        # Step 2: Embedding search for candidates
        candidates = []
        if self.has_vec:
            try:
                name_vec = self._encode(name)
                # Search for similar entity names via vector
                similar = self.conn.execute(
                    "SELECT f.subject_id, e.name, e.type, distance FROM facts_vec "
                    "JOIN facts f ON facts_vec.fact_id = f.id "
                    "JOIN entities e ON f.subject_id = e.id "
                    "WHERE facts_vec.embedding MATCH ? AND k = 5 ORDER BY distance",
                    (name_vec.tobytes(), 5)
                ).fetchall()
                seen = set()
                for s in similar:
                    if s["subject_id"] not in seen and s["name"] != name:
                        seen.add(s["subject_id"])
                        candidates.append({"id": s["subject_id"], "name": s["name"],
                                          "type": s["type"], "distance": s["distance"]})
            except EmbeddingConfigurationError:
                raise
            except Exception:
                pass

        # Fallback: LIKE search
        if not candidates:
            fuzzy = self.conn.execute(
                "SELECT id, name, type FROM entities WHERE name LIKE ? LIMIT 3",
                (f"%{name}%",)
            ).fetchall()
            for f in fuzzy:
                if f["name"] != name:
                    candidates.append({"id": f["id"], "name": f["name"],
                                      "type": f["type"], "distance": 0.5})

        if not candidates:
            return self._ensure_entity(name, etype)

        # Step 3: LLM resolution
        candidate_list = "\n".join([
            f"[{i}] {c['name']} (type: {c['type']})" for i, c in enumerate(candidates)
        ])
        prompt = f"""New entity: "{name}" (type: {etype})

Existing candidates:
{candidate_list}

Is "{name}" the SAME entity as any of the candidates?
Consider: name variations, translations, abbreviations, typos.
Output ONLY JSON: {{"match": <index or null>, "reason": "..."}}
If no match, match=null.
JSON:"""
        try:
            result = _llm(prompt, "Resolve entity identity. Output JSON only.")
            resolution = json.loads(result)
            match_idx = resolution.get("match")
            if match_idx is not None and 0 <= match_idx < len(candidates):
                matched = candidates[match_idx]
                # Update summary with new name
                self.conn.execute(
                    "UPDATE entities SET summary = summary || '; aka: ' || ? WHERE id=?",
                    (name, matched["id"])
                )
                return matched["id"]
        except:
            pass

        # No match → create new
        return self._ensure_entity(name, etype)

    # ═══════════════════════════════════════════════
    # JKG 5.0: BI-TEMPORAL INVALIDATION (из Zep)
    # ═══════════════════════════════════════════════

    def _invalidate_with_edge(self, old_fact_id: int, new_fact_id: int, 
                                new_valid_from: str = None) -> int:
        """Bi-temporal: mark old fact invalid, track WHICH fact invalidated it."""
        valid_until = new_valid_from or datetime.now().strftime("%Y-%m-%d")
        self.conn.execute(
            "UPDATE facts SET valid_until=?, superseded_by=?, invalidated_by_fact_id=?, "
            "forget_status='invalidated' WHERE id=?",
            (valid_until, new_fact_id, new_fact_id, old_fact_id))
        return old_fact_id

    # ═══════════════════════════════════════════════
    # JKG 5.0: INTENTIONAL FORGETTING — Memory Utility Score
    # ═══════════════════════════════════════════════

    def _compute_utility(self, fact_id: int) -> float:
        """Memory Utility Score: access×0.3 + PageRank×0.2 + recency×0.2 + confidence×0.15 - age×0.15"""
        f = self.conn.execute(
            "SELECT f.confidence, f.access_count, f.last_accessed_at, f.created_at, "
            "f.valid_until, f.forget_status, "
            "COALESCE((SELECT COUNT(*) FROM relations r WHERE r.subject_id=f.subject_id OR r.object_id=f.subject_id), 0) as degree "
            "FROM facts f WHERE f.id=?", (fact_id,)
        ).fetchone()
        if not f: return 0.0

        access_freq = min(f["access_count"] / 10.0, 1.0)

        # Recency: days since last access
        try:
            last = datetime.fromisoformat(f["last_accessed_at"] or f["created_at"])
            days = (datetime.now() - last).days
            recency = max(0, 1.0 - days / 90.0)
        except: recency = 0.5

        # Graph centrality proxy
        degree = min(f["degree"] / 5.0, 1.0)

        confidence = f["confidence"] or 0.5

        # Age decay
        try:
            created = datetime.fromisoformat(f["created_at"])
            age_days = (datetime.now() - created).days
            age_penalty = min(age_days / 365.0, 1.0)
        except: age_penalty = 0.0

        # Invalidated facts lose utility
        if f["valid_until"] or f["forget_status"] == "invalidated":
            age_penalty += 0.5

        utility = (
            0.3 * access_freq +
            0.2 * degree +
            0.2 * recency +
            0.15 * confidence -
            0.15 * age_penalty
        )
        return max(0.0, min(1.0, utility))

    def update_utility_scores(self):
        """Recalculate utility for all active facts."""
        facts = self.conn.execute(
            "SELECT id FROM facts WHERE forget_status='active'").fetchall()
        updated = 0
        for f in facts:
            score = self._compute_utility(f["id"])
            self.conn.execute("UPDATE facts SET utility_score=? WHERE id=?", (score, f["id"]))
            updated += 1
        self.conn.commit()
        return {"updated": updated}

    def prune_memory(self, threshold: float = 0.15, dry_run: bool = True) -> dict:
        """Intentional forgetting: archive facts with utility < threshold."""
        candidates = self.conn.execute(
            "SELECT id, subject_id, predicate, object_text, utility_score, "
            "access_count, valid_until FROM facts "
            "WHERE forget_status='active' AND utility_score < ? "
            "ORDER BY utility_score ASC", (threshold,)
        ).fetchall()

        archived, deleted = 0, 0
        for c in candidates:
            # Archive if valid_until set (historical data) or low access
            if c["valid_until"] and c["access_count"] == 0:
                if not dry_run:
                    self.conn.execute(
                        "UPDATE facts SET forget_status='forgotten' WHERE id=?", (c["id"],))
                    self.conn.execute(
                        "INSERT INTO forget_log(fact_id, subject_id, predicate, object_text, "
                        "utility_score, action, reason) VALUES(?,?,?,?,?,?,?)",
                        (c["id"], c["subject_id"], c["predicate"], c["object_text"],
                         c["utility_score"], "forgotten", "low_utility_no_access"))
                deleted += 1
            else:
                if not dry_run:
                    self.conn.execute(
                        "UPDATE facts SET forget_status='archived' WHERE id=?", (c["id"],))
                    self.conn.execute(
                        "INSERT INTO forget_log(fact_id, subject_id, predicate, object_text, "
                        "utility_score, action, reason) VALUES(?,?,?,?,?,?,?)",
                        (c["id"], c["subject_id"], c["predicate"], c["object_text"],
                         c["utility_score"], "archived", "low_utility"))
                archived += 1

        if not dry_run:
            self.conn.commit()

        return {"dry_run": dry_run, "candidates": len(candidates),
                "archived": archived, "deleted": deleted, "threshold": threshold}

    # ═══════════════════════════════════════════════
    # JKG 5.0: EMOTIONAL MEMORY
    # ═══════════════════════════════════════════════

    def _detect_emotion(self, text: str) -> tuple:
        """LLM detects emotion + intensity from text."""
        prompt = f"""Analyze emotional tone. Output ONLY JSON.

Text: {text[:1000]}

Format: {{"emotion": "positive|neutral|negative|angry|sad|excited|anxious|grateful", "intensity": 0.0-1.0, "reason": "brief"}}
JSON:"""
        try:
            result = _llm(prompt, "Detect emotion. Output JSON only.")
            data = json.loads(result)
            return data.get("emotion", "neutral"), data.get("intensity", 0.0)
        except:
            return "neutral", 0.0

    def get_emotional_context(self, owner: str = "алтынай", limit: int = 10) -> list:
        """Get recent emotional episodes for adaptation."""
        rows = self.conn.execute(
            "SELECT emotion, emotion_intensity, title, reference_time "
            "FROM episodes WHERE emotion != 'neutral' "
            "ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [{"emotion": r["emotion"], "intensity": r["emotion_intensity"],
                 "title": r["title"], "time": r["reference_time"]} for r in rows]

    # ═══════════════════════════════════════════════
    # JKG 5.0: SELF-EVOLVING GRAPH
    # ═══════════════════════════════════════════════

    def evolve_schema(self, dry_run: bool = True) -> dict:
        """LLM detects patterns in graph → proposes new entity/relation types."""
        entities = self.conn.execute(
            "SELECT type, COUNT(*) as cnt FROM entities GROUP BY type ORDER BY cnt DESC"
        ).fetchall()
        relations = self.conn.execute(
            "SELECT predicate, COUNT(*) as cnt FROM relations GROUP BY predicate ORDER BY cnt DESC"
        ).fetchall()

        e_summary = "\n".join([f"  {e['type']}: {e['cnt']}" for e in entities[:20]])
        r_summary = "\n".join([f"  {r['predicate']}: {r['cnt']}" for r in relations[:20]])

        prompt = f"""Current knowledge graph schema:

Entity types:
{e_summary}

Relation types:
{r_summary}

Analyze the graph and suggest NEW entity types or relation types that would improve representation.
Focus on: missing abstractions, reusable patterns, domain-specific concepts.
Output ONLY JSON array:
[{{"type": "entity|relation", "name": "...", "description": "why needed", "evidence": "what patterns suggest this"}}]
JSON:"""
        try:
            result = _llm(prompt, "Propose schema improvements. Output JSON only.")
            proposals = json.loads(result)
        except:
            return {"proposals": [], "error": "LLM parse failed"}

        stored = 0
        for p in proposals[:10]:
            self.conn.execute(
                "INSERT INTO schema_evolution(pattern_type, pattern_description, proposed_schema, confidence) "
                "VALUES(?,?,?,?)",
                (p.get("type", "entity"), p.get("description", ""),
                 json.dumps(p, ensure_ascii=False), 0.6))
            stored += 1
        self.conn.commit()

        # Accept high-confidence proposals (2+ evidence from different sources = apply)
        return {"proposals": len(proposals), "stored": stored, "dry_run": dry_run,
                "details": proposals[:5]}

    # ═══════════════════════════════════════════════
    # JKG 5.0: PRIVACY-FIRST — GDPR deletion + audit
    # ═══════════════════════════════════════════════

    def gdpr_delete(self, entity_name: str, request_id: str = None, 
                     verified: bool = False) -> dict:
        """Delete matching data across every memory layer and retain an audit record."""
        if not verified:
            return {"status": "requires_verification",
                    "instruction": "Set verified=True after human confirmation"}

        eid = self._resolve_entity(entity_name)
        if not eid:
            return {"status": "not_found", "entity": entity_name}

        pattern = f"%{entity_name.strip()}%"
        facts = self.conn.execute(
            "SELECT id, subject_id, predicate, object_text FROM facts "
            "WHERE subject_id=? OR object_text LIKE ?",
            (eid, pattern),
        ).fetchall()
        fact_ids = [fact["id"] for fact in facts]

        # Remove vector rows and external FTS content for affected facts.
        for fact in facts:
            if self.has_vec:
                try:
                    self.conn.execute("DELETE FROM facts_vec WHERE fact_id=?", (fact["id"],))
                except Exception:
                    pass
            self.conn.execute("DELETE FROM facts_fts_content WHERE id=?", (fact["id"],))

        # Delete relations
        rel_count = self.conn.execute(
            "SELECT COUNT(*) FROM relations WHERE subject_id=? OR object_id=?", 
            (eid, eid)).fetchone()[0]
        self.conn.execute("DELETE FROM relations WHERE subject_id=? OR object_id=?", (eid, eid))

        # Preserve a sanitized fact-level audit entry before deletion.
        for fact in facts:
            audit_object = fact["object_text"]
            if entity_name.casefold() in audit_object.casefold():
                audit_object = "[redacted]"
            self.conn.execute(
                "INSERT INTO forget_log(fact_id, subject_id, predicate, object_text, "
                "utility_score, action, reason) VALUES(?,?,?,?,?,?,?)",
                (fact["id"], fact["subject_id"], fact["predicate"], audit_object, 0.0,
                 "gdpr_deleted", f"GDPR request {request_id}"))

        fact_count = len(facts)
        for fact_id in fact_ids:
            self.conn.execute(
                "UPDATE facts SET superseded_by=NULL, invalidated_by_fact_id=NULL "
                "WHERE superseded_by=? OR invalidated_by_fact_id=?",
                (fact_id, fact_id),
            )
            self.conn.execute("DELETE FROM facts WHERE id=?", (fact_id,))

        try:
            self.conn.execute("INSERT INTO facts_fts(facts_fts) VALUES('rebuild')")
        except Exception:
            pass

        self.conn.execute("DELETE FROM entities WHERE id=?", (eid,))
        self.conn.execute("UPDATE entities SET summary='' WHERE summary LIKE ?", (pattern,))

        # Audit trail
        self.conn.execute(
            "INSERT INTO deletion_log(entity_id, reason, gdpr_request_id, verified_by) "
            "VALUES(?,?,?,?)",
            (eid, "GDPR right to erasure", request_id, "system"))
            
        # Scrub every memory layer that can retain free text.
        self.conn.execute(
            "DELETE FROM episodes WHERE title LIKE ? OR body LIKE ?",
            (pattern, pattern),
        )
        self.conn.execute(
            "DELETE FROM memory_sessions WHERE title LIKE ? OR summary LIKE ? OR full_text LIKE ?",
            (pattern, pattern, pattern),
        )
        self.conn.execute(
            "DELETE FROM memory_profile WHERE key LIKE ? OR value LIKE ?",
            (pattern, pattern),
        )
        self.conn.execute(
            "DELETE FROM memory_skills WHERE name LIKE ? OR description LIKE ? OR triggers LIKE ?",
            (pattern, pattern, pattern),
        )
        self.conn.execute(
            "DELETE FROM memory_semantic WHERE subject LIKE ? OR canonical_key LIKE ? "
            "OR statement LIKE ? OR details_json LIKE ? OR tags_json LIKE ?",
            (pattern, pattern, pattern, pattern, pattern),
        )
        self.conn.execute(
            "DELETE FROM schema_evolution WHERE pattern_description LIKE ? OR proposed_schema LIKE ?",
            (pattern, pattern),
        )
        self.conn.execute(
            "UPDATE forget_log SET object_text='[redacted]' WHERE object_text LIKE ?",
            (pattern,),
        )

        self.conn.commit()
        return {"status": "deleted", "entity": entity_name, "eid": eid,
                "facts_deleted": fact_count, "relations_deleted": rel_count,
                "request_id": request_id, "gdpr_compliant": True}

    # ═══════════════════════════════════════════════
    # REMEMBER — ATOMIC graph + vector insert
    # ═══════════════════════════════════════════════

    def remember(self, text: str, source: str = "manual", ground_to: str = None,
                 title: str = "", reference_time: str = None) -> dict:
        """JKG 5.0: Extract + store with EMOTION, BI-TEMPORAL, SELF-EVOLVING."""
        import uuid as _uuid
        if not text.strip():
            raise ValueError("Memory text must not be empty")

        # ── EPISODE with EMOTION ──
        ep_uuid = str(_uuid.uuid4())
        ref_time = reference_time or datetime.now().strftime("%Y-%m-%d")
        emotion, emotion_intensity = self._detect_emotion(text)
        cur = self.conn.execute(
            "INSERT INTO episodes(uuid, title, body, source_type, reference_time, emotion, emotion_intensity) "
            "VALUES(?,?,?,?,?,?,?)",
            (ep_uuid, title or text[:80], text, source, ref_time, emotion, emotion_intensity))
        episode_id = cur.lastrowid

        # ── TEMPORAL EXTRACTION ──
        temporal = self._extract_temporal(text)
        temp_ref = temporal.get("reference_time") or ref_time
        temp_facts = {f"{self._norm(f.get('subject',''))}|{self._norm(f.get('predicate',''))}|{self._norm(str(f.get('object','')))}": f
                      for f in temporal.get("facts", [])}

        # ── ENTITY EXTRACTION ──
        entities_json = _extract_entities(text)
        try:
            data = json.loads(entities_json)
        except json.JSONDecodeError:
            data = {"entities": [], "relations": [], "facts": []}

        stored = {"entities": 0, "facts": 0, "relations": 0, "vectors": 0,
                  "invalidated": 0, "duplicates": 0, "episode_id": episode_id}
        new_fact_ids = []
        new_entity_ids = []

        # ── JKG 7.1: TEMPORAL LINKING — connect end_year/resigned_from to works_at ──
        data = self._link_temporal(data)

        with self.conn:
            # ── Entities with RESOLUTION ──
            for ent in data.get("entities", []):
                name = self._norm(ent.get("name", ent.get("entity", str(ent))))
                etype = ent.get("type", "entity")
                # Use resolve_entity for smart dedup
                eid = self.resolve_entity(name, etype)
                new_entity_ids.append(eid)
                if self.conn.total_changes > 0:
                    stored["entities"] += 1

            # ── Facts with DEDUP + TEMPORAL + INVALIDATION ──
            for fact in data.get("facts", []):
                subj = self._norm(fact.get("subject", ""))
                if not subj: continue
                sid = self.resolve_entity(subj)  # Entity resolution
                new_entity_ids.append(sid)
                pred = self._norm(fact.get("predicate", "has_property"))
                obj = self._norm(str(fact.get("object", fact.get("value", ""))))

                # Drop low-value temporal helper artifacts from generic extraction.
                # Example: store lived_in=bishkek, not lived_in_year=2023.
                if re.fullmatch(r"\d{4}", obj or "") and (
                    pred.endswith("_year") or pred in {"moved_in", "lived_in_year", "moved_to_year"}
                ):
                    if any(self._norm(tf.get("subject", "")) == subj for tf in temporal.get("facts", [])):
                        continue

                # Dedup check
                fhash = self._compute_fact_hash(subj, pred, obj)
                dup = self.conn.execute(
                    "SELECT id FROM facts WHERE fact_hash=?", (fhash,)).fetchone()
                if dup:
                    # Update access count, don't duplicate
                    self.conn.execute(
                        "UPDATE facts SET access_count = access_count + 1, "
                        "last_accessed_at = datetime('now') WHERE id=?",
                        (dup["id"],))
                    stored["duplicates"] += 1
                    continue

                # Temporal data
                tf_key = f"{subj}|{pred}|{obj}"
                tf = temp_facts.get(tf_key, {})
                vfrom = tf.get("valid_from") or temp_ref
                vuntil = tf.get("valid_until")

                # Find conflicts before insert so the new fact can explicitly supersede them
                conflicting_fact_ids = self._find_conflicting_fact_ids(subj, pred, obj)

                # Insert new fact
                cur = self.conn.execute(
                    "INSERT INTO facts(subject_id, predicate, object_text, source, "
                    "episode_id, fact_hash, valid_from, valid_until) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (sid, pred, obj, source, episode_id, fhash, vfrom, vuntil))
                fid = cur.lastrowid

                for old_fact_id in conflicting_fact_ids:
                    self._invalidate_with_edge(old_fact_id, fid, vfrom)

                new_fact_ids.append(fid)
                stored["facts"] += 1
                stored["invalidated"] += len(conflicting_fact_ids)

                # BM25 sync
                self.conn.execute(
                    "INSERT INTO facts_fts_content(id, entity_name, predicate, object_text) VALUES(?,?,?,?)",
                    (fid, self._get_name(sid), pred, obj))

            # ── Relations ──
            for rel in data.get("relations", []):
                subj = self._norm(rel.get("subject", ""))
                obj = self._norm(rel.get("object", ""))
                if not subj or not obj: continue
                sid = self.resolve_entity(subj)
                oid = self.resolve_entity(obj)
                pred = rel.get("predicate", "related_to")
                cursor = self.conn.execute(
                    "INSERT INTO relations(subject_id, predicate, object_id, source) "
                    "SELECT ?,?,?,? WHERE NOT EXISTS ("
                    "SELECT 1 FROM relations WHERE subject_id=? AND predicate=? AND object_id=?)",
                    (sid, pred, oid, source, sid, pred, oid))
                stored["relations"] += cursor.rowcount

            # Auto-ground
            if ground_to:
                gid = self._eid(self._norm(ground_to))
                for eid in set(new_entity_ids):
                    if eid != gid:
                        self.conn.execute(
                            "INSERT OR IGNORE INTO relations(subject_id, predicate, object_id, source) VALUES(?,?,?,?)",
                            (gid, "connected_to", eid, "auto-ground"))

            # VECTORS
            if self.has_vec and new_fact_ids:
                for fid in new_fact_ids:
                    try:
                        fact = self.conn.execute(
                            "SELECT f.predicate, f.object_text, e.name FROM facts f "
                            "JOIN entities e ON f.subject_id = e.id WHERE f.id=?",
                            (fid,)).fetchone()
                        if fact:
                            fact_text = f"{fact['name']} {fact['predicate']} {fact['object_text']}"
                            vec = self._encode(fact_text)
                            self.conn.execute(
                                "INSERT INTO facts_vec(fact_id, embedding) VALUES(?,?)",
                                (fid, vec.tobytes()))
                            stored["vectors"] += 1
                    except EmbeddingConfigurationError:
                        raise
                    except Exception:
                        raise

            # Rebuild FTS
            try:
                self.conn.execute("INSERT INTO facts_fts(facts_fts) VALUES('rebuild')")
            except Exception:
                raise

        stored["extraction_status"] = "structured" if stored["facts"] or stored["relations"] else (
            "duplicate" if stored["duplicates"] else "raw_only"
        )
        if stored["extraction_status"] == "raw_only":
            stored["warning"] = "Text saved as an episode; no structured facts were recognized."
        return stored

    # ═══════════════════════════════════════════════
    # GRAPH TRAVERSAL
    # ═══════════════════════════════════════════════

    def _get_outgoing(self, eid: str) -> list:
        edges = []
        for f in self.conn.execute(
            "SELECT predicate, object_text FROM facts WHERE subject_id = ?", (eid,)
        ).fetchall():
            edges.append({"from_id": eid, "predicate": f["predicate"],
                          "to_text": f["object_text"], "edge_type": "fact"})
        for r in self.conn.execute(
            "SELECT r.predicate, r.object_id, e.name FROM relations r "
            "JOIN entities e ON r.object_id = e.id WHERE r.subject_id = ?", (eid,)
        ).fetchall():
            edges.append({"from_id": eid, "to_id": r["object_id"],
                          "predicate": r["predicate"], "to_text": r["name"],
                          "edge_type": "relation"})
        for r in self.conn.execute(
            "SELECT r.subject_id, r.predicate, e.name FROM relations r "
            "JOIN entities e ON r.subject_id = e.id WHERE r.object_id = ?", (eid,)
        ).fetchall():
            edges.append({"from_id": r["subject_id"], "to_id": eid,
                          "predicate": r["predicate"], "from_text": r["name"],
                          "edge_type": "relation_in"})
        return edges

    def traverse(self, entity_name: str, depth: int = 3, max_results: int = 50) -> dict:
        eid = self._resolve_entity(entity_name)
        if not eid:
            return {"entity": entity_name, "facts": [], "paths": [], "error": "not found"}

        visited = {eid}
        queue = deque([(eid, 0, [])])
        all_facts = []
        all_paths = []

        while queue and len(all_facts) < max_results:
            cur, d, path = queue.popleft()
            if d >= depth: continue

            for edge in self._get_outgoing(cur):
                if edge["edge_type"] == "fact":
                    all_facts.append({
                        "entity": self._get_name(cur),
                        "predicate": edge["predicate"],
                        "value": edge["to_text"],
                        "depth": d, "path": path,
                    })
                else:
                    neighbor = edge.get("to_id") or edge.get("from_id")
                    nname = edge.get("to_text") or edge.get("from_text")
                    if neighbor and neighbor not in visited:
                        visited.add(neighbor)
                        new_path = path + [
                            f"{self._get_name(cur)} --[{edge['predicate']}]--> {nname}"]
                        all_paths.append(new_path[-1])
                        queue.append((neighbor, d + 1, new_path))

        return {"entity": self._get_name(eid), "facts": all_facts,
                "paths": all_paths, "facts_count": len(all_facts)}

    def find_path(self, frm: str, to: str, max_depth: int = 5) -> dict:
        fid = self._resolve_entity(frm)
        tid = self._resolve_entity(to)
        if not fid or not tid:
            return {"from": frm, "to": to, "path": None, "error": "not found"}
        if fid == tid:
            return {"from": frm, "to": to, "path": ["same"], "length": 0}

        visited = {fid}
        queue = deque([(fid, [self._get_name(fid)])])

        while queue:
            eid, path = queue.popleft()
            if len(path) > max_depth + 1: continue
            for edge in self._get_outgoing(eid):
                neighbor = edge.get("to_id") or edge.get("from_id")
                if neighbor == tid:
                    path.append(f"--[{edge['predicate']}]--> {self._get_name(tid)}")
                    return {"from": self._get_name(fid), "to": self._get_name(tid),
                            "path": path, "length": len(path) - 1}
                if neighbor and neighbor not in visited:
                    visited.add(neighbor)
                    nname = edge.get("to_text") or edge.get("from_text")
                    queue.append((neighbor, path + [
                        f"--[{edge['predicate']}]--> {nname}"]))

        return {"from": frm, "to": to, "path": None, "error": "no path"}

    # ═══════════════════════════════════════════════
    # PAGERANK
    # ═══════════════════════════════════════════════

    def pagerank(self, damping=0.85, iterations=20) -> dict:
        """Compute PageRank for all entities. Returns {eid: score}."""
        entities = self.conn.execute("SELECT id FROM entities").fetchall()
        N = len(entities)
        if N == 0: return {}

        scores = {e["id"]: 1.0 for e in entities}
        out_degree = {}
        for r in self.conn.execute(
            "SELECT subject_id, COUNT(*) as cnt FROM relations GROUP BY subject_id"
        ).fetchall():
            out_degree[r["subject_id"]] = r["cnt"]

        for _ in range(iterations):
            new_scores = {}
            for eid in scores:
                incoming = self.conn.execute(
                    "SELECT subject_id FROM relations WHERE object_id=?", (eid,)
                ).fetchall()
                rank_sum = sum(
                    scores[r["subject_id"]] / max(out_degree.get(r["subject_id"], 1), 1)
                    for r in incoming)
                new_scores[eid] = (1 - damping) / N + damping * rank_sum
            scores = new_scores

        return scores

    # ═══════════════════════════════════════════════
    # HYBRID RECALL — BM25 + KNN + Graph → RRF
    # ═══════════════════════════════════════════════

    def recall(self, query: str, limit: int = 20) -> list:
        """Hybrid recall: BM25 + vector KNN + graph expansion → RRF fusion."""
        results_lists = []

        # ── Method 1: BM25 keyword search ──
        try:
            bm25_rows = self.conn.execute(
                "SELECT rowid as id, entity_name, predicate, object_text "
                "FROM facts_fts WHERE facts_fts MATCH ? LIMIT ?",
                (self._fts_query(query), limit * 3)
            ).fetchall()
            results_lists.append([
                {"fact_id": r["id"], "entity": r["entity_name"],
                 "predicate": r["predicate"], "value": r["object_text"]}
                for r in bm25_rows
            ])
        except Exception:
            pass

        # ── Method 2: Vector KNN search ──
        if self.has_vec:
            try:
                query_vec = self._encode(query)
                vec_rows = self.conn.execute(
                    "SELECT fact_id, distance FROM facts_vec "
                    "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                    (query_vec.tobytes(), min(limit * 3, 50))
                ).fetchall()
                vec_fact_ids = [r["fact_id"] for r in vec_rows]
                if vec_fact_ids:
                    vec_rank = {fact_id: rank for rank, fact_id in enumerate(vec_fact_ids)}
                    vec_distance = {r["fact_id"]: float(r["distance"]) for r in vec_rows}
                    placeholders = ",".join(["?"] * len(vec_fact_ids))
                    fact_data = self.conn.execute(
                        f"SELECT f.id, e.name, f.predicate, f.object_text FROM facts f "
                        f"JOIN entities e ON f.subject_id = e.id "
                        f"WHERE f.id IN ({placeholders})",
                        vec_fact_ids
                    ).fetchall()
                    fact_data = sorted(fact_data, key=lambda fact: vec_rank[fact["id"]])
                    results_lists.append([
                        {"fact_id": f["id"], "entity": f["name"],
                         "predicate": f["predicate"], "value": f["object_text"],
                         "vector_distance": vec_distance[f["id"]]}
                        for f in fact_data
                    ])
            except EmbeddingConfigurationError:
                raise
            except Exception:
                pass

        # ── Method 3: Graph expansion from top matched entities ──
        graph_facts = self.recall_graph(query, limit=limit)

        # ── RRF FUSION ──
        return self._rrf_fusion(results_lists, graph_facts, limit)

    def _fts_query(self, query: str) -> str:
        """Convert user query to FTS5 query string."""
        words = self._search_terms(query)
        return " OR ".join(f'"{w}"' for w in words) if words else query

    def _keyword_overlap_bonus(self, query: str, entity: str, predicate: str, value: str) -> float:
        """Small lexical bonus/penalty to keep semantic retrieval anchored to the query."""
        q_tokens = set(self._search_terms(query))
        c_tokens = set(self._search_terms(f"{entity} {predicate} {value}"))
        if not q_tokens or not c_tokens:
            return 0.0

        overlap = len(q_tokens & c_tokens)
        bonus = overlap * 0.08

        q = self._norm(query)
        c = self._norm(f"{entity} {predicate} {value}")

        food_terms = {"recipe", "pie", "pies", "fruit", "food", "dessert", "рецепт", "пирог", "еда"}
        company_terms = {"company", "technology", "record", "label", "founded", "jobs", "beatles"}
        if any(term in q for term in food_terms):
            if any(term in c for term in food_terms):
                bonus += 0.35
            if any(term in c for term in company_terms):
                bonus -= 0.50  # Strong penalty — food query + company context = noise

        if any(term in q for term in {"where", "live", "lived", "moved", "location", "где", "живет", "жил", "переех"}):
            if predicate.endswith("_year") or (re.fullmatch(r"\d{4}", str(value or "")) is not None):
                bonus -= 0.22
            if predicate in {"lived_in", "lives_in", "moved_to", "located_in"}:
                bonus += 0.18

        return bonus

    def recall_graph(self, query: str, limit: int = 20) -> list:
        """Graph traversal recall from keyword-matched entities."""
        words = self._search_terms(query)
        if not words: return []

        conditions = " OR ".join(["name LIKE ?" for _ in words])
        params = [f"%{w}%" for w in words]
        matched = self.conn.execute(
            f"SELECT id, name FROM entities WHERE {conditions} LIMIT 5",
            params).fetchall()

        all_facts = {}
        for m in matched:
            trav = self.traverse(m["name"], depth=2, max_results=10)
            for f in trav.get("facts", []):
                key = f"{f['entity']}|{f['predicate']}|{f['value']}"
                if key not in all_facts:
                    all_facts[key] = f

        return list(all_facts.values())[:limit]

    def _rrf_fusion(self, lists: list, graph_facts: list, limit: int, k: int = 60) -> list:
        """Reciprocal Rank Fusion — объединяет N ранжированных списков."""
        scores = defaultdict(float)

        for lst in lists:
            for rank, item in enumerate(lst):
                key = f"{item.get('entity','')}|{item.get('predicate','')}|{item.get('value','')}"
                scores[key] += 1.0 / (k + rank + 1)

        # Also include graph facts
        for rank, item in enumerate(graph_facts):
            key = f"{item['entity']}|{item['predicate']}|{item['value']}"
            scores[key] += 1.0 / (k + rank + 1)

        # Sort and deduplicate
        seen = set()
        result = []
        for key, score in sorted(scores.items(), key=lambda x: -x[1]):
            parts = key.split("|", 2)
            if len(parts) == 3:
                entity, predicate, value = parts
                sig = f"{entity}|{predicate}|{value}"
                if sig not in seen:
                    seen.add(sig)
                    result.append({
                        "entity": entity, "predicate": predicate,
                        "value": value, "rrf_score": round(score, 4),
                    })
                    if len(result) >= limit: break

        return result

    # ═══════════════════════════════════════════════
    # JKG 6.0: CLARK — Confidence-Layered Adaptive Retrieval for Knowledge
    # Inspired by Clark's Nutcracker spatial memory.
    # Stage 1: graph signals → Stage 2: hybrid best-first retrieval → Stage 3: access tracking
    # ═══════════════════════════════════════════════

    def propagate_confidence(self, iterations: int = 5, lambda_factor: float = 0.7) -> dict:
        """Stage 1: Value Iteration for belief propagation through the graph.
        
        Each fact's confidence is iteratively updated based on its neighbors.
        Converges to the stationary distribution of the Markov random field.
        
        Recurrence:
          V_new[f] = λ × original_confidence[f] + (1-λ) × mean(V[neighbors of f's entities])
        
        Complexity: O(iterations × |F| × d), d = avg entity degree.
        """
        # ── Schema detection: check for forget_status, confidence ──
        fact_cols = [c[1] for c in self.conn.execute("PRAGMA table_info(facts)").fetchall()]
        has_forget = 'forget_status' in fact_cols
        has_confidence = 'confidence' in fact_cols
        
        if not has_confidence:
            return {"iterations": 0, "updated": 0, "message": "no confidence column — run migration first"}
        
        # ── Load all facts with current confidences ──
        if has_forget:
            facts = self.conn.execute("""
                SELECT f.id, f.subject_id, f.confidence, f.valid_until,
                       e.name as entity_name
                FROM facts f JOIN entities e ON f.subject_id = e.id
                WHERE f.forget_status = 'active' AND f.confidence > 0
            """).fetchall()
        else:
            facts = self.conn.execute("""
                SELECT f.id, f.subject_id, f.confidence, f.valid_until,
                       e.name as entity_name
                FROM facts f JOIN entities e ON f.subject_id = e.id
                WHERE f.confidence > 0
            """).fetchall()
        
        if not facts:
            return {"iterations": 0, "updated": 0, "message": "no facts to propagate"}
        
        N = len(facts)
        orig_confidence = {f["id"]: f["confidence"] for f in facts}
        V = dict(orig_confidence)
        
        # ── Precompute entity→fact_ids mapping ──
        entity_facts = defaultdict(list)
        for f in facts:
            entity_facts[f["subject_id"]].append(f["id"])
        
        # ── Precompute neighboring entity pairs for each entity ──
        # Entities are neighbors if they share a relation edge
        entity_neighbors = defaultdict(set)
        for rel in self.conn.execute(
            "SELECT subject_id, object_id FROM relations"
        ).fetchall():
            entity_neighbors[rel["subject_id"]].add(rel["object_id"])
            entity_neighbors[rel["object_id"]].add(rel["subject_id"])
        
        # ── Value Iteration ──
        for it in range(iterations):
            V_new = {}
            max_delta = 0.0
            
            for f in facts:
                fid = f["id"]
                eid = f["subject_id"]
                
                # Gather confidence of facts belonging to neighboring entities
                neighbor_confidences = []
                for neighbor_eid in entity_neighbors.get(eid, set()):
                    for nfid in entity_facts.get(neighbor_eid, []):
                        neighbor_confidences.append(V.get(nfid, orig_confidence.get(nfid, 0.5)))
                
                # If no neighbors, keep original
                if neighbor_confidences:
                    neighbor_mean = sum(neighbor_confidences) / len(neighbor_confidences)
                else:
                    neighbor_mean = orig_confidence[fid]
                
                V_new[fid] = lambda_factor * orig_confidence[fid] + (1 - lambda_factor) * neighbor_mean
                delta = abs(V_new[fid] - V.get(fid, orig_confidence[fid]))
                if delta > max_delta:
                    max_delta = delta
            
            V = V_new
            
            # ── Early convergence ──
            if max_delta < 0.001:
                break
        
        # ── Write back to DB ──
        updated = 0
        for fid, conf in V.items():
            if abs(conf - orig_confidence.get(fid, 0.5)) > 0.001:
                self.conn.execute(
                    "UPDATE facts SET confidence = MIN(1.0, ?) WHERE id = ?",
                    (round(conf, 4), fid))
                updated += 1
        self.conn.commit()
        
        return {
            "iterations": it + 1, "updated": updated,
            "total_facts": N, "converged": max_delta < 0.001,
            "avg_confidence": round(sum(V.values()) / N, 4) if N else 0
        }

    def _landmark_selection(self, limit: int = 20, include_historical: bool = False) -> list:
        """Select top entities as 'landmarks' by combined PageRank × avg confidence.
        
        Returns list of {eid, name, score, fact_ids}.
        """
        pr = self.pagerank()
        if not pr:
            return []
        
        # Schema detection
        fact_cols = [c[1] for c in self.conn.execute("PRAGMA table_info(facts)").fetchall()]
        has_forget = 'forget_status' in fact_cols
        
        query = "SELECT id, confidence FROM facts WHERE subject_id=?"
        if has_forget and not include_historical:
            query += " AND forget_status='active'"
        
        scored = []
        for e in self.conn.execute("SELECT id, name FROM entities").fetchall():
            eid = e["id"]
            facts = self.conn.execute(query, (eid,)).fetchall()
            if not facts:
                continue
            avg_conf = sum(f["confidence"] for f in facts) / len(facts)
            pr_score = pr.get(eid, 1.0 / max(len(pr), 1))
            # Combined score: PageRank × confidence × log(fact_count)
            combined = pr_score * avg_conf * (1 + math.log(len(facts) + 1))
            scored.append({
                "eid": eid, "name": e["name"],
                "score": combined,
                "fact_ids": [f["id"] for f in facts],
                "fact_count": len(facts),
                "pagerank": pr_score,
                "avg_confidence": avg_conf,
            })
        
        scored.sort(key=lambda x: -x["score"])
        return scored[:limit]

    def _clark_astar_entities(self, seed_costs: dict, max_nodes: int = 50,
                              max_depth: int = 3) -> dict:
        """Run multi-source A* from a virtual query node over entity relations.

        Query-to-seed and relation costs are non-negative negative-log
        probabilities. The heuristic is zero, which is admissible and consistent;
        therefore every entity popped from the queue has its optimal path cost.
        This is A* in its uniform-cost special case and favors reliable paths over
        merely short ones.
        """
        if not seed_costs:
            return {}

        frontier = []
        best_cost = {}
        best_depth = {}
        parent = {}
        for entity_id, initial_cost in seed_costs.items():
            cost = max(0.0, float(initial_cost))
            if cost < best_cost.get(entity_id, float("inf")):
                best_cost[entity_id] = cost
                best_depth[entity_id] = 0
                parent[entity_id] = None
                heapq.heappush(frontier, (cost, cost, 0, entity_id))

        settled = {}
        while frontier and len(settled) < max_nodes:
            _f_score, g_score, depth, entity_id = heapq.heappop(frontier)
            if entity_id in settled or g_score > best_cost.get(entity_id, float("inf")) + 1e-12:
                continue
            settled[entity_id] = g_score
            if depth >= max_depth:
                continue

            rows = self.conn.execute("""
                SELECT subject_id, object_id, confidence
                FROM relations WHERE subject_id=? OR object_id=?
            """, (entity_id, entity_id)).fetchall()
            for row in rows:
                neighbor = row["object_id"] if row["subject_id"] == entity_id else row["subject_id"]
                reliability = max(1e-6, min(1.0, float(row["confidence"] or 0.5)))
                tentative = g_score - math.log(reliability)
                next_depth = depth + 1
                if tentative + 1e-12 >= best_cost.get(neighbor, float("inf")):
                    continue
                best_cost[neighbor] = tentative
                best_depth[neighbor] = next_depth
                parent[neighbor] = entity_id
                heuristic = 0.0
                heapq.heappush(frontier, (tentative + heuristic, tentative, next_depth, neighbor))

        names = {
            row["id"]: row["name"] for row in self.conn.execute(
                f"SELECT id, name FROM entities WHERE id IN ({','.join('?' for _ in settled)})",
                tuple(settled),
            ).fetchall()
        } if settled else {}
        result = {}
        for entity_id, cost in settled.items():
            path_ids = []
            cursor = entity_id
            while cursor is not None:
                path_ids.append(cursor)
                cursor = parent.get(cursor)
            path_ids.reverse()
            result[entity_id] = {
                "cost": cost,
                "depth": best_depth.get(entity_id, 0),
                "path_ids": path_ids,
                "path": [names.get(node_id, node_id) for node_id in path_ids],
                "path_reliability": math.exp(-cost),
            }
        return result

    def retrieve_clark(self, query: str, top_k: int = 10, beam_width: int = 5) -> list:
        """CLARK retrieval: landmarks, optimal A* graph paths, and fact ranking."""
        if not self.has_vec:
            return self.recall(query, limit=top_k)

        query_vec = self._encode(query)
        timeframe = self._extract_query_timeframe(query)
        include_historical = timeframe.get("mode") == "bounded"

        candidate_limit = max(top_k * 8, 50)
        vector_scores = {}
        seed_fact_ids = set()
        seed_relevance = {}

        try:
            rows = self.conn.execute(
                "SELECT fact_id, distance FROM facts_vec "
                "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (query_vec.tobytes(), min(candidate_limit, 100)),
            ).fetchall()
            for row in rows:
                distance = max(0.0, float(row["distance"]))
                vector_scores[row["fact_id"]] = self._distance_similarity(distance)
                seed_fact_ids.add(row["fact_id"])
                seed_relevance[row["fact_id"]] = vector_scores[row["fact_id"]]
        except Exception:
            pass

        try:
            rows = self.conn.execute(
                "SELECT rowid AS fact_id FROM facts_fts WHERE facts_fts MATCH ? LIMIT ?",
                (self._fts_query(query), candidate_limit),
            ).fetchall()
            for rank, row in enumerate(rows):
                seed_fact_ids.add(row["fact_id"])
                seed_relevance[row["fact_id"]] = max(
                    seed_relevance.get(row["fact_id"], 0.0), 1.0 / (rank + 1)
                )
        except Exception:
            pass

        if not seed_fact_ids:
            return self.recall(query, limit=top_k)

        placeholders = ",".join("?" for _ in seed_fact_ids)
        seed_entities = self.conn.execute(
            f"SELECT id, subject_id FROM facts WHERE id IN ({placeholders})",
            tuple(seed_fact_ids),
        ).fetchall()
        seed_costs = {}
        for row in seed_entities:
            relevance = max(1e-6, min(1.0, seed_relevance.get(row["id"], 0.5)))
            cost = -math.log(relevance)
            seed_costs[row["subject_id"]] = min(seed_costs.get(row["subject_id"], float("inf")), cost)

        astar = self._clark_astar_entities(
            seed_costs, max_nodes=max(25, beam_width * 10), max_depth=3
        )
        entity_ids = list(astar)
        entity_placeholders = ",".join("?" for _ in entity_ids)
        active_clause = "" if include_historical else (
            "AND f.forget_status='active' AND f.superseded_by IS NULL "
            "AND (f.valid_until IS NULL OR f.valid_until >= date('now')) "
            "AND (f.valid_from IS NULL OR f.valid_from <= date('now'))"
        )
        facts = self.conn.execute(
            f"SELECT f.id, f.subject_id, f.predicate, f.object_text, f.confidence, "
            f"f.valid_from, f.valid_until, e.name AS entity_name "
            f"FROM facts f JOIN entities e ON e.id=f.subject_id "
            f"WHERE f.subject_id IN ({entity_placeholders}) {active_clause} "
            f"LIMIT ?",
            (*entity_ids, candidate_limit * 2),
        ).fetchall()

        query_tokens = {
            token for token in re.findall(r"[a-zA-Zа-яА-Я0-9_]+", self._norm(query))
            if len(token) > 2
        }
        missing_vectors = [
            fact for fact in facts if fact["id"] not in vector_scores
        ]
        if missing_vectors:
            texts = [
                f"{fact['entity_name']} {fact['predicate']} {fact['object_text']}"
                for fact in missing_vectors
            ]
            try:
                vectors = self.embedder.encode(texts, normalize_embeddings=True)
                query_array = np.asarray(query_vec, dtype=np.float32).reshape(-1)
                for fact, vector in zip(missing_vectors, vectors):
                    fact_array = np.asarray(vector, dtype=np.float32).reshape(-1)
                    denominator = np.linalg.norm(query_array) * np.linalg.norm(fact_array)
                    similarity = float(np.dot(query_array, fact_array) / denominator) if denominator else 0.0
                    vector_scores[fact["id"]] = max(0.0, min(1.0, similarity))
            except EmbeddingConfigurationError:
                raise
            except Exception:
                pass

        ranked = []
        now = datetime.now()
        for fact in facts:
            candidate_tokens = {
                token for token in re.findall(
                    r"[a-zA-Zа-яА-Я0-9_]+",
                    self._norm(f"{fact['entity_name']} {fact['predicate']} {fact['object_text']}")
                ) if len(token) > 2
            }
            union = query_tokens | candidate_tokens
            lexical = len(query_tokens & candidate_tokens) / len(union) if union else 0.0
            lexical += self._keyword_overlap_bonus(
                query, fact["entity_name"], fact["predicate"], fact["object_text"]
            )
            lexical = max(0.0, min(1.0, lexical))

            temporal = 1.0
            if fact["valid_until"]:
                fact_end = self._parse_partial_date(fact["valid_until"], default_end=True)
                temporal = 0.2 if fact_end and fact_end < now else 0.8
            if timeframe.get("mode") != "any":
                temporal = 1.0 if self._fact_matches_timeframe(
                    fact["valid_from"], fact["valid_until"], timeframe
                ) else 0.0

            path_info = astar.get(fact["subject_id"], {})
            graph_proximity = float(path_info.get("path_reliability", 0.0))
            similarity = max(0.0, min(1.0, vector_scores.get(fact["id"], 0.0)))
            confidence = max(0.0, min(1.0, float(fact["confidence"] or 0.5)))
            score = (
                0.45 * similarity +
                0.25 * lexical +
                0.15 * confidence +
                0.10 * temporal +
                0.05 * graph_proximity
            )
            ranked.append({
                "fact_id": fact["id"],
                "entity": fact["entity_name"],
                "predicate": fact["predicate"],
                "value": fact["object_text"],
                "clark_score": round(max(0.0, min(1.0, score)), 4),
                "confidence": round(confidence, 4),
                "cosine_sim": round(similarity, 4),
                "lexical_score": round(lexical, 4),
                "temporal_bonus": round(temporal, 4),
                "graph_depth": path_info.get("depth", 0),
                "graph_path": path_info.get("path", [fact["entity_name"]]),
                "astar_cost": round(float(path_info.get("cost", 0.0)), 6),
            })

        ranked.sort(key=lambda item: item["clark_score"], reverse=True)
        return ranked[:top_k]

    def _clark_confidence_update(self, retrieved: list, boost: float = 0.05):
        """Record retrieval without mutating truth confidence.

        Kept under its historical private name for compatibility. Confidence
        represents evidence quality; salience and access history represent use.
        """
        if not retrieved:
            return 0
        
        updated = 0
        for item in retrieved:
            fact_id = item.get("fact_id")
            if not fact_id:
                continue
            self.conn.execute(
                "UPDATE facts SET access_count=access_count+1, last_accessed_at=datetime('now'), "
                "salience=MIN(1.0, salience + ?) WHERE id=?",
                (boost, fact_id),
            )
            self.conn.execute(
                "INSERT INTO memory_access(memory_layer,memory_id) VALUES('factual',?)",
                (str(fact_id),),
            )
            updated += 1
        self.conn.commit()
        return updated

    def add_entity_alias(self, entity: str, alias: str, confidence: float = 1.0,
                         source: str = "manual") -> dict:
        entity_id = self._resolve_entity(entity)
        if not entity_id:
            entity_id = self._ensure_entity(entity)
        normalized = self._norm(alias)
        self.conn.execute(
            "INSERT OR REPLACE INTO entity_aliases(alias,entity_id,confidence,source) VALUES(?,?,?,?)",
            (normalized, entity_id, max(0.0, min(1.0, confidence)), source),
        )
        self.conn.commit()
        return {"status": "ok", "entity_id": entity_id, "alias": normalized}

    def merge_entities(self, source: str, target: str) -> dict:
        """Merge a duplicate entity into a canonical entity, preserving an alias."""
        source_id = self._resolve_entity(source)
        target_id = self._resolve_entity(target)
        if not source_id or not target_id:
            return {"status": "not_found", "source": source, "target": target}
        if source_id == target_id:
            return {"status": "noop", "entity_id": target_id}
        self.conn.execute("UPDATE facts SET subject_id=? WHERE subject_id=?", (target_id, source_id))
        self.conn.execute("UPDATE relations SET subject_id=? WHERE subject_id=?", (target_id, source_id))
        self.conn.execute("UPDATE relations SET object_id=? WHERE object_id=?", (target_id, source_id))
        self.conn.execute("UPDATE entity_aliases SET entity_id=? WHERE entity_id=?", (target_id, source_id))
        self.conn.execute(
            "INSERT OR REPLACE INTO entity_aliases(alias,entity_id,source) VALUES(?,?,?)",
            (self._norm(source), target_id, "entity-merge"),
        )
        self.conn.execute("DELETE FROM entities WHERE id=?", (source_id,))
        self.conn.commit()
        return {"status": "merged", "source_id": source_id, "target_id": target_id}

    def feedback(self, layer: str, memory_id, signal: str, note: str = "") -> dict:
        """Attach explicit user feedback without rewriting source memory."""
        allowed = {item.value for item in FeedbackSignal}
        if signal not in allowed:
            raise ValueError(f"signal must be one of: {', '.join(sorted(allowed))}")
        self.conn.execute(
            "INSERT INTO memory_feedback(memory_layer,memory_id,signal,note) VALUES(?,?,?,?)",
            (layer, str(memory_id), signal, note[:1000]),
        )
        if layer == "factual" and signal in {"confirm", "useful"}:
            self.conn.execute("UPDATE facts SET salience=MIN(1.0,salience+0.1) WHERE id=?", (memory_id,))
        elif layer == "factual" and signal in {"reject", "not_useful"}:
            self.conn.execute("UPDATE facts SET salience=MAX(0.0,salience-0.2) WHERE id=?", (memory_id,))
        self.conn.commit()
        return {"status": "recorded", "layer": layer, "memory_id": str(memory_id), "signal": signal}

    def consolidate(self, dry_run: bool = True) -> dict:
        """Find safe maintenance work; optionally remove only true orphan entities."""
        orphans = self.conn.execute("""
            SELECT id, name FROM entities e
            WHERE NOT EXISTS (SELECT 1 FROM facts f WHERE f.subject_id=e.id)
              AND NOT EXISTS (SELECT 1 FROM relations r WHERE r.subject_id=e.id OR r.object_id=e.id)
              AND NOT EXISTS (SELECT 1 FROM entity_aliases a WHERE a.entity_id=e.id)
        """).fetchall()
        duplicate_aliases = self.conn.execute("""
            SELECT lower(name) AS normalized, COUNT(*) AS count
            FROM entities GROUP BY lower(name) HAVING COUNT(*) > 1
        """).fetchall()
        if not dry_run and orphans:
            self.conn.executemany("DELETE FROM entities WHERE id=?", [(row["id"],) for row in orphans])
            self.conn.commit()
        return {
            "dry_run": dry_run,
            "orphan_entities": [dict(row) for row in orphans],
            "duplicate_entity_groups": [dict(row) for row in duplicate_aliases],
            "deleted": 0 if dry_run else len(orphans),
        }
    # ═══════════════════════════════════════════════
    # BIDIRECTIONAL BRIDGE
    # ═══════════════════════════════════════════════

    def bridge_graph_to_embeddings(self) -> dict:
        """Graph enriches embeddings: PageRank → boost confidence."""
        if not self.has_vec: return {"error": "vec0 not available"}

        pr = self.pagerank()
        boosted = 0
        for eid, pr_score in pr.items():
            if pr_score < 0.01: continue
            facts = self.conn.execute(
                "SELECT id FROM facts WHERE subject_id=?", (eid,)
            ).fetchall()
            for f in facts:
                self.conn.execute(
                    "UPDATE facts SET confidence = MIN(1.0, ?) WHERE id=?",
                    (0.5 + pr_score * 0.3, f["id"]))
                boosted += 1
        self.conn.commit()
        return {"pagerank_entities": len(pr), "boosted_facts": boosted}

    def bridge_embeddings_to_graph(self, threshold: float = 0.85, max_new_links: int = 20) -> dict:
        """Embeddings suggest new graph links: similar entities → proposed relations."""
        if not self.has_vec: return {"error": "vec0 not available"}

        entities = self.conn.execute("SELECT id, name FROM entities").fetchall()
        new_links = 0

        for i, e1 in enumerate(entities):
            if new_links >= max_new_links: break
            # Get embedding for entity (average of its facts)
            facts = self.conn.execute(
                "SELECT predicate, object_text FROM facts WHERE subject_id=?", (e1["id"],)
            ).fetchall()
            if not facts: continue
            e1_text = f"{e1['name']} " + " ".join(
                f"{f['predicate']} {f['object_text']}" for f in facts[:5])
            e1_vec = self._encode(e1_text)

            # Find similar via vec0
            try:
                similar = self.conn.execute(
                    "SELECT fact_id, distance FROM facts_vec "
                    "WHERE embedding MATCH ? AND k = 5 ORDER BY distance",
                    (e1_vec.tobytes(), 5)
                ).fetchall()
            except Exception:
                continue

            for s in similar:
                if new_links >= max_new_links: break
                sf = self.conn.execute(
                    "SELECT f.subject_id, e.name FROM facts f "
                    "JOIN entities e ON f.subject_id = e.id WHERE f.id=?",
                    (s["fact_id"],)
                ).fetchone()
                if not sf or sf["subject_id"] == e1["id"]: continue

                sim = 1.0 - min(s["distance"], 2.0) / 2.0
                if sim < threshold: continue

                # Check if relation already exists
                existing = self.conn.execute(
                    "SELECT 1 FROM relations WHERE "
                    "(subject_id=? AND object_id=?) OR (subject_id=? AND object_id=?)",
                    (e1["id"], sf["subject_id"], sf["subject_id"], e1["id"])
                ).fetchone()

                if not existing:
                    self.conn.execute(
                        "INSERT INTO relations(subject_id, predicate, object_id, "
                        "confidence, source) VALUES(?,?,?,?,?)",
                        (e1["id"], "semantically_similar_to", sf["subject_id"],
                         round(sim, 3), "embedding-bridge"))
                    new_links += 1

        self.conn.commit()
        return {"new_links": new_links, "threshold": threshold}

    # ═══════════════════════════════════════════════
    # JKG 7.1: PREDICATE ALIASES — semantic equivalents
    # ═══════════════════════════════════════════════

    # When user asks "где работает", also match builds/founded/develops etc.
    _PREDICATE_ALIASES = {
        "works_at": ["builds", "founded", "develops", "creates", "runs", "leads",
                     "строит", "основала", "основал", "разрабатывает", "запустила",
                     "запустил", "работает_над", "владеет", "управляет"],
        "работает_в": ["строит", "основала", "основал", "разрабатывает", "запустила",
                       "запустил", "работает_над", "builds", "founded", "develops",
                       "creates", "runs", "leads"],
        "работает": ["строит", "основала", "основал", "разрабатывает", "запустила",
                     "запустил", "builds", "founded", "develops", "creates"],
    }

    def _expand_predicates(self, query: str) -> set:
        """Return set of predicates to search for, including semantic aliases."""
        predicates = {"works_at", "работает_в", "employed_at", "position"}
        # If query is about current work, expand
        if any(w in query.lower() for w in ["работает", "работа", "работаю",
                                              "где сейчас", "чем занимается",
                                              "где трудится", "works", "work",
                                              "job", "current"]):
            for alias_group in self._PREDICATE_ALIASES.values():
                predicates.update(alias_group)
        return predicates

    def llm_rerank(self, query: str, candidates: list, top_k: int = 10) -> list:
        """DeepSeek ranks candidates by relevance to query."""
        if not candidates: return []

        candidate_text = "\n".join([
            f"[{i}] {c['entity']} | {c['predicate']} | {c.get('value', c.get('object',''))}"
            for i, c in enumerate(candidates[:30])
        ])
        prompt = f"""Query: {query}

Rate each candidate's relevance to the query on a scale of 1-10.
Output ONLY valid JSON array: [{{"id": 0, "relevance": 8, "reason": "..."}}, ...]

Candidates:
{candidate_text}

JSON:"""
        result = _llm(prompt, "You rank search results by relevance. Output JSON only.")
        try:
            rankings = json.loads(result)
            ranked = {}
            for r in rankings:
                ranked[r["id"]] = r["relevance"]
            return sorted(candidates, key=lambda c, i=candidates.index(c): -ranked.get(i, 0))[:top_k]
        except:
            return candidates[:top_k]

    # ═══════════════════════════════════════════════
    # ASK — full hybrid pipeline
    # ═══════════════════════════════════════════════

    def ask(self, question: str, owner: str = "алтынай") -> dict:
        """Full hybrid pipeline with temporal awareness and access tracking."""
        # Detect if question is about past/present
        timeframe = self._extract_query_timeframe(question)
        past_indicators = ["раньше", "до", "был", "была", "было", "были", "прошлом",
                          "в 202", "история", "использовал", "работал", "was", "before"]
        is_past = any(w in question.lower() for w in past_indicators) or timeframe.get("mode") == "bounded"

        # Step 1: hybrid best-first graph retrieval
        candidates = self.retrieve_clark(question, top_k=20)

        # Step 1.5: record access and increase salience, never truth confidence
        self._clark_confidence_update(candidates, boost=0.05)

        # Step 2: Owner traversal (filter by validity)
        owner_trav = self.traverse(owner, depth=3, max_results=30)
        existing = {f"{c['entity']}|{c['predicate']}|{c.get('value',c.get('object',''))}"
                    for c in candidates}
        for f in owner_trav.get("facts", []):
            key = f"{f['entity']}|{f['predicate']}|{f['value']}"
            if key not in existing:
                candidates.append(f)
                existing.add(key)

        # Step 3: Enrich with temporal metadata + predicate alias awareness
        work_predicates = self._expand_predicates(question)
        for c in candidates:
            entity_name = c.get('entity', '')
            pred = c.get('predicate', '')
            # Look up temporal info
            sid = self._resolve_entity(entity_name)
            if sid:
                fact = self.conn.execute(
                    "SELECT valid_from, valid_until, confidence, superseded_by "
                    "FROM facts WHERE subject_id=? AND predicate=? AND object_text=? "
                    "ORDER BY created_at DESC LIMIT 1",
                    (sid, pred, c.get('value', c.get('object', '')))
                ).fetchone()
                if fact:
                    c["valid_from"] = fact["valid_from"]
                    c["valid_until"] = fact["valid_until"]
                    if not self._fact_matches_timeframe(fact["valid_from"], fact["valid_until"], timeframe):
                        c["_penalty"] = True
                    if fact["valid_until"]:
                        c["status"] = f"INVALID (ended {fact['valid_until']})" if not is_past else "HISTORICAL"
                        if not is_past:
                            c["_penalty"] = True  # Downrank invalid facts
                    elif fact["valid_from"]:
                        # If predicate is a work-alias, mark as current activity
                        if pred in work_predicates:
                            c["status"] = f"CURRENT since {fact['valid_from']}"
                        else:
                            c["status"] = f"VALID since {fact['valid_from']}"
                    c["confidence"] = fact["confidence"]

        # Filter: drop timeframe mismatches and invalid present-tense facts
        candidates = [c for c in candidates if not c.get("_penalty")]

        # JKG 7.1: Predicate expansion — for "где работает" questions,
        # boost facts with work-alias predicates (builds, founded, etc.)
        if work_predicates and not is_past:
            for c in candidates:
                if c.get('predicate', '') in work_predicates and not c.get('_penalty'):
                    c['_is_current_activity'] = True

        # Step 4: LLM rerank
        ranked = self.llm_rerank(question, candidates, top_k=15)

        # Step 5: Build context with temporal annotations
        context_parts = []
        for r in ranked:
            path = " → ".join(r.get("path", [])) if r.get("path") else ""
            status = f" [{r.get('status','')}]" if r.get("status") else ""
            if path:
                context_parts.append(f"TRACE {path} | {r['entity']} {r['predicate']} {r.get('value',r.get('object',''))}{status}")
            else:
                context_parts.append(f"FACT {r['entity']} {r['predicate']} {r.get('value',r.get('object',''))}{status}")

        context_parts.append("")
        context_parts.append("GRAPH PATHS:")
        for p in owner_trav.get("paths", [])[:10]:
            context_parts.append(f"  {p}")

        context = "\n".join(context_parts) if context_parts else "(ничего не найдено)"

        # Step 6: LLM synthesize
        temporal_note = (
            "Facts marked [CURRENT since] are the person's present occupation/activity. "
            "Facts marked [VALID since] are currently true. "
            "Facts marked [INVALID] or [HISTORICAL] are no longer true. "
            "For 'где работает' questions, prefer [CURRENT] facts, "
            "even if the predicate is 'builds', 'founded', 'develops', etc. — "
            "these ARE the person's current work."
        ) if not is_past else "Include both current and historical facts in your answer."
        answer = _llm(
            f"Knowledge graph context (hybrid graph+semantic search):\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Instructions:\n"
            f"- {temporal_note}\n"
            f"- Use GRAPH PATHS to trace entity connections\n"
            f"- Give definitive answers when data exists, include dates\n"
            f"- NAME every company, person, and entity explicitly — never paraphrase them away\n"
            f"- For career/история questions: list EVERY position chronologically with company names\n"
            f"- If entity X is connected to {owner} via a path, it IS relevant\n"
            f"Answer in Russian.",
            system="You answer questions using a temporal knowledge graph. "
                   "Respect FACT validity times. Be definitive and specific. "
                   "ALWAYS name entities explicitly — never replace names with generic descriptions."
        )

        return {"question": question, "answer": answer,
                "candidates": len(candidates), "ranked": len(ranked),
                "temporal_filter": not is_past}

    # ═══════════════════════════════════════════════
    # JKG 4.0: SEARCH EPISODES
    # ═══════════════════════════════════════════════

    def search_episodes(self, query: str, limit: int = 10) -> list:
        """Search raw episodes by keywords."""
        words = self._search_terms(query)
        if not words: return []
        conditions = " OR ".join(["body LIKE ?" for _ in words])
        params = [f"%{w}%" for w in words]
        rows = self.conn.execute(
            f"SELECT id, uuid, title, body, source_type, reference_time, created_at "
            f"FROM episodes WHERE {conditions} ORDER BY created_at DESC LIMIT ?",
            params + [limit]
        ).fetchall()
        return [{"id": r["id"], "uuid": r["uuid"], "title": r["title"],
                 "body": r["body"][:500], "source_type": r["source_type"],
                 "reference_time": r["reference_time"], "created_at": r["created_at"]}
                for r in rows]

    def get_fact_episode(self, fact_id: int) -> dict:
        """Trace a fact back to its source episode."""
        fact = self.conn.execute(
            "SELECT f.*, e.title, e.body, e.reference_time, e.source_type "
            "FROM facts f LEFT JOIN episodes e ON f.episode_id = e.id "
            "WHERE f.id=?", (fact_id,)
        ).fetchone()
        if not fact: return {"error": "not found"}
        return {"fact_id": fact_id, "subject": self._get_name(fact["subject_id"]),
                "predicate": fact["predicate"], "object": fact["object_text"],
                "episode_title": fact["title"], "episode_body": fact["body"][:300] if fact["body"] else "",
                "reference_time": fact["reference_time"], "source_type": fact["source_type"]}

    # ═══════════════════════════════════════════════
    # STATS
    # ═══════════════════════════════════════════════

    def stats(self) -> dict:
        entities = self.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        facts = self.conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        episodes = 0
        try: episodes = self.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        except: pass

        relations = 0
        try: relations = self.conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        except: pass

        orphans = 0
        try:
            orphans = self.conn.execute("""
                SELECT COUNT(*) FROM entities e
                WHERE e.id NOT IN (SELECT subject_id FROM relations
                                   UNION SELECT object_id FROM relations)
                AND NOT EXISTS (SELECT 1 FROM facts f WHERE f.subject_id = e.id)
            """).fetchone()[0]
        except: pass

        vectors = 0
        if self.has_vec:
            try: vectors = self.conn.execute("SELECT COUNT(*) FROM facts_vec").fetchone()[0]
            except: pass

        # Detect schema: check if forget_status column exists
        fact_cols = [c[1] for c in self.conn.execute("PRAGMA table_info(facts)").fetchall()]
        has_forget = 'forget_status' in fact_cols
        has_utility = 'utility_score' in fact_cols
        has_valid = 'valid_until' in fact_cols

        valid, historical, forgotten = 0, 0, 0
        if has_forget and has_valid:
            valid = self.conn.execute(
                "SELECT COUNT(*) FROM facts WHERE valid_until IS NULL AND superseded_by IS NULL AND forget_status='active'"
            ).fetchone()[0]
            historical = self.conn.execute(
                "SELECT COUNT(*) FROM facts WHERE valid_until IS NOT NULL OR superseded_by IS NOT NULL"
            ).fetchone()[0]
            forgotten = self.conn.execute(
                "SELECT COUNT(*) FROM facts WHERE forget_status IN ('forgotten','archived')"
            ).fetchone()[0]
        elif has_valid:
            valid = self.conn.execute(
                "SELECT COUNT(*) FROM facts WHERE valid_until IS NULL AND superseded_by IS NULL"
            ).fetchone()[0]
            historical = self.conn.execute(
                "SELECT COUNT(*) FROM facts WHERE valid_until IS NOT NULL OR superseded_by IS NOT NULL"
            ).fetchone()[0]
        else:
            valid = facts

        emotional = 0
        try:
            emotional = self.conn.execute(
                "SELECT COUNT(*) FROM episodes WHERE emotion != 'neutral' AND emotion_intensity > 0.3"
            ).fetchone()[0]
        except: pass

        schema_proposals = 0
        try: schema_proposals = self.conn.execute("SELECT COUNT(*) FROM schema_evolution").fetchone()[0]
        except: pass

        # Average utility
        avg_util = None
        if has_utility:
            try:
                avg_util = self.conn.execute(
                    "SELECT AVG(utility_score) FROM facts WHERE forget_status='active'"
                ).fetchone()[0]
            except: pass

        return {
            "entities": entities, "connected": entities - orphans, "orphans": orphans,
            "facts": facts, "valid_facts": valid, "historical_facts": historical,
            "forgotten_facts": forgotten, "relations": relations,
            "episodes": episodes, "emotional_episodes": emotional,
            "vectors": vectors, "has_embeddings": self.has_vec,
            "schema_proposals": schema_proposals,
            "avg_utility": round(avg_util, 3) if avg_util else None,
            "version": "8.1.0",
            "namespace": self.namespace,
            "embedding_signature": self.config.embedding_signature,
            "architecture": "five-layer+graph+full-text+vectors+temporal+lifecycle",
        }

    def list_entities(self, limit=50) -> list:
        rows = self.conn.execute(
            "SELECT e.name, e.type, "
            "(SELECT COUNT(*) FROM facts f WHERE f.subject_id=e.id) as fc, "
            "(SELECT COUNT(*) FROM relations r WHERE r.subject_id=e.id OR r.object_id=e.id) as rc "
            "FROM entities e ORDER BY fc + rc DESC LIMIT ?", (limit,)
        ).fetchall()
        return [{"name": r["name"], "type": r["type"],
                 "facts": r["fc"], "relations": r["rc"]} for r in rows]

    def forget(self, name: str) -> dict:
        eid = self._eid(name)
        # Delete vectors
        if self.has_vec:
            fact_ids = self.conn.execute(
                "SELECT id FROM facts WHERE subject_id=?", (eid,)).fetchall()
            for f in fact_ids:
                try:
                    self.conn.execute("DELETE FROM facts_vec WHERE fact_id=?", (f["id"],))
                except: pass
        # Delete graph data
        self.conn.execute("DELETE FROM relations WHERE subject_id=? OR object_id=?", (eid, eid))
        self.conn.execute("DELETE FROM facts WHERE subject_id=?", (eid,))
        self.conn.execute("DELETE FROM entities WHERE id=?", (eid,))
        self.conn.commit()
        return {"status": "deleted", "entity": name}

    # ═══════════════════════════════════════════════════════════════
    # JKG 7.1: UNIFIED MEMORY FABRIC — Multi-Layer Ingest API
    # ═══════════════════════════════════════════════════════════════

    # ── PROFILE LAYER ─────────────────────────────────────────────

    def remember_profile(self, text: str, category: str = "preference",
                         source: str = "explicit") -> dict:
        """Store explicit preferences; preserve unsupported input as a raw episode."""
        if not text.strip():
            raise ValueError("Profile text must not be empty")
        prompt = f"""Extract profile facts about the user from this text. Output ONLY valid JSON.

Text: {text[:2000]}

Format:
{{"facts": [{{"key": "snake_case_key", "value": "the fact value",
  "category": "identity|preference|environment|goal|contact",
  "confidence": 0.0-1.0}}]}}

Rules:
- 'identity': who the user IS (name, role, location, education)
- 'preference': what the user LIKES/DISLIKES (communication style, tools, habits)
- 'environment': facts about the user's setup (OS, devices, accounts)
- 'goal': what the user wants to achieve
- 'contact': social media, email, phone

Only extract NEW facts not yet known. Skip generic or obvious info.
JSON:"""
        try:
            raw = _llm(prompt, "You extract user profile facts. Output JSON only.")
            data = json.loads(raw.strip().replace("```json","").replace("```",""))
        except Exception:
            data = _fallback_profile_payload(text)

        stored = []
        if (not isinstance(data, dict) or not isinstance(data.get("facts"), list)
                or any(not isinstance(f, dict) or not isinstance(f.get("key"), str)
                       or not isinstance(f.get("value"), str) for f in data.get("facts", []))):
            data = _fallback_profile_payload(text)
        if not data.get("facts"):
            raw = self.remember(text, source=f"profile:{source}")
            return {"status": "no_profile_facts", "stored": 0, "facts": [],
                    "episode_id": raw["episode_id"], "extraction_status": "raw_only",
                    "warning": "Original text retained as an episode; no profile facts recognized."}
        with self.conn:
            for f in data.get("facts", []):
                key = f["key"].lower().replace(" ", "_")
                value = f["value"]
                cat = f.get("category", category)
                conf = f.get("confidence", 0.9)

                # Upsert: replace if exists
                self.conn.execute("""
                    INSERT INTO memory_profile(key, value, category, confidence, source, updated_at)
                    VALUES (?, ?, ?, ?, ?, datetime('now'))
                    ON CONFLICT(key) DO UPDATE SET
                        value=excluded.value, category=excluded.category,
                        confidence=excluded.confidence, source=excluded.source,
                        updated_at=datetime('now'), is_active=1
                """, (key, value, cat, conf, source))
                row = self.conn.execute(
                    "SELECT id FROM memory_profile WHERE key=?", (key,)
                ).fetchone()
                if row:
                    try:
                        self._replace_vector(
                            "memory_profile_vec", "profile_row_id", row["id"],
                            f"{key} {value} {cat}",
                        )
                    except EmbeddingConfigurationError:
                        raise
                    except Exception:
                        raise
                stored.append({"key": key, "value": value, "category": cat})

        return {"status": "ok", "stored": len(stored), "facts": stored}

    def get_profile(self, category: str = None) -> list:
        """Retrieve all active profile facts, optionally filtered by category."""
        if category:
            rows = self.conn.execute(
                "SELECT * FROM memory_profile WHERE is_active=1 AND category=? ORDER BY confidence DESC",
                (category,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM memory_profile WHERE is_active=1 ORDER BY category, confidence DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    # ── SEMANTIC LAYER ────────────────────────────────────────────

    def _semantic_id(self, canonical_key: str) -> str:
        return hashlib.md5(canonical_key.encode("utf-8")).hexdigest()[:16]

    def _semantic_topic_from_statement(self, statement: str) -> str:
        tokens = [
            token for token in re.findall(r"[a-zA-Zа-яА-Я0-9_]+", statement.lower())
            if token not in {"the", "and", "for", "with", "this", "that", "what", "how", "our", "your", "from"}
        ]
        return "-".join(tokens[:6]) or hashlib.md5(statement.encode("utf-8")).hexdigest()[:8]

    def _build_semantic_key(self, kind: str, scope: str, subject: str, statement: str) -> str:
        return ":".join([
            _slugify(kind or "project"),
            _slugify(scope or "global"),
            _slugify(subject or kind or "memory"),
            _slugify(self._semantic_topic_from_statement(statement)),
        ])

    def _remember_semantic_entry(
        self,
        *,
        kind: str,
        scope: str,
        subject: str,
        statement: str,
        canonical_key: str = "",
        tags: list | None = None,
        details: dict | None = None,
        confidence: float = 0.9,
        source: str = "manual",
    ) -> dict:
        normalized_kind = self._norm(kind or "project")
        normalized_scope = (scope or "global").strip()
        normalized_subject = (subject or normalized_kind).strip()
        normalized_statement = statement.strip()
        key = canonical_key or self._build_semantic_key(
            normalized_kind, normalized_scope, normalized_subject, normalized_statement
        )
        semantic_id = self._semantic_id(key)
        tags_json = json.dumps(sorted({str(tag).strip().lower() for tag in (tags or []) if str(tag).strip()}), ensure_ascii=False)
        details_json = json.dumps(details or {}, ensure_ascii=False, sort_keys=True)

        existing = self.conn.execute(
            "SELECT id, access_count FROM memory_semantic WHERE canonical_key=?",
            (key,),
        ).fetchone()

        with self.conn:
            if existing:
                self.conn.execute("""
                    UPDATE memory_semantic
                    SET kind=?, scope=?, subject=?, statement=?, details_json=?, tags_json=?,
                        confidence=?, source=?, status='active', updated_at=datetime('now'),
                        last_reinforced_at=datetime('now')
                    WHERE canonical_key=?
                """, (
                    normalized_kind, normalized_scope, normalized_subject, normalized_statement,
                    details_json, tags_json, float(confidence), source, key,
                ))
                row_id = existing["id"]
                action = "updated"
            else:
                cur = self.conn.execute("""
                    INSERT INTO memory_semantic(
                        semantic_id, kind, scope, subject, canonical_key, statement,
                        details_json, tags_json, confidence, source
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    semantic_id, normalized_kind, normalized_scope, normalized_subject, key,
                    normalized_statement, details_json, tags_json, float(confidence), source,
                ))
                row_id = cur.lastrowid
                action = "created"

            if self.has_vec:
                try:
                    semantic_text = " ".join([
                        normalized_kind, normalized_scope, normalized_subject,
                        normalized_statement, " ".join(json.loads(tags_json)),
                    ]).strip()
                    self._replace_vector("memory_semantic_vec", "semantic_row_id", row_id, semantic_text)
                except EmbeddingConfigurationError:
                    raise
                except Exception:
                    raise

        return {
            "status": "ok",
            "action": action,
            "semantic_id": semantic_id,
            "canonical_key": key,
            "kind": normalized_kind,
            "scope": normalized_scope,
            "subject": normalized_subject,
            "statement": normalized_statement,
        }

    def remember_semantic(
        self,
        statement: str,
        *,
        kind: str = "project",
        scope: str = "global",
        subject: str = "",
        canonical_key: str = "",
        tags: list | None = None,
        details: dict | None = None,
        confidence: float = 0.9,
        source: str = "manual",
    ) -> dict:
        """Store durable semantic knowledge about the project, tools, or rules."""
        try:
            prompt = f"""Extract durable semantic memory entries from this text. Output ONLY valid JSON.

Text: {statement[:3000]}

Format:
{{"entries":[{{
  "kind":"project|tool|rule",
  "scope":"workspace/project/team/global",
  "subject":"short subject",
  "canonical_key":"stable_key_or_empty",
  "statement":"durable truth",
  "tags":["tag1","tag2"],
  "confidence":0.0-1.0,
  "details":{{}}
}}]}}

Only extract stable knowledge worth remembering across sessions.
Skip one-off events, timestamps, and transient chat context.
JSON:"""
            raw = _llm(prompt, "You extract durable semantic memory. Output JSON only.")
            payload = json.loads(raw.strip().replace("```json", "").replace("```", ""))
        except Exception:
            payload = _fallback_semantic_payload(
                statement,
                kind=kind,
                scope=scope,
                subject=subject,
                canonical_key=canonical_key,
                tags=tags,
                confidence=confidence,
            )

        stored = []
        for entry in payload.get("entries", []):
            stored.append(self._remember_semantic_entry(
                kind=entry.get("kind", kind),
                scope=entry.get("scope", scope),
                subject=entry.get("subject", subject),
                statement=entry.get("statement", statement),
                canonical_key=canonical_key or entry.get("canonical_key", ""),
                tags=entry.get("tags", tags),
                details=entry.get("details", details),
                confidence=float(entry.get("confidence", confidence)),
                source=source,
            ))
        return {"status": "ok", "stored": len(stored), "entries": stored}

    def remember_project(
        self,
        statement: str,
        *,
        project: str = "default",
        canonical_key: str = "",
        tags: list | None = None,
        details: dict | None = None,
        confidence: float = 0.95,
        source: str = "manual",
    ) -> dict:
        return self._remember_semantic_entry(
            kind="project",
            scope=project,
            subject=project,
            statement=statement,
            canonical_key=canonical_key,
            tags=(tags or []) + ["project"],
            details=details,
            confidence=confidence,
            source=source,
        )

    def remember_tool(
        self,
        statement: str,
        *,
        tool_name: str,
        canonical_key: str = "",
        tags: list | None = None,
        details: dict | None = None,
        confidence: float = 0.93,
        source: str = "manual",
    ) -> dict:
        return self._remember_semantic_entry(
            kind="tool",
            scope="tooling",
            subject=tool_name,
            statement=statement,
            canonical_key=canonical_key,
            tags=(tags or []) + ["tool", _slugify(tool_name)],
            details=details,
            confidence=confidence,
            source=source,
        )

    def remember_rule(
        self,
        statement: str,
        *,
        rule_name: str,
        canonical_key: str = "",
        tags: list | None = None,
        details: dict | None = None,
        confidence: float = 0.97,
        source: str = "manual",
    ) -> dict:
        return self._remember_semantic_entry(
            kind="rule",
            scope="rules",
            subject=rule_name,
            statement=statement,
            canonical_key=canonical_key,
            tags=(tags or []) + ["rule", _slugify(rule_name)],
            details=details,
            confidence=confidence,
            source=source,
        )

    def list_semantic(self, kind: str = None, scope: str = None, limit: int = 50) -> list:
        clauses = ["status='active'"]
        params = []
        if kind:
            clauses.append("kind=?")
            params.append(self._norm(kind))
        if scope:
            clauses.append("scope=?")
            params.append(scope)
        rows = self.conn.execute(
            f"SELECT * FROM memory_semantic WHERE {' AND '.join(clauses)} "
            "ORDER BY kind, confidence DESC, updated_at DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["tags"] = json.loads(item.get("tags_json", "[]"))
            item["details"] = json.loads(item.get("details_json", "{}"))
            results.append(item)
        return results

    def search_semantic(self, query: str, limit: int = 8, kind: str = None) -> list:
        norm_q = self._norm(query)
        terms = self._search_terms(query)
        base_rows = self.list_semantic(kind=kind, limit=250)
        if not base_rows:
            return []

        q_vec = None
        lexical_candidates = []
        embedding_candidates = []
        for row in base_rows:
            searchable = " ".join([
                row.get("kind", ""), row.get("scope", ""), row.get("subject", ""),
                row.get("canonical_key", ""), row.get("statement", ""),
                " ".join(row.get("tags", [])),
            ]).strip()
            lexical_hits = sum(1 for term in terms if term in searchable.lower())
            lexical_bonus = self._keyword_overlap_bonus(
                query, row.get("subject", ""), row.get("kind", ""), row.get("statement", "")
            )
            candidate = (lexical_hits, lexical_bonus, searchable, row)
            if lexical_hits > 0 or lexical_bonus > 0:
                lexical_candidates.append(candidate)
            else:
                embedding_candidates.append(candidate)

        results = []
        for lexical_hits, lexical_bonus, _searchable, row in lexical_candidates:
            item = dict(row)
            item["layer"] = "semantic"
            item["cosine_sim"] = None
            item["score"] = round((item.get("confidence", 0.5) * 0.55) + lexical_bonus + (lexical_hits * 0.12), 4)
            results.append(item)

        # Query persisted vectors rather than re-encoding every candidate.
        if len(results) < limit and self.has_vec:
            try:
                q_vec = np.asarray(self._encode(query), dtype=np.float32)
                vector_rows = self.conn.execute("""
                    SELECT semantic_row_id, distance FROM memory_semantic_vec
                    WHERE embedding MATCH ? AND k = ? ORDER BY distance
                """, (q_vec.tobytes(), max(limit * 3, 10))).fetchall()
                by_id = {row["id"]: row for row in base_rows}
                existing_ids = {item["id"] for item in results}
                for vector_row in vector_rows:
                    row = by_id.get(vector_row["semantic_row_id"])
                    if not row or row["id"] in existing_ids:
                        continue
                    cosine = self._distance_similarity(vector_row["distance"])
                    if terms and cosine < 0.18:
                        continue
                    item = dict(row)
                    item["layer"] = "semantic"
                    item["cosine_sim"] = cosine
                    item["score"] = round((item.get("confidence", 0.5) * 0.45) + (cosine * 0.45), 4)
                    results.append(item)
            except EmbeddingConfigurationError:
                raise
            except Exception:
                pass

        results.sort(key=lambda item: item.get("score", 0), reverse=True)
        top = results[:limit]
        if top:
            self.conn.executemany(
                "UPDATE memory_semantic SET access_count = access_count + 1, last_reinforced_at=datetime('now') WHERE id=?",
                [(item["id"],) for item in top],
            )
            self.conn.commit()
        return top

    # ── EPISODIC LAYER ────────────────────────────────────────────

    def remember_session(self, session_id: str, text: str, source: str = "terminal",
                         title: str = "", summary: str = "", importance: float = 0.5) -> dict:
        """Index a session transcript into episodic memory."""
        sid = hashlib.md5(f"{source}:{session_id}".encode()).hexdigest()[:16]

        # Auto-generate summary if not provided
        if not summary and len(text) > 200:
            try:
                sprompt = f"Summarize this conversation in 2-3 sentences. Be factual and concise.\n\n{text[:4000]}"
                summary = _llm(sprompt, "You summarize conversations concisely.")
            except Exception:
                summary = text[:300]

        # Auto-detect emotion
        emotion = "neutral"
        emotion_intensity = 0.0
        try:
            eprompt = f"Detect the dominant emotion. Output ONLY a JSON: {{\"emotion\":\"...\",\"intensity\":0.0-1.0}}\n\n{text[:2000]}"
            raw = _llm(eprompt, "You detect emotions. Output JSON only.")
            edata = json.loads(raw.strip().replace("```json","").replace("```",""))
            emotion = edata.get("emotion", "neutral")
            emotion_intensity = float(edata.get("intensity", 0.0))
        except Exception:
            pass

        self.conn.execute("""
            INSERT OR REPLACE INTO memory_sessions(id, session_id, source, title, summary, full_text, emotion, emotion_intensity, importance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (sid, session_id, source, title or f"Session {session_id[:12]}",
              summary, text[:10000], emotion, emotion_intensity, importance))
        session_row = self.conn.execute(
            "SELECT rowid AS vector_id FROM memory_sessions WHERE id=?", (sid,)
        ).fetchone()
        if session_row:
            try:
                self._replace_vector(
                    "memory_session_vec", "session_row_id", session_row["vector_id"],
                    f"{title} {summary} {text[:10000]}",
                )
            except EmbeddingConfigurationError:
                raise
            except Exception:
                pass
        self.conn.commit()

        # Also try to extract facts and profile updates from the session
        facts_extracted = 0
        try:
            # Extract factual knowledge
            result = self.remember(text, source=f"session:{session_id[:16]}")
            facts_extracted = result.get("facts", 0)
            self.conn.execute(
                "UPDATE memory_sessions SET fact_count=? WHERE id=?",
                (facts_extracted, sid))
            self.conn.commit()
        except EmbeddingConfigurationError:
            raise
        except Exception:
            pass

        return {
            "status": "ok", "session_id": session_id, "id": sid,
            "emotion": emotion, "facts_extracted": facts_extracted,
            "summary": summary[:200]
        }

    def search_sessions(self, query: str, limit: int = 5) -> list:
        """Search episodic memory by keyword + embedding (if available)."""
        results = []
        norm_q = self._norm(query)

        # Keyword search
        for term in self._search_terms(norm_q):
            rows = self.conn.execute("""
                SELECT id, session_id, source, title, summary, emotion, importance, created_at
                FROM memory_sessions
                WHERE (title LIKE ? OR summary LIKE ? OR full_text LIKE ?)
                ORDER BY importance DESC, created_at DESC
                LIMIT ?
            """, (f"%{term}%", f"%{term}%", f"%{term}%", limit)).fetchall()
            for r in rows:
                if r["id"] not in {x.get("id") for x in results}:
                    results.append(dict(r))

        # Persisted vector search
        if self.has_vec:
            try:
                q_vec = np.asarray(self._encode(query), dtype=np.float32)
                rows = self.conn.execute("""
                    SELECT s.id, s.session_id, s.source, s.title, s.summary, s.emotion,
                           s.importance, s.created_at, v.distance
                    FROM memory_session_vec v
                    JOIN memory_sessions s ON s.rowid=v.session_row_id
                    WHERE v.embedding MATCH ? AND k = ? ORDER BY v.distance
                """, (q_vec.tobytes(), max(limit * 2, 5))).fetchall()
                known = {item["id"] for item in results}
                for row in rows:
                    if row["id"] in known:
                        continue
                    item = dict(row)
                    item["retrieval_score"] = self._distance_similarity(item.pop("distance"))
                    results.append(item)
            except EmbeddingConfigurationError:
                raise
            except Exception:
                pass

        for item in results:
            lexical = self._lexical_similarity(query, f"{item.get('title', '')} {item.get('summary', '')}")
            item["retrieval_score"] = max(item.get("retrieval_score", 0.0), lexical)
        results.sort(key=lambda x: (x.get("retrieval_score", 0.0), x["importance"]), reverse=True)
        return results[:limit]

    # ── PROCEDURAL LAYER ──────────────────────────────────────────

    def index_skill(self, name: str, description: str = "", triggers: list = None,
                    category: str = "general", version: str = "1.0.0") -> dict:
        """Index a skill into procedural memory for retrieval."""
        sid = hashlib.md5(f"skill:{name}".encode()).hexdigest()[:12]
        triggers_json = json.dumps(triggers or [], ensure_ascii=False)

        # Check if skill already indexed
        existing = self.conn.execute("SELECT id FROM memory_skills WHERE name=?", (name,)).fetchone()

        if existing:
            self.conn.execute("""
                UPDATE memory_skills SET description=?, triggers=?, category=?, version=?,
                updated_at=datetime('now') WHERE name=?
            """, (description, triggers_json, category, version, name))
        else:
            self.conn.execute("""
                INSERT INTO memory_skills(id, name, description, triggers, category, version)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (sid, name, description, triggers_json, category, version))

        skill_row = self.conn.execute(
            "SELECT rowid AS vector_id, id FROM memory_skills WHERE name=?", (name,)
        ).fetchone()
        if skill_row:
            sid = skill_row["id"]
            try:
                self._replace_vector(
                    "memory_skill_vec", "skill_row_id", skill_row["vector_id"],
                    f"{name} {description} {triggers_json} {category}",
                )
            except EmbeddingConfigurationError:
                raise
            except Exception:
                pass

        self.conn.commit()
        return {"status": "ok", "name": name, "id": sid, "action": "updated" if existing else "created"}

    def search_skills(self, query: str, limit: int = 5) -> list:
        """Search procedural memory by keyword in name, description, and triggers."""
        norm_q = self._norm(query)
        results = []
        for term in self._search_terms(norm_q):
            rows = self.conn.execute("""
                SELECT id, name, description, triggers, category, version
                FROM memory_skills WHERE is_active=1
                AND (name LIKE ? OR description LIKE ? OR triggers LIKE ?)
                ORDER BY updated_at DESC LIMIT ?
            """, (f"%{term}%", f"%{term}%", f"%{term}%", limit)).fetchall()
            for r in rows:
                if r["id"] not in {x.get("id") for x in results}:
                    d = dict(r)
                    try: d["triggers"] = json.loads(d.get("triggers", "[]"))
                    except: d["triggers"] = []
                    d["retrieval_score"] = self._lexical_similarity(
                        query, f"{d['name']} {d['description']} {' '.join(d['triggers'])}"
                    )
                    results.append(d)
        if self.has_vec:
            try:
                q_vec = np.asarray(self._encode(query), dtype=np.float32)
                rows = self.conn.execute("""
                    SELECT s.id, s.name, s.description, s.triggers, s.category, s.version,
                           v.distance
                    FROM memory_skill_vec v JOIN memory_skills s ON s.rowid=v.skill_row_id
                    WHERE s.is_active=1 AND v.embedding MATCH ? AND k = ? ORDER BY v.distance
                """, (q_vec.tobytes(), max(limit * 2, 5))).fetchall()
                known = {item["id"] for item in results}
                for row in rows:
                    if row["id"] in known:
                        continue
                    item = dict(row)
                    item["triggers"] = json.loads(item.get("triggers", "[]"))
                    item["retrieval_score"] = self._distance_similarity(item.pop("distance"))
                    results.append(item)
            except EmbeddingConfigurationError:
                raise
            except Exception:
                pass
        results.sort(key=lambda item: item.get("retrieval_score", 0.0), reverse=True)
        return results[:limit]

    def _lexical_similarity(self, query: str, candidate: str) -> float:
        query_tokens = {
            token for token in re.findall(r"[a-zA-Zа-яА-Я0-9_]+", self._norm(query))
            if len(token) > 2
        }
        candidate_tokens = {
            token for token in re.findall(r"[a-zA-Zа-яА-Я0-9_]+", self._norm(candidate))
            if len(token) > 2
        }
        union = query_tokens | candidate_tokens
        return len(query_tokens & candidate_tokens) / len(union) if union else 0.0

    def _fuse_layer_results(self, query: str, results: list, limit: int) -> list:
        """Calibrate heterogeneous layer scores and remove duplicate context."""
        query_norm = self._norm(query)
        weights = {
            "semantic": 1.0,
            "profile": 1.0,
            "factual": 1.05,
            "episodic": 0.95,
            "procedural": 0.95,
        }
        if any(term in query_norm for term in ("rule", "policy", "tool", "project", "правил", "инструмент", "проект")):
            weights["semantic"] = 1.2
        if any(term in query_norm for term in ("prefer", "profile", "like", "about me", "предпоч", "профил", "люб")):
            weights["profile"] = 1.2
        if any(term in query_norm for term in ("session", "discuss", "conversation", "yesterday", "сесс", "обсуж", "разговор", "вчера")):
            weights["episodic"] = 1.2
        if any(term in query_norm for term in ("how to", "workflow", "skill", "procedure", "как сделать", "навык", "процедур")):
            weights["procedural"] = 1.2

        grouped = defaultdict(list)
        for result in results:
            grouped[result.get("layer", "factual")].append(result)

        fused = []
        for layer, layer_results in grouped.items():
            layer_results.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
            for rank, result in enumerate(layer_results):
                item = dict(result)
                raw_score = max(0.0, min(1.0, float(item.get("score", 0.0))))
                rank_score = 1.0 / (rank + 1)
                fused_score = (0.75 * raw_score + 0.25 * rank_score) * weights.get(layer, 1.0)
                item["raw_score"] = round(raw_score, 4)
                item["score"] = round(max(0.0, min(1.0, fused_score)), 4)
                fused.append(item)

        def signature(item):
            if item.get("layer") == "profile":
                return self._norm(f"profile|{item.get('key')}|{item.get('value')}")
            if item.get("layer") == "semantic":
                return self._norm(f"semantic|{item.get('canonical_key')}|{item.get('statement')}")
            if item.get("layer") == "procedural":
                return self._norm(f"procedural|{item.get('name')}|{item.get('description')}")
            if item.get("layer") == "episodic":
                return self._norm(f"episodic|{item.get('session_id', item.get('uuid', item.get('id')))}")
            return self._norm(f"factual|{item.get('entity')}|{item.get('predicate')}|{item.get('value', item.get('object_text'))}")

        deduplicated = {}
        for item in fused:
            key = signature(item)
            if key not in deduplicated or item["score"] > deduplicated[key]["score"]:
                deduplicated[key] = item
        return sorted(deduplicated.values(), key=lambda item: item["score"], reverse=True)[:limit]

    # ═══════════════════════════════════════════════════════════════
    # JKG 7.1: UNIFIED QUERY — Multi-layer search with CLARK fusion
    # ═══════════════════════════════════════════════════════════════

    def query(self, text: str, layers: list = None, limit: int = 10) -> dict:
        """Unified query across all memory layers. CLARK-fused and ranked.

        Args:
            text: natural language query
            layers: subset of ['semantic','profile','factual','episodic','procedural'] (default: all)
            limit: max results per layer before fusion

        Returns:
            dict with 'results' (ranked across layers) and 'by_layer' breakdown
        """
        if layers is None:
            layers = ["semantic", "profile", "factual", "episodic", "procedural"]

        all_results = []
        by_layer = {}
        errors = {}
        timeframe = self._extract_query_timeframe(text)

        # ── Semantic Layer ──
        if "semantic" in layers and timeframe.get("mode") == "any":
            try:
                semantic_results = self.search_semantic(text, limit=limit)
                by_layer["semantic"] = len(semantic_results)
                all_results.extend(semantic_results)
            except EmbeddingConfigurationError:
                raise
            except Exception as exc:
                by_layer["semantic"] = 0
                errors["semantic"] = str(exc)[:200]

        # ── Profile Layer ──
        if "profile" in layers and timeframe.get("mode") == "any":
            norm_q = self._norm(text)
            profile_results = []
            # Direct key/value search
            for term in self._search_terms(norm_q):
                rows = self.conn.execute("""
                    SELECT key, value, category, confidence, source
                    FROM memory_profile WHERE is_active=1
                    AND (key LIKE ? OR value LIKE ?)
                    ORDER BY confidence DESC LIMIT ?
                """, (f"%{term}%", f"%{term}%", limit)).fetchall()
                for r in rows:
                    d = dict(r)
                    d["layer"] = "profile"
                    d["score"] = float(d.get("confidence", 0.5)) * 0.9
                    profile_results.append(d)

            # Embedding fallback: if keyword found nothing, try semantic search
            if (
                not profile_results and self.has_vec and self.embedder
                and timeframe.get("mode") == "any"
                and not any(term in norm_q for term in ["where", "live", "lived", "moved", "location", "где", "живет", "жил", "переех", "recipe", "pie", "рецепт", "пирог"])
            ):
                try:
                    q_vec = np.asarray(self._encode(text), dtype=np.float32)
                    rows = self.conn.execute("""
                        SELECT p.key, p.value, p.category, p.confidence, p.source, v.distance
                        FROM memory_profile_vec v
                        JOIN memory_profile p ON p.id=v.profile_row_id
                        WHERE p.is_active=1 AND v.embedding MATCH ? AND k = ?
                        ORDER BY v.distance
                    """, (q_vec.tobytes(), max(limit * 2, 5))).fetchall()
                    if rows:
                        scored = []
                        for p in rows:
                            sim = self._distance_similarity(p["distance"])
                            if sim < 0.18:
                                continue
                            d = dict(p)
                            d.pop("distance", None)
                            d["layer"] = "profile"
                            d["score"] = sim * 0.8  # embed score slightly lower than keyword
                            scored.append(d)
                        scored.sort(key=lambda x: x["score"], reverse=True)
                        profile_results = scored[:limit]
                except EmbeddingConfigurationError:
                    raise
                except Exception:
                    pass

            by_layer["profile"] = len(profile_results)
            all_results.extend(profile_results)

        # ── Factual Layer (JKG core — CLARK retrieval) ──
        if "factual" in layers:
            try:
                factual_results = self.retrieve_clark(text, top_k=limit * 3)
                filtered_factual = []
                for r in factual_results:
                    metadata = self._get_fact_metadata(
                        r.get("entity", ""),
                        r.get("predicate", ""),
                        r.get("value", r.get("object_text", ""))
                    )
                    if metadata:
                        r["valid_from"] = metadata["valid_from"]
                        r["valid_until"] = metadata["valid_until"]
                        r["confidence"] = metadata["confidence"]
                        if not self._fact_matches_timeframe(metadata["valid_from"], metadata["valid_until"], timeframe):
                            continue
                    r["layer"] = "factual"
                    lexical_bonus = self._keyword_overlap_bonus(
                        text,
                        r.get("entity", ""),
                        r.get("predicate", ""),
                        r.get("value", r.get("object_text", ""))
                    )
                    if (lexical_bonus <= 0 and (self.config.embedding_provider == "hash" or r.get("cosine_sim", 0) < 0.4)
                            and len(re.findall(r"[a-zA-Zа-яА-Я0-9_]+", self._norm(text))) >= 3):
                        continue
                    base_score = r.get("clark_score", r.get("confidence", 0.5))
                    r["score"] = float(base_score) + lexical_bonus
                    if r.get("predicate") in {"loves", "hates", "favorite", "least_favorite", "has_favorite_color", "least_favorite_color"}:
                        r["value"] = f"{r.get('predicate')} {r.get('value', '')}".strip()
                    if timeframe.get("mode") == "bounded":
                        r["score"] += 0.35
                    filtered_factual.append(r)
                factual_results = filtered_factual[:limit]
                if not factual_results and timeframe.get("mode") == "bounded":
                    fallback_candidates = self.recall(text, limit=limit * 5)
                    for r in fallback_candidates:
                        metadata = self._get_fact_metadata(
                            r.get("entity", ""),
                            r.get("predicate", ""),
                            r.get("value", r.get("object_text", ""))
                        )
                        if not metadata:
                            continue
                        if not self._fact_matches_timeframe(metadata["valid_from"], metadata["valid_until"], timeframe):
                            continue
                        lexical_bonus = self._keyword_overlap_bonus(
                            text,
                            r.get("entity", ""),
                            r.get("predicate", ""),
                            r.get("value", r.get("object_text", ""))
                        )
                        if lexical_bonus <= 0:
                            continue
                        r["valid_from"] = metadata["valid_from"]
                        r["valid_until"] = metadata["valid_until"]
                        r["confidence"] = metadata["confidence"]
                        r["layer"] = "factual"
                        r["score"] = 0.6 + lexical_bonus
                        filtered_factual.append(r)
                        if len(filtered_factual) >= limit:
                            break
                    factual_results = filtered_factual[:limit]
                by_layer["factual"] = len(factual_results)
                all_results.extend(factual_results)
            except EmbeddingConfigurationError:
                raise
            except Exception as e:
                # Fallback to RRF recall
                errors["factual_clark"] = str(e)[:200]
                try:
                    factual_results = self.recall(text, limit=limit * 3)
                    filtered_factual = []
                    for r in factual_results:
                        metadata = self._get_fact_metadata(
                            r.get("entity", ""),
                            r.get("predicate", ""),
                            r.get("value", r.get("object_text", ""))
                        )
                        if metadata:
                            r["valid_from"] = metadata["valid_from"]
                            r["valid_until"] = metadata["valid_until"]
                            r["confidence"] = metadata["confidence"]
                            if not self._fact_matches_timeframe(metadata["valid_from"], metadata["valid_until"], timeframe):
                                continue
                        r["layer"] = "factual"
                        r["score"] = r.get("rrf_score", 0.5)
                        filtered_factual.append(r)
                    factual_results = filtered_factual[:limit]
                    by_layer["factual"] = len(factual_results)
                    all_results.extend(factual_results)
                except EmbeddingConfigurationError:
                    raise
                except Exception as fallback_exc:
                    by_layer["factual"] = 0
                    errors["factual"] = str(fallback_exc)[:200]

        # ── Episodic Layer ──
        if "episodic" in layers:
            try:
                ep_results = self.search_sessions(text, limit=limit)
                raw_episodes = self.search_episodes(text, limit=limit)
                seen_episode_keys = {r.get("session_id") or r.get("id") for r in ep_results}
                for episode in raw_episodes:
                    key = episode.get("uuid") or episode.get("id")
                    if key in seen_episode_keys:
                        continue
                    episode["importance"] = 0.45
                    ep_results.append(episode)
                for r in ep_results:
                    r["layer"] = "episodic"
                    searchable = " ".join([
                        str(r.get("title", "")), str(r.get("summary", "")),
                        str(r.get("body", "")),
                    ])
                    relevance = self._lexical_similarity(text, searchable)
                    contextual_bonus = self._keyword_overlap_bonus(text, "episode", "mentions", searchable)
                    r["score"] = max(0.0, min(
                        1.0,
                        0.65 * relevance + 0.35 * float(r.get("importance", 0.5)) + contextual_bonus,
                    ))
                by_layer["episodic"] = len(ep_results)
                all_results.extend(ep_results)
            except EmbeddingConfigurationError:
                raise
            except Exception as exc:
                by_layer["episodic"] = 0
                errors["episodic"] = str(exc)[:200]

        # ── Procedural Layer ──
        if "procedural" in layers:
            try:
                sk_results = self.search_skills(text, limit=limit)
                for r in sk_results:
                    r["layer"] = "procedural"
                    searchable = f"{r.get('name', '')} {r.get('description', '')} {' '.join(r.get('triggers', []))}"
                    r["score"] = min(1.0, 0.4 + 0.6 * self._lexical_similarity(text, searchable))
                by_layer["procedural"] = len(sk_results)
                all_results.extend(sk_results)
            except EmbeddingConfigurationError:
                raise
            except Exception as exc:
                by_layer["procedural"] = 0
                errors["procedural"] = str(exc)[:200]

        # ── Calibrated cross-layer fusion ──
        top = self._fuse_layer_results(text, all_results, limit)

        return {
            "query": text,
            "embedding_provider": self.config.embedding_provider,
            "semantic_search": self.config.embedding_provider != "hash",
            "layers_searched": layers,
            "total_candidates": sum(by_layer.values()),
            "by_layer": by_layer,
            "errors": errors,
            "results": top
        }

    # ═══════════════════════════════════════════════════════════════
    # JKG 7.1: SESSION START INJECTION ENGINE
    # ═══════════════════════════════════════════════════════════════

    def session_start_context(self, owner: str = "алтынай",
                              recent_sessions: int = 3,
                              max_chars: int = 12000) -> str:
        """Generate a dynamic context block for session start prompt injection.

        Replaces flat memory injection with CLARK-ranked, multi-layer context
        that adapts to what's relevant NOW.
        """
        blocks = []

        # 1. Profile facts (user identity + preferences)
        profile = self.get_profile()
        if profile:
            by_cat = defaultdict(list)
            for p in profile:
                by_cat[p.get("category", "preference")].append(p)

            profile_lines = []
            for cat, items in by_cat.items():
                for item in items[:5]:  # top 5 per category
                    profile_lines.append(f"  {item['key']}: {item['value']}")

            if profile_lines:
                blocks.append("[Profile]\n" + "\n".join(profile_lines[:20]))

        # 2. Active factual knowledge about owner
        try:
            owner_eid = self._resolve_entity(owner)
            if owner_eid:
                facts = self.conn.execute("""
                    SELECT f.predicate, f.object_text, f.confidence, f.valid_until,
                           e.name as entity_name
                    FROM facts f
                    JOIN entities e ON f.subject_id = e.id
                    WHERE f.subject_id = ? AND f.valid_until IS NULL
                    AND f.superseded_by IS NULL
                    ORDER BY f.confidence DESC, f.access_count DESC
                    LIMIT 15
                """, (owner_eid,)).fetchall()

                if facts:
                    fact_lines = []
                    for f in facts:
                        marker = "🔗" if f["confidence"] > 0.8 else "  "
                        fact_lines.append(f"  {marker} {f['entity_name']} {f['predicate']} {f['object_text']}")
                    blocks.append("[Active Facts]\n" + "\n".join(fact_lines))
        except Exception:
            pass

        # 3. Recent episodes (last N sessions)
        try:
            sessions = self.conn.execute("""
                SELECT title, summary, emotion, created_at
                FROM memory_sessions
                ORDER BY created_at DESC LIMIT ?
            """, (recent_sessions,)).fetchall()

            if sessions:
                ep_lines = []
                for s in sessions:
                    emoji = {"happy": "😊", "focused": "🎯", "neutral": "💬", "excited": "🔥", "curious": "🤔"}.get(s["emotion"], "")
                    ep_lines.append(f"  {emoji} {s['title']}: {s['summary'][:150]}")
                blocks.append("[Recent Episodes]\n" + "\n".join(ep_lines))
        except Exception:
            pass

        # 4. Stable semantic memory for project / tools / rules
        try:
            semantic = self.list_semantic(limit=18)
            if semantic:
                grouped = defaultdict(list)
                for item in semantic:
                    grouped[item.get("kind", "project")].append(item)

                semantic_lines = []
                for label, kind in [("Project", "project"), ("Tools", "tool"), ("Rules", "rule")]:
                    for item in grouped.get(kind, [])[:6]:
                        semantic_lines.append(
                            f"  [{label}] {item.get('subject')}: {item.get('statement')[:140]}"
                        )
                if semantic_lines:
                    blocks.append("[Semantic Memory]\n" + "\n".join(semantic_lines))
        except Exception:
            pass

        # 5. Relevant skills (most recently updated)
        try:
            skills = self.conn.execute("""
                SELECT name, description, category
                FROM memory_skills WHERE is_active=1
                ORDER BY updated_at DESC LIMIT 8
            """).fetchall()

            if skills:
                sk_lines = []
                for s in skills:
                    sk_lines.append(f"  {s['name']}: {s['description'][:120]}")
                blocks.append("[Relevant Skills]\n" + "\n".join(sk_lines))
        except Exception:
            pass

        # Build final injection block
        header = "DYNAMIC CONTEXT (JKG 8 Unified Memory)"
        budget = ContextBudget(max_chars=max(1000, max_chars))
        layer_by_heading = {
            "[Profile]": "profile", "[Active Facts]": "factual",
            "[Recent Episodes]": "episodic", "[Semantic Memory]": "semantic",
            "[Relevant Skills]": "procedural",
        }
        bounded = []
        for block in blocks:
            layer = next((value for heading, value in layer_by_heading.items() if block.startswith(heading)), "")
            limit = budget.limit_for(layer) if layer else max_chars
            bounded.append(block[:limit])
        body = "\n\n".join(bounded) if bounded else "(no context loaded)"
        body = body[:max_chars]

        return f"══════════════════════════════════════════════\n{header}\n══════════════════════════════════════════════\n{body}"

    # ═══════════════════════════════════════════════════════════════
    # JKG 7.1: SYNC BRIDGES — Auto-ingest from other systems
    # ═══════════════════════════════════════════════════════════════

    def sync_from_memory(self, content: str, target: str = "memory") -> dict:
        """Bridge from built-in memory tool → JKG 7.1.

        When memory(action='add') is called, this auto-indexes into the right layer.
        target='user' → profile layer, target='memory' → factual layer.
        """
        if target == "user":
            return self.remember_profile(content)
        else:
            return self.remember(content)

    def sync_from_session(self, session_id: str, transcript: str, source: str = "terminal",
                          summary: str = "") -> dict:
        """Bridge from session end → JKG 7.1 episodic + factual layers."""
        return self.remember_session(
            session_id=session_id, text=transcript, source=source,
            summary=summary, importance=0.7
        )

    def sync_from_skill(self, name: str, description: str = "",
                        triggers: list = None, category: str = "general") -> dict:
        """Bridge from skill create/update → JKG 7.1 procedural layer."""
        return self.index_skill(name=name, description=description,
                                triggers=triggers, category=category)

    def sync_semantic_context(
        self,
        *,
        project: list | None = None,
        tools: list | None = None,
        rules: list | None = None,
        scope: str = "default",
        source: str = "sync",
    ) -> dict:
        """Bulk-ingest stable project/tool/rule knowledge into the semantic layer."""
        stored = []
        for item in project or []:
            stored.append(self.remember_project(
                item["statement"],
                project=item.get("project", scope),
                canonical_key=item.get("canonical_key", ""),
                tags=item.get("tags"),
                details=item.get("details"),
                confidence=float(item.get("confidence", 0.95)),
                source=source,
            ))
        for item in tools or []:
            stored.append(self.remember_tool(
                item["statement"],
                tool_name=item.get("tool_name") or item.get("subject") or "tool",
                canonical_key=item.get("canonical_key", ""),
                tags=item.get("tags"),
                details=item.get("details"),
                confidence=float(item.get("confidence", 0.93)),
                source=source,
            ))
        for item in rules or []:
            stored.append(self.remember_rule(
                item["statement"],
                rule_name=item.get("rule_name") or item.get("subject") or "rule",
                canonical_key=item.get("canonical_key", ""),
                tags=item.get("tags"),
                details=item.get("details"),
                confidence=float(item.get("confidence", 0.97)),
                source=source,
            ))
        return {"status": "ok", "stored": len(stored), "entries": stored}

    def migrate_builtin_memory(self, memory_data: str) -> dict:
        """One-shot: migrate existing flat memory injection into JKG profile layer.

        Parses the current memory injection format and stores each entry as a profile fact.
        """
        results = {"imported": 0, "skipped": 0, "errors": []}

        # Parse the memory block format (key: value or bullet points)
        lines = memory_data.split("\n")
        current_key = None

        for line in lines:
            line = line.strip()
            if not line or line.startswith("═") or line.startswith("MEMORY"):
                continue

            # Profile-style entries: "Не нравится обращение"
            if any(kw in line.lower() for kw in ["предпочитает", "нравится", "любит", "не терпит", "студент", "github", "instagram", "рабочий процесс"]):
                try:
                    self.remember_profile(line, source="migration")
                    results["imported"] += 1
                except Exception as e:
                    results["errors"].append(str(e))
            else:
                results["skipped"] += 1

        return results


# ═══════════════════════════════════════════════
# LLM EXTRACTION
# ═══════════════════════════════════════════════

def _extract_entities(text: str) -> str:
    prompt = f"""Extract entities, facts, and relations. Output ONLY valid JSON.

Text: {text[:3000]}

Format:
{{"entities": [{{"name": "...", "type": "person|company|technology|concept|..."}}],
 "facts": [{{"subject": "entity", "predicate": "is|has|owns|uses|...", "object": "value"}}],
 "relations": [{{"subject": "e1", "predicate": "works_at|competes_with|owns|...", "object": "e2"}}]}}

CRITICAL: For EVERY fact, ALSO create a relation linking entities.
Example: "Altynai's Instagram is blocked" → fact: instagram is_blocked, relation: altynai owns instagram
JSON:"""
    try:
        raw = _llm(prompt, "You extract knowledge graph triples. Always create relations. Output JSON only.")
        data = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
        if not isinstance(data, dict):
            raise ValueError("Expected an extraction object")
        for key, fields in {"entities": ("name",), "facts": ("subject", "predicate", "object"),
                            "relations": ("subject", "predicate", "object")}.items():
            if not isinstance(data.get(key), list) or any(
                not isinstance(item, dict) or any(not isinstance(item.get(field), str) for field in fields)
                for item in data[key]
            ):
                raise ValueError(f"Invalid extraction field: {key}")
        return json.dumps(data, ensure_ascii=False)
    except Exception:
        return json.dumps(_fallback_extract_entities_payload(text), ensure_ascii=False)


def main_cli():
    """Entry point for `python -m jkg.memory` and the `jkg` console script."""
    import runpy
    runpy.run_path(os.path.abspath(__file__), run_name="__main__")


# ═══════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stats"
    hm = HybridMemory()

    if cmd == "remember":
        text = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else sys.stdin.read()
        result = hm.remember(text, ground_to="алтынай")
        print(json.dumps(result, ensure_ascii=False))

    elif cmd == "recall":
        query = " ".join(sys.argv[2:])
        for r in hm.recall(query):
            print(f"  [{r.get('rrf_score','')}] {r['entity']} {r['predicate']} {r['value']}")

    elif cmd == "traverse":
        entity = sys.argv[2] if len(sys.argv) > 2 else "алтынай"
        depth = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        result = hm.traverse(entity, depth=depth)
        print(f"Traversing '{result['entity']}' (depth {depth}): {result['facts_count']} facts")
        for f in result["facts"]:
            print(f"  [{f['depth']}] {f['entity']} {f['predicate']} {f['value']}")
        for p in result["paths"]:
            print(f"  → {p}")

    elif cmd == "path":
        print(json.dumps(hm.find_path(
            sys.argv[2] if len(sys.argv) > 2 else "алтынай",
            sys.argv[3] if len(sys.argv) > 3 else "trace.so"),
            ensure_ascii=False))

    elif cmd == "ask":
        question = " ".join(sys.argv[2:])
        result = hm.ask(question)
        print(f"Q: {result['question']}")
        print(f"A: {result['answer']}")
        print(f"  ({result['candidates']} candidates → {result['ranked']} reranked)")

    elif cmd == "bridge":
        sub = sys.argv[2] if len(sys.argv) > 2 else "both"
        if sub in ("both", "graph"):
            print("Graph→Embeddings:", json.dumps(hm.bridge_graph_to_embeddings(), ensure_ascii=False))
        if sub in ("both", "embeddings"):
            print("Embeddings→Graph:", json.dumps(hm.bridge_embeddings_to_graph(), ensure_ascii=False))

    elif cmd == "stats":
        print(json.dumps(hm.stats(), indent=2, ensure_ascii=False))

    elif cmd == "entities":
        for e in hm.list_entities():
            print(f"  [{e['type']}] {e['name']} ({e['facts']} facts, {e['relations']} rels)")

    elif cmd == "forget":
        print(json.dumps(hm.forget(" ".join(sys.argv[2:])), ensure_ascii=False))

    elif cmd == "episodes":
        query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""
        results = hm.search_episodes(query) if query else hm.conn.execute(
            "SELECT id, uuid, title, body, reference_time FROM episodes ORDER BY created_at DESC LIMIT 10"
        ).fetchall()
        if isinstance(results, list) and results and isinstance(results[0], dict):
            for e in results:
                print(f"  [{e['reference_time']}] {e['title']}")
                print(f"    {e['body'][:200]}")
        else:
            for e in (results or []):
                print(f"  [{e['reference_time']}] {e['title']} — {e['body'][:150]}")

    elif cmd == "resolve":
        name = " ".join(sys.argv[2:])
        eid = hm.resolve_entity(name)
        ename = hm._get_name(eid)
        print(f"'{name}' → [{eid}] {ename}")

    elif cmd == "invalidate":
        if len(sys.argv) >= 5:
            result = hm._invalidate_conflicting(sys.argv[2], sys.argv[3], sys.argv[4])
            print(f"Invalidated: {result}")

    elif cmd == "utility":
        result = hm.update_utility_scores()
        print(json.dumps(result, ensure_ascii=False))

    elif cmd == "prune":
        threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
        dry = "--execute" not in sys.argv
        result = hm.prune_memory(threshold=threshold, dry_run=dry)
        print(json.dumps(result, ensure_ascii=False))

    elif cmd == "emotions":
        ctx = hm.get_emotional_context()
        for e in ctx:
            print(f"  [{e['intensity']:.1f}] {e['emotion']}: {e['title'][:80]} ({e['time']})")

    elif cmd == "evolve":
        result = hm.evolve_schema(dry_run="--execute" not in sys.argv)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif cmd == "gdpr-delete":
        name = sys.argv[2] if len(sys.argv) > 2 else ""
        rid = sys.argv[3] if len(sys.argv) > 3 else f"REQ-{datetime.now().strftime('%Y%m%d%H%M')}"
        verified = "--confirm" in sys.argv
        result = hm.gdpr_delete(name, request_id=rid, verified=verified)
        print(json.dumps(result, ensure_ascii=False))

    elif cmd == "clark":
        # Parse: clark [top_k] query...
        query = " ".join(sys.argv[2:])
        top_k = 10
        if len(sys.argv) > 2 and sys.argv[2].isdigit():
            top_k = int(sys.argv[2])
            query = " ".join(sys.argv[3:]) if len(sys.argv) > 3 else ""
        print(f"CLARK retrieval: \"{query}\"")
        for r in hm.retrieve_clark(query, top_k=top_k):
            print(f"  [{r['clark_score']}] {r['entity']} {r['predicate']} {r['value']}")
            print(f"    conf={r['confidence']} cos={r['cosine_sim']} temporal={r['temporal_bonus']}")

    elif cmd == "propagate":
        iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        result = hm.propagate_confidence(iterations=iterations)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    # ═══ JKG 7.1: Unified Query CLI ═══

    elif cmd == "query":
        query = " ".join(sys.argv[2:])
        layers = None
        # Support: query --layers semantic,profile,factual "text"
        if "--layers" in sys.argv:
            idx = sys.argv.index("--layers")
            layers = sys.argv[idx+1].split(",")
            query = " ".join(sys.argv[2:idx] + sys.argv[idx+2:])
        result = hm.query(query, layers=layers)
        print(f"Query: {result['query']}")
        print(f"Layers: {result['layers_searched']} → {result['total_candidates']} candidates")
        for r in result["results"]:
            layer_tag = f"[{r['layer']}]"
            if r["layer"] == "profile":
                print(f"  {layer_tag} {r['key']}: {r['value']} (conf={r.get('confidence','?')})")
            elif r["layer"] == "semantic":
                print(f"  {layer_tag} [{r.get('kind','?')}] {r.get('subject','?')}: {r.get('statement','')} (score={r.get('score','?')})")
            elif r["layer"] == "factual":
                print(f"  {layer_tag} {r.get('entity','?')} {r.get('predicate','?')} {r.get('value', r.get('object_text','?'))} (score={r.get('score','?')})")
            elif r["layer"] == "episodic":
                print(f"  {layer_tag} {r.get('title','?')}: {r.get('summary','')[:120]}")
            elif r["layer"] == "procedural":
                print(f"  {layer_tag} {r['name']}: {r.get('description','')[:120]}")

    elif cmd == "session-start":
        print(hm.session_start_context())

    elif cmd == "remember-profile":
        text = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else sys.stdin.read()
        result = hm.remember_profile(text)
        print(json.dumps(result, ensure_ascii=False))

    elif cmd == "remember-session":
        sid = sys.argv[2] if len(sys.argv) > 2 else f"manual-{int(time.time())}"
        text = sys.stdin.read() if not sys.stdin.isatty() else " ".join(sys.argv[3:])
        result = hm.remember_session(sid, text)
        print(json.dumps(result, ensure_ascii=False))

    elif cmd == "index-skill":
        name = sys.argv[2] if len(sys.argv) > 2 else "unknown"
        desc = " ".join(sys.argv[3:]) if len(sys.argv) > 3 else ""
        result = hm.index_skill(name, description=desc)
        print(json.dumps(result, ensure_ascii=False))

    elif cmd == "remember-project":
        project = sys.argv[2] if len(sys.argv) > 2 else "default"
        text = sys.stdin.read() if not sys.stdin.isatty() else " ".join(sys.argv[3:])
        result = hm.remember_project(text, project=project)
        print(json.dumps(result, ensure_ascii=False))

    elif cmd == "remember-tool":
        tool_name = sys.argv[2] if len(sys.argv) > 2 else "tool"
        text = sys.stdin.read() if not sys.stdin.isatty() else " ".join(sys.argv[3:])
        result = hm.remember_tool(text, tool_name=tool_name)
        print(json.dumps(result, ensure_ascii=False))

    elif cmd == "remember-rule":
        rule_name = sys.argv[2] if len(sys.argv) > 2 else "rule"
        text = sys.stdin.read() if not sys.stdin.isatty() else " ".join(sys.argv[3:])
        result = hm.remember_rule(text, rule_name=rule_name)
        print(json.dumps(result, ensure_ascii=False))

    elif cmd == "semantic":
        kind = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
        for item in hm.list_semantic(kind=kind):
            print(f"  [{item['kind']}] {item['subject']}: {item['statement']} (key={item['canonical_key']})")

    elif cmd == "migrate-memory":
        text = sys.stdin.read() if not sys.stdin.isatty() else " ".join(sys.argv[2:])
        result = hm.migrate_builtin_memory(text)
        print(json.dumps(result, ensure_ascii=False))

    elif cmd == "profile":
        cat = sys.argv[2] if len(sys.argv) > 2 else None
        for p in hm.get_profile(cat):
            print(f"  [{p['category']}] {p['key']}: {p['value']} (conf={p['confidence']})")

    elif cmd == "sessions":
        query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""
        for s in hm.search_sessions(query):
            print(f"  [{s['emotion']}] {s['title']}: {s.get('summary','')[:150]}")

    else:
        print("Commands: remember, recall, clark, traverse, path, ask, bridge, "
              "stats, entities, forget, episodes, resolve, invalidate, "
              "utility, prune, propagate, emotions, evolve, gdpr-delete, "
              "query, session-start, remember-profile, remember-session, "
              "index-skill, remember-project, remember-tool, remember-rule, "
              "semantic, migrate-memory, profile, sessions")
