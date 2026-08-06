<div align="center">

# 🧠 Jessica Knowledge Graph

### One local database for semantic, profile, factual, episodic, and procedural memory

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-8.0.0-7C3AED)](https://github.com/altyshalu/jessica-knowledge-graph)
[![License: MIT](https://img.shields.io/badge/license-MIT-22C55E)](LICENSE)
[![Status: Beta](https://img.shields.io/badge/status-beta-F59E0B)](#project-status)

**Local-first · SQLite-backed · Unified retrieval · Optional cloud embeddings**

</div>

---

Jessica Knowledge Graph (JKG) is a Python memory engine that keeps five kinds of knowledge in one SQLite database and retrieves them through one query interface. It combines structured knowledge graphs, full-text search, vector similarity, temporal validity, and CLARK ranking without requiring an external database for recall.

## Why JKG

Memory-heavy applications often split preferences, facts, transcripts, rules, and workflows across unrelated stores. JKG keeps those layers transactionally consistent in a single database:

| Layer | Stores | Example |
|---|---|---|
| **Semantic** | Durable project, tool, and rule knowledge | `rule:file-safety → ask before overwrite` |
| **Profile** | Identity, preferences, and environment | `communication_style → concise` |
| **Factual** | Entities, relations, and time-aware facts | `alice works_at acme [2023 → now]` |
| **Episodic** | Session transcripts and summaries | `May 15 → discussed roadmap` |
| **Procedural** | Skills, triggers, and workflows | `code-review → security, performance, style` |

### Highlights

- One isolated SQLite database per namespace, with all five memory layers
- Unified cross-layer query with CLARK-style score fusion
- Knowledge graph traversal, entity resolution, and path finding
- BM25 full-text search plus vector similarity through `sqlite-vec`
- Bi-temporal facts, explicit corrections, conflict invalidation, and utility-based forgetting
- Persisted vector indexes for every layer and an embedding compatibility registry
- Feedback, access history, entity aliases, consolidation, and bounded context assembly
- Local deterministic embeddings by default; Gemini embeddings are optional
- Optional DeepSeek-powered extraction, reranking, and summarization
- Deletion workflow with an audit trail (beta)
- CLI and Python API

## Architecture

```text
                         query("what does Alice prefer?")
                                      │
             ┌────────────┬───────────┼───────────┬────────────┐
             ▼            ▼           ▼           ▼            ▼
         Semantic      Profile     Factual     Episodic    Procedural
        tools/rules   preferences    graph      sessions      skills
             └────────────┴───────────┼───────────┴────────────┘
                                      ▼
                   CLARK fusion · ranking · top-N results
                                      │
                                      ▼
                            one SQLite database
```

The factual layer uses a CLARK-inspired hybrid retrieval pipeline:

1. **Value Iteration** propagates confidence through the graph.
2. **Best-first retrieval** seeds candidates with vector and BM25 results, then expands graph neighbors.
3. **Calibrated ranking** combines semantic, lexical, temporal, evidence-confidence, and graph-distance signals.

Retrieval never increases truth confidence. Reads update a separate salience score and access log.

## Quick start

### 1. Install

```bash
git clone https://github.com/altyshalu/jessica-knowledge-graph.git
cd jessica-knowledge-graph
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

### 2. Store and query memory

```python
from jkg import HybridMemory

memory = HybridMemory(db_path="./data/jkg.sqlite3", namespace="my-product")

memory.remember("Alice works at Acme since 2023")
memory.remember_profile("Alice prefers concise answers")
memory.remember_project(
    "The product uses one SQLite database for all memory layers.",
    project="jkg",
)
memory.remember_rule(
    "Never overwrite user-owned files without approval.",
    rule_name="file-safety",
)

result = memory.query("How does Alice prefer to communicate?")
for item in result["results"]:
    print(item["layer"], item.get("score"), item)
```

### 3. Use the CLI

The package installs the `jkg` command. `python -m jkg.memory` works as an equivalent entry point.

```bash
jkg remember "Alice is a software engineer at Acme since 2023"
jkg remember-profile "Alice prefers concise answers"
jkg remember-project jkg "JKG stores all memory layers in SQLite"
jkg remember-rule file-safety "Never overwrite user files without approval"
jkg query "What does Alice prefer?"
jkg stats
```

## Configuration

JKG works without API keys. When no embedding provider is configured, it uses deterministic local hash embeddings.

| Variable | Purpose | Default |
|---|---|---|
| `JKG_DB_PATH` | SQLite database location | `jkg/memory_v3.db` |
| `JKG_NAMESPACE` | Physical memory namespace | `default` |
| `GEMINI_API_KEY` | Enables Gemini embeddings | unset |
| `DEEPSEEK_API_KEY` | Enables remote extraction and synthesis | unset |
| `JKG_ENV_PATH` | Path to a custom `.env` file | auto-discovery |
| `JKG_LLM_TIMEOUT_SECONDS` | LLM request timeout | `20` |
| `JKG_EMBEDDING_TIMEOUT_SECONDS` | Embedding request timeout | `8` |
| `JKG_SQLITE_BUSY_TIMEOUT_MS` | SQLite lock wait time | `15000` |

Example `.env`:

```dotenv
JKG_DB_PATH=./data/jkg.sqlite3
GEMINI_API_KEY=
DEEPSEEK_API_KEY=
```

Never commit real API keys. `.env` and database files are ignored by Git.

## Memory APIs

```python
# Semantic
memory.remember_project("...", project="my-project")
memory.remember_tool("...", tool_name="my-tool")
memory.remember_rule("...", rule_name="my-rule")

# Profile, factual, episodic, procedural
memory.remember_profile("...")
memory.remember("...")
memory.remember_session("session-001", "...")
memory.index_skill("code-review", description="...", triggers=["review PR"])

# Unified retrieval
memory.query("...", layers=["semantic", "profile", "factual"])
memory.session_start_context(max_chars=8000)

# Lifecycle and maintenance
memory.feedback("factual", fact_id, "useful")
memory.add_entity_alias("alice", "Al")
memory.consolidate(dry_run=True)
memory.reindex_embeddings(force=True)
```

More detail about durable project, tool, and rule memory is available in [Semantic Memory](docs/SEMANTIC_MEMORY.md).

## CLI reference

| Area | Commands |
|---|---|
| Ingest | `remember`, `remember-profile`, `remember-project`, `remember-tool`, `remember-rule`, `remember-session`, `index-skill` |
| Query | `query`, `recall`, `ask`, `clark`, `semantic`, `profile`, `sessions` |
| Graph | `entities`, `traverse`, `path`, `resolve`, `bridge` |
| Maintenance | `stats`, `propagate`, `utility`, `prune`, `forget`, `evolve`, `gdpr-delete` |
| Context | `session-start`, `episodes`, `emotions`, `migrate-memory` |

Run `jkg` without arguments to print database statistics. An unknown command prints the available command list.

## Benchmarks

An exploratory LongMemEval-S script is available in [`benchmark/run_longmemeval.py`](benchmark/run_longmemeval.py). Its historical sample results are not claimed as a reproducible JKG 8 benchmark because the script evaluates a specialized retrieval index rather than the complete public `HybridMemory` pipeline.

These results are preliminary and are not a substitute for a full, independently reproduced benchmark. Workload, embedding backend, hardware, and dataset size can materially change results.

## Production notes

- Set `JKG_DB_PATH` to a persistent application data directory; do not keep production data inside the installed package.
- Back up the SQLite database and test restore procedures before relying on it for durable state.
- Keep `.env` and API keys outside version control and rotate any key that may have been exposed.
- The embedding registry rejects an incompatible provider/model/dimension. Reopen with `reindex_on_mismatch=True` for an explicit rebuild, preferably after a backup.
- Use a namespace per user, workspace, or tenant. Namespaces map to separate database files to prevent cross-tenant reads.
- SQLite is a strong fit for a single service instance. Evaluate a server database or explicit coordination before using one database file across multiple hosts.
- Run privacy-deletion tests against your own ingestion paths before making regulatory guarantees.

## Project status

JKG 8.0 is **beta software**. Correctness, namespace isolation, lifecycle, and vector-index invariants are tested locally, but broader load, migration, and retrieval-quality testing is still required.

## Development

```bash
python -m pip install -e .
python -m unittest discover -v
python -m compileall -q jkg examples
```

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Please report security issues through the process in [SECURITY.md](SECURITY.md).

## License

Released under the [MIT License](LICENSE). Copyright © 2026 Altynai.

<div align="center">

**One database. Five memory layers. Zero drift.**

</div>
