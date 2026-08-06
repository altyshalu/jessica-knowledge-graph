<div align="center">

# 🧠 Jessica Knowledge Graph

### One local database for semantic, profile, factual, episodic, and procedural memory

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-7.1.0-7C3AED)](https://github.com/altyshalu/jessica-knowledge-graph)
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

- One SQLite database for all memory layers
- Unified cross-layer query with CLARK-style score fusion
- Knowledge graph traversal, entity resolution, and path finding
- BM25 full-text search plus vector similarity through `sqlite-vec`
- Bi-temporal facts, confidence propagation, conflict invalidation, and utility-based forgetting
- Local deterministic embeddings by default; Gemini embeddings are optional
- Optional DeepSeek-powered extraction, reranking, and summarization
- GDPR-oriented deletion workflow with an audit trail
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

The factual layer uses CLARK retrieval in three stages:

1. **Value Iteration** propagates confidence through the graph.
2. **A\* Search** selects relevant entities using graph importance, semantic similarity, and temporal signals.
3. **Confidence Update** reinforces retrieved facts and their neighbors.

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

memory = HybridMemory(db_path="./data/jkg.sqlite3")

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
memory.session_start_context()
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

The preliminary LongMemEval-S run reports **96.8% Recall@5** on 100 of 500 questions and **100% Recall@10** for that sample. The benchmark runner and methodology are available in [`benchmark/run_longmemeval.py`](benchmark/run_longmemeval.py).

These results are preliminary and are not a substitute for a full, independently reproduced benchmark. Workload, embedding backend, hardware, and dataset size can materially change results.

## Production notes

- Set `JKG_DB_PATH` to a persistent application data directory; do not keep production data inside the installed package.
- Back up the SQLite database and test restore procedures before relying on it for durable state.
- Keep `.env` and API keys outside version control and rotate any key that may have been exposed.
- Use one embedding dimension per database. Switching between local 384-dimensional and Gemini 768-dimensional embeddings requires a fresh database or a migration.
- SQLite is a strong fit for a single service instance. Evaluate a server database or explicit coordination before using one database file across multiple hosts.
- Run privacy-deletion tests against your own ingestion paths before making regulatory guarantees.

## Project status

JKG 7.1 is **beta software**. The core package and semantic-memory tests are covered by CI, but several advanced behaviors still need broader stress testing. See [Architecture Audit & Fix Checklist](docs/BUG_REPORT_AND_FIXES.md) for the current engineering notes.

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
