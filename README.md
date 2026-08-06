# 🧠 Jessica Knowledge Graph 7.1

> **Unified Memory Fabric — durable semantic memory plus profile, factual, episodic, and procedural recall in a single SQLite transaction.**
> Project knowledge. Tool knowledge. Rules. Profile. Facts. Sessions. Skills. All fused by CLARK.
> Local-first SQLite architecture with optional Gemini embeddings via `GEMINI_API_KEY`.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![LongMemEval](https://img.shields.io/badge/LongMemEval--S-96.8%25%20R@5-green.svg)](https://arxiv.org/abs/2410.10813)
[![Lines](https://img.shields.io/badge/code-2622%20lines-purple.svg)](jkg/memory.py)

---

## The Problem

Every AI agent has **fragmented memory**. User preferences sit in one system. Factual knowledge in another. Session transcripts in a third. Skills in flat files. They don't talk to each other — and every retrieval means manually stitching together four separate queries.

## What JKG 7.1 Is

**One database. Five memory layers. One query.**

```
┌─────────────────────────────────────────────────────┐
│                 unified query("...")                │
│          CLARK fusion across all layers             │
└──────────┬──────────┬──────────┬───────────────────┘
           │          │          │
     ┌─────▼──┐ ┌─────▼──┐ ┌─────▼─────┐ ┌──────────▼──┐
     │PROFILE │ │FACTUAL │ │ EPISODIC  │ │ PROCEDURAL   │
     │        │ │        │ │           │ │              │
     │Who am  │ │What do │ │What did   │ │How do I     │
     │I? What │ │I know? │ │we discuss │ │do X?        │
     │do I    │ │Entities │ │in past    │ │Skills +     │
     │like?   │ │+ facts  │ │sessions?  │ │workflows    │
     │        │ │+ graph  │ │           │ │              │
     └────────┘ └────────┘ └───────────┘ └──────────────┘
```

### Five Layers, One SQLite Database

| Layer | Table | Purpose | Example |
|-------|-------|---------|---------|
| **Semantic** | `memory_semantic` | Stable project/tool/rule knowledge | `rule:file-safety -> never overwrite user-owned files without explicit approval` |
| **Profile** | `memory_profile` | Identity, preferences, environment | `communication_style: warm but fast` |
| **Factual** | `facts` + `entities` + `relations` | Knowledge graph with temporal validity | `alice works_at acme [2023→now]` |
| **Episodic** | `memory_sessions` | Session transcripts + auto-summaries | `May 15: discussed memory architecture` |
| **Procedural** | `memory_skills` | Skills with trigger conditions | `response-formatting: triggered by "format"` |

---

## Why JKG 7.0

Most AI memory systems keep components in separate databases. Graph in Neo4j. Vectors in pgvector. Sessions in another store. **State drift is inevitable** — the graph says one thing, embeddings say another.

JKG 7.0 runs **everything in one SQLite transaction**: graph, embeddings, temporal validity, emotional context, intentional forgetting, self-evolving schema, profile preferences, session transcripts, and skill indexes. **No drift. No orphans. No external APIs for recall.**

| Feature | Mem0 | Cognee | Zep | Letta | **JKG 7.0** |
|---------|:----:|:------:|:---:|:-----:|:-----------:|
| Four memory layers | ❌ 2 | ❌ 2 | ❌ 3 | ❌ 2 | ✅ |
| Unified cross-layer query | ❌ | ❌ | ❌ | ❌ | ✅ CLARK |
| Graph + Embeddings, one DB | — | — | — | — | ✅ |
| Dynamic session context injection | ❌ | ❌ | ❌ | ⚠️ | ✅ |
| Auto-sync bridges | ❌ | ❌ | ❌ | ❌ | ✅ |
| Bi-temporal validity (T+T') | — | — | ✅ | — | ✅ |
| Emotional context | — | — | — | — | ✅ |
| Intentional forgetting (utility) | — | — | — | — | ✅ |
| Self-evolving schema | — | — | — | — | ✅ |
| GDPR true deletion + audit trail | — | — | — | — | ✅ |
| CLARK retrieval (Value Iteration + A*) | — | — | — | — | ✅ |
| Fully local (no cloud for recall) | — | — | — | — | ✅ |

---

## Quick Start

```bash
git clone https://github.com/altyshalu/jessica-knowledge-graph.git
cd jessica-knowledge-graph
pip install -e .
```

### Embeddings setup

```bash
# Optional but recommended for semantic retrieval quality
export GEMINI_API_KEY=your_key_here
```

If `GEMINI_API_KEY` is not set, JKG still runs. It falls back to deterministic local hash embeddings, so semantic retrieval still works without cloud calls, just with lower quality than Gemini.

### Your First Memory

```bash
# Remember a fact (factual layer)
python3 -m jkg.memory remember "Alice is a software engineer at Acme Corp since 2023"

# Remember a user preference (profile layer)
python3 -m jkg.memory remember-profile "Alice prefers concise, no-fluff answers"

# Remember stable project / tool / rule knowledge (semantic layer)
python3 -m jkg.memory remember-project "A Zone automates startup supply-side operations before scoring" "a-zone"
python3 -m jkg.memory remember-tool "Hermes is the runtime behind Jessica" "Hermes"
python3 -m jkg.memory remember-rule "Never overwrite user-owned files without approval" "file-safety"

# Index a session transcript (episodic layer)
python3 -m jkg.memory remember-session "session-001" "Discussed Q3 roadmap and memory architecture"

# Index a skill (procedural layer)
python3 -m jkg.memory index-skill "code-review" "Reviews PRs for security, performance, and style"

# Query across ALL layers at once
python3 -m jkg.memory query "what does Alice prefer?"

# Get dynamic session-start context (replaces flat prompt injection)
python3 -m jkg.memory session-start
```

### Python API

```python
from jkg import HybridMemory

hm = HybridMemory()

# ── Ingest ──────────────────────────────
hm.remember("Alice works at Acme")                # factual
hm.remember_profile("Alice prefers fast replies")  # profile
hm.remember_project("A Zone prepares startups before investor matching", project="a-zone")
hm.remember_tool("Hermes is Jessica's runtime", tool_name="Hermes")
hm.remember_rule("Never overwrite user-owned files without approval", rule_name="file-safety")
hm.remember_session("sess-001", "transcript...")  # episodic
hm.index_skill("my-skill", description="...", triggers=["when ..."])  # procedural

# ── Unified Query ───────────────────────
# One call, all layers, CLARK-ranked
result = hm.query("how does Alice communicate?")
for r in result["results"]:
    print(f"[{r['layer']}] score={r['score']:.2f} — {r}")
# [profile]  score=0.92 — key: communication_style, value: fast
# [factual]  score=0.87 — alice prefers быстрые ответы
# [episodic] score=0.72 — Session May 12: "don't call me bestie"

# ── Session Context ─────────────────────
context = hm.session_start_context()
# DYNAMIC CONTEXT (JKG 7.0)
# [Profile] communication_style: fast...
# [Active Facts] alice works_at acme...
# [Recent Episodes] 🎯 May 15: discussed architecture...
# [Relevant Skills] code-review, response-formatting...
```

---

## Architecture

### CLARK Retrieval (bio-inspired)

Inspired by the **Clark's Nutcracker** (*Nucifraga columbiana*) — a bird with the best long-term spatial memory in the animal kingdom (30,000+ seed caches, 9+ month retrieval window).

**Three stages:**

1. **Value Iteration** — elastic adaptation (like seasonal hippocampus growth): propagates confidence through the graph
2. **A\* Search** — retrieval (like landmark-based navigation): selects top entities by PageRank × confidence, scores with cosine similarity + temporal bonus
3. **Confidence Update** — self-learning (like neurogenesis): retrieved facts get +0.05 confidence boost, neighbors get +0.017

```
Value Iteration     A* Search          Confidence Update
   ┌───┐           ┌──────┐           ┌──────┐
   │ V │ ────────→ │  A*  │ ────────→ │  +Δ  │
   └───┘   graph   └──────┘  retrieve └──────┘  learn
  propagate         landmarks           boost
```

### Multi-Layer Query Flow

```
unified_query("how does Alice communicate?")
  │
  ├─► Semantic Layer: stable project/tool/rule retrieval
  │    → "Hermes is the runtime behind Jessica" (score=0.96)
  │
  ├─► Profile Layer:  key/value LIKE search + embedding fallback
  │    → "communication_style: fast" (score=0.92)
  │
  ├─► Factual Layer:  CLARK (Value Iteration → A* → Confidence Update)
  │    → "alice prefers быстрые ответы" (score=0.87)
  │
  ├─► Episodic Layer: full_text LIKE search
  │    → "Session May 12: don't call me bestie" (score=0.72)
  │
  └─► Procedural Layer: name/description/triggers LIKE search
       → "response-formatting: no tables, emoji..." (score=0.65)
  │
  ▼
CLARK-style fusion: sort by score, return top-N across layers
```

### Database Schema (21 tables, 1 file)

```
memory_v3.db
├── SEMANTIC       memory_semantic    (kind, scope, subject, canonical_key, statement)
├── PROFILE        memory_profile     (key, value, category, confidence)
├── FACTUAL        entities           (id, name, type, summary)
│                  facts              (subject, predicate, object, temporal, confidence, utility)
│                  relations          (subject → predicate → object)
│                  episodes           (uuid, body, emotion)
│                  facts_fts          (BM25 keyword search)
│                  facts_vec          (sqlite-vec KNN search)
├── EPISODIC       memory_sessions    (session_id, full_text, summary, emotion, importance)
├── PROCEDURAL     memory_skills      (name, description, triggers, version)
├── EVOLUTION      schema_evolution   (pattern proposals)
│                  forget_log         (utility-based pruning trail)
│                  deletion_log       (GDPR audit trail)
```

---

## Benchmarks

**LongMemEval-S** — industry standard for agent memory retrieval ([arXiv:2410.10813](https://arxiv.org/abs/2410.10813))

| System | Recall@5 | Recall@10 | Notes |
|--------|:--------:|:---------:|-------|
| **JKG 7.0** | **96.8%** | **100%** | CLARK A* retrieval |
| MemPalace | 96.6% | — | Graph + LLM rerank |
| agentmemory | 95.2% | — | Baseline |

*Preliminary on 100/500 questions. Full 500-question run requires GPU (~30 min on CPU). Results saved to `benchmark/longmemeval_jkg_results.json`.*

---

## Comparison with Industry (May 2026)

| System | 4 Layers | Unified Query | Dynamic Injection | Sync Bridges | Graph+Emb 1DB | Temporal | Emotional | Forgetting | Self-Evolve | Privacy | $/mo |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **JKG 7.0** | ✅ | ✅ CLARK | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | $0 |
| Graphiti/Zep | ❌ 3 | ❌ | ❌ | ❌ | ❌ Neo4j+PG | ✅ | ❌ | ❌ | ❌ | ❌ | $$ |
| Cognee | ❌ 2 | ❌ | ❌ | ❌ | ❌ 3 DBs | ❌ | ❌ | ❌ | ❌ | ❌ | $$ |
| Mem0 | ❌ 2 | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | $$ |
| Letta/MemGPT | ❌ 2 | ❌ | ⚠️ | ❌ | ✅ | ⚠️ | ❌ | ❌ | ❌ | ❌ | $ |

---

## Commands Reference

```bash
# ═══ INGEST ═══
python3 -m jkg.memory remember "text"              # factual layer
python3 -m jkg.memory remember-profile "pref"       # profile layer
python3 -m jkg.memory remember-session "id"         # episodic layer
python3 -m jkg.memory index-skill "name" "desc"     # procedural layer

# ═══ QUERY ═══
python3 -m jkg.memory query "question?"             # ALL layers
python3 -m jkg.memory query "q?" --layers profile,factual
python3 -m jkg.memory clark "query"                 # factual-only CLARK
python3 -m jkg.memory ask "question?"               # factual Q&A with LLM synthesis
python3 -m jkg.memory recall "query"                # RRF hybrid (factual)

# ═══ SESSION ═══
python3 -m jkg.memory session-start                 # dynamic context injection
python3 -m jkg.memory profile                       # view profile facts
python3 -m jkg.memory sessions "keyword"            # search sessions

# ═══ MAINTENANCE ═══
python3 -m jkg.memory stats                         # full statistics
python3 -m jkg.memory propagate                     # Value Iteration (CLARK stage 1)
python3 -m jkg.memory utility                       # recalculate memory utility scores
python3 -m jkg.memory prune 0.15                    # forget low-utility facts (dry-run)
python3 -m jkg.memory prune 0.15 --execute          # actually forget
python3 -m jkg.memory evolve                        # propose new entity/relation types
python3 -m jkg.memory gdpr-delete "name" --confirm  # full GDPR deletion

# ═══ GRAPH TRAVERSAL ═══
python3 -m jkg.memory traverse "alice" 3            # BFS from entity, depth 3
python3 -m jkg.memory path "alice" "acme"           # find path between entities
python3 -m jkg.memory resolve "alic"                # fuzzy entity resolution
python3 -m jkg.memory bridge both                   # graph↔embeddings bridge
python3 -m jkg.memory emotions                      # emotional context summary
```

---

## Installation

```bash
# Clone
git clone https://github.com/altyshalu/jessica-knowledge-graph.git
cd jessica-knowledge-graph

# Install
pip install -e .

# Dependencies
pip install sentence-transformers scipy numpy requests sqlite-vec
```

**Requirements:** Python 3.10+, SQLite 3.40+, [sqlite-vec](https://github.com/asg017/sqlite-vec)

LLM extraction uses DeepSeek API by default (set `DEEPSEEK_API_KEY` in `.env`). Pluggable — any OpenAI-compatible endpoint works.

---

## Papers & Inspiration

- [Clark's Nutcracker spatial memory](https://en.wikipedia.org/wiki/Clark%27s_nutcracker) — bio-inspiration for CLARK retrieval
- [Zep/Graphiti: Temporal Knowledge Graphs for Agentic Apps](https://arxiv.org/abs/2501.13956) — bi-temporal model
- [LongMemEval: Benchmarking Long-Context LLMs on Memory Tasks](https://arxiv.org/abs/2410.10813) — eval framework
- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) — agent memory architecture
- [Cognee: Scalable GraphRAG for AI Agents](https://www.cognee.ai/) — multi-database graph memory

---

## License

MIT © Altynai

---

<p align="center">
  <i>One database. Four layers. Zero drift.</i>
</p>
