# Semantic Memory

This repository now includes a dedicated **semantic memory layer** for durable knowledge that should survive across sessions and not be diluted by ordinary chat history.

## What belongs here

- **Project knowledge**
  - mission
  - current architecture
  - key entities
  - operating context
- **Tool knowledge**
  - which runtimes are used
  - where data lives
  - which systems are authoritative
- **Rules**
  - safety constraints
  - formatting expectations
  - workflow rules
  - operator policies

## Why this layer exists

The factual graph is good for extracted triples, and episodic memory is good for session history. But stable knowledge about a project often needs:

- a canonical key
- explicit scope
- reinforcement over time
- predictable updates instead of duplicate facts

That is what `memory_semantic` provides.

## API

```python
from jkg import HybridMemory

hm = HybridMemory()

hm.remember_project(
    "A Zone automates startup supply-side operations before investor matching.",
    project="a-zone",
    canonical_key="project:a-zone:mission",
)

hm.remember_tool(
    "Hermes is the runtime behind Jessica.",
    tool_name="Hermes",
    canonical_key="tool:hermes:runtime",
)

hm.remember_rule(
    "Never overwrite user-owned files without explicit approval.",
    rule_name="file-safety",
    canonical_key="rule:file-safety",
)
```

## Query behavior

Semantic memory is included in unified `query(...)` by default and shows up with:

- `layer="semantic"`
- `kind in {"project", "tool", "rule"}`
- a CLARK-style fused score

## Design notes

- `canonical_key` is the main anti-duplication mechanism
- updates with the same key overwrite the statement and refresh reinforcement timestamps
- the layer is fully local in SQLite
- when Gemini is not configured, the repo falls back to deterministic local hash embeddings instead of failing
