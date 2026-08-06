# JKG 8.0 — Architecture Audit and Remaining Work

## Current Situation
JKG 8 keeps five memory layers in an isolated SQLite database per namespace. The current automated evaluation passes all 8 edge-case scenarios plus 9 unit/invariant tests.

## Bug Checklist & Remediation Plan

### 1. Hardcoded Heavy Dependencies (`sentence-transformers` & PyTorch)
- **Status:** Resolved in 7.1. Local hash embeddings are the default fallback and CLARK cosine scoring uses NumPy.
- **What is it:** The system forces a 500MB+ PyTorch download on `pip install`, making it unusable as a lightweight local memory solution.
- **Why it exists:** The code hardcodes `SentenceTransformer("all-MiniLM-L6-v2")` in `jkg/memory.py` instead of allowing API-based embeddings.
- **Why fix it:** Local memory for agents should be fast to deploy and not consume massive VRAM/disk space if the user prefers API embeddings.
- **Expected Behavior:** System should dynamically support lightweight external APIs (like Gemini embeddings with 768 dimensions) or fallback to local models only if explicitly requested.

### 2. GDPR Cascading Delete Flaw
- **Status:** Resolved in 7.1 with cross-layer deletion, FTS rebuilding, and sanitized audit entries.
- **What is it:** `gdpr_delete(entity_name)` removes graph nodes and vectors, but leaves traces of the entity in FTS5 (Full-Text Search) shadow tables and episodic logs.
- **Why it exists:** The SQL transaction deletes from `entities`, `facts`, and `facts_vec`, but forgets to target `facts_fts` and `memory_sessions`.
- **Why fix it:** "Right to be forgotten" is compromised. Security and privacy flaw.
- **Expected Behavior:** A single GDPR delete command must scrub the entity string from ALL tables, including raw text transcripts and FTS indexes.

### 3. Missing Conflict Resolution (Schizophrenic Context)
- **Status:** Resolved for the current automated conflict scenario.
- **What is it:** Storing "User loves JS" and then "User hates JS" results in both facts being retrieved simultaneously.
- **Why it exists:** The system appends new facts without a semantic invalidation step for polar opposite predicates.
- **Why fix it:** The LLM receives contradictory context, leading to hallucinations and poor reasoning.
- **Expected Behavior:** Before inserting a fact, the system should check for high-confidence contradictions and either overwrite the old fact or explicitly state a change in preference.

### 4. Bi-Temporal Conflation
- **Status:** Resolved for the current automated location timeline scenario.
- **What is it:** The system cannot distinguish between past states (e.g., "Lived in Bishkek in 2023") and present states ("Lives in SF in 2026") when queried.
- **Why it exists:** The LLM prompt for extracting facts treats temporal data as a generic string attached to the relation, rather than a filterable bounding box.
- **Why fix it:** Temporal logic is crucial for autonomous agents tracking user state over time.
- **Expected Behavior:** Queries specifying a timeframe (e.g., "in 2024") should only return facts valid during that timeframe.

### 5. Semantic Noise Isolation
- **Status:** Resolved for the deterministic regression scenario in 8.0.
- **What is it:** Ambiguous entities can still leak unrelated context into the top results when deterministic local fallback embeddings are used.
- **Current mitigation:** Stopword filtering, domain-aware episodic scoring, lexical gating, and calibrated layer fusion.
- **Expected Behavior:** Context-specific queries should prioritize the correct entity sense and suppress unrelated meanings.
