#!/usr/bin/env python3
"""
JKG LongMemEval Runner
Tests JKG's hybrid architecture (BM25 + Vector + RRF) against the standard
LongMemEval-S benchmark.

Methodology (faithful to JKG architecture, comparable to agentmemory):
- Index each haystack SESSION as raw text
- BM25 search via SQLite FTS5 (same engine as JKG)
- Vector search via all-MiniLM-L6-v2 (same model as JKG)  
- RRF fusion (same algorithm as JKG)
- Metric: recall_any@K — does ANY gold session appear in top-K?

Comparison targets:
- agentmemory BM25+Vector: 95.2% R@5, 98.6% R@10
- agentmemory BM25-only:   86.2% R@5, 94.6% R@10
- MemPalace vector-only:   96.6% R@5
- Zep/Graphiti QA:         71.2% (full QA, not retrieval-only)
"""
import json, os, sys, re, sqlite3, hashlib, math, time
from collections import defaultdict
import numpy as np

# ── Config ──
DATASET_PATH = os.path.expanduser("~/.hermes/benchmark/data/longmemeval_s_cleaned.json")
BENCH_DB = os.path.expanduser("~/.hermes/benchmark/longmemeval_bench.db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
MAX_QUESTIONS = 500  # Set lower for quick tests
TOP_K_VALUES = [1, 3, 5, 10, 20]

# ── Load embedder once ──
print("Loading embedding model...")
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer(EMBEDDING_MODEL)
print("  Done.")

# ── Load dataset ──
print(f"Loading dataset ({MAX_QUESTIONS} questions)...")
with open(DATASET_PATH) as f:
    all_questions = json.load(f)
questions = all_questions[:MAX_QUESTIONS]
print(f"  Loaded {len(questions)} questions.")

# ── Results storage ──
results = {
    "total": 0,
    "skipped_abstention": 0,
    "by_type": defaultdict(lambda: {"total": 0, "hits": defaultdict(int)}),
    "recall_at_k": defaultdict(int),  # K -> hits
    "mrr_sum": 0.0,
    "ndcg_sum": defaultdict(float),
}

# ── Per-question processing ──
for qi, qdata in enumerate(questions):
    qid = qdata["question_id"]
    qtype = qdata["question_type"]
    question = qdata["question"]
    gold_sessions = set(qdata["answer_session_ids"])
    haystack_sessions = qdata["haystack_sessions"]
    haystack_ids = qdata["haystack_session_ids"]

    # Skip abstention questions (no gold answer)
    if qid.endswith("_abs") or len(gold_sessions) == 0:
        results["skipped_abstention"] += 1
        continue

    results["total"] += 1
    results["by_type"][qtype]["total"] += 1

    # ── Build fresh index per question ──
    # Use in-memory SQLite for speed
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA journal_mode=OFF")

    # Create FTS5 table for BM25
    conn.execute("""
        CREATE VIRTUAL TABLE sessions_fts USING fts5(
            session_id, content, tokenize='unicode61'
        )
    """)

    # Prepare session texts and embeddings
    session_texts = []
    for i, (sessions, sid) in enumerate(zip(haystack_sessions, haystack_ids)):
        # Concatenate all turns in the session
        turns = []
        for turn in sessions:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            turns.append(f"{role}: {content}")
        session_text = " ".join(turns)

        # Truncate very long sessions
        if len(session_text) > 8000:
            session_text = session_text[:8000]

        session_texts.append((sid, session_text))

        # Index in FTS
        conn.execute(
            "INSERT INTO sessions_fts(session_id, content) VALUES(?,?)",
            (sid, session_text)
        )

    # ── BM25 search (via FTS5) ──
    # Build FTS query from question words
    words = [w for w in re.findall(r'[^\s,?.!;:\"\'()\[\]{}]+', question) if len(w) > 1]
    fts_query = " OR ".join(f'"{w}"' for w in words) if words else question

    bm25_results = []
    try:
        rows = conn.execute(
            "SELECT session_id, rank FROM sessions_fts WHERE sessions_fts MATCH ? "
            "ORDER BY rank LIMIT 50",
            (fts_query,)
        ).fetchall()
        bm25_results = [(r[0], -r[1]) for r in rows]  # rank is negative, higher = better
    except Exception:
        pass

    # ── Vector search ──
    # Encode question + all sessions
    question_vec = embedder.encode([question], normalize_embeddings=True)[0]
    session_vecs = embedder.encode(
        [st[1] for st in session_texts], normalize_embeddings=True
    )

    # Cosine similarity
    similarities = np.dot(session_vecs, question_vec)
    vec_ranked = sorted(
        [(session_texts[i][0], float(similarities[i]))
         for i in range(len(session_texts))],
        key=lambda x: -x[1]
    )[:50]

    # ── RRF Fusion (same k=60 as JKG) ──
    RRF_K = 60
    scores = defaultdict(float)

    for rank, (sid, _) in enumerate(bm25_results):
        scores[sid] += 1.0 / (RRF_K + rank + 1)

    for rank, (sid, sim) in enumerate(vec_ranked):
        scores[sid] += 1.0 / (RRF_K + rank + 1)

    # Sort by RRF score
    rrf_ranked = sorted(scores.items(), key=lambda x: -x[1])

    # ── Also compute BM25-only and Vector-only for ablation ──
    bm25_only_ranked = [sid for sid, _ in bm25_results]
    vec_only_ranked = [sid for sid, _ in vec_ranked]

    # ── Compute metrics ──
    # recall_any@K
    for K in TOP_K_VALUES:
        # Hybrid
        top_k_hybrid = set(sid for sid, _ in rrf_ranked[:K])
        if top_k_hybrid & gold_sessions:
            results["recall_at_k"][f"hybrid@{K}"] += 1
            results["by_type"][qtype]["hits"][f"hybrid@{K}"] += 1

        # BM25-only
        top_k_bm25 = set(bm25_only_ranked[:K])
        if top_k_bm25 & gold_sessions:
            results["recall_at_k"][f"bm25@{K}"] += 1

        # Vector-only
        top_k_vec = set(vec_only_ranked[:K])
        if top_k_vec & gold_sessions:
            results["recall_at_k"][f"vec@{K}"] += 1

    # MRR (Mean Reciprocal Rank) for hybrid
    for rank, (sid, _) in enumerate(rrf_ranked):
        if sid in gold_sessions:
            results["mrr_sum"] += 1.0 / (rank + 1)
            break

    # NDCG@10 for hybrid
    dcg = 0.0
    idcg = 0.0
    for rank, (sid, _) in enumerate(rrf_ranked[:10]):
        rel = 1.0 if sid in gold_sessions else 0.0
        dcg += rel / math.log2(rank + 2)  # rank+2 because log2(1)=0

    # Ideal DCG: gold sessions at top
    n_gold = len(gold_sessions)
    for i in range(min(n_gold, 10)):
        idcg += 1.0 / math.log2(i + 2)

    if idcg > 0:
        results["ndcg_sum"]["hybrid"] += dcg / idcg

    conn.close()

    # Progress
    if (qi + 1) % 50 == 0:
        elapsed = time.time()
        n = results["total"]
        r5 = results["recall_at_k"].get("hybrid@5", 0)
        print(f"  [{qi+1}/{len(questions)}] R@5={r5/n*100:.1f}% ({r5}/{n})")

# ── Final report ──
total = results["total"]
print(f"\n{'='*60}")
print(f"🎯 JKG LongMemEval-S Results ({total} questions)")
print(f"{'='*60}")

print(f"\n📊 OVERALL RECALL (hybrid = JKG BM25+Vector+RRF):")
print(f"  {'Metric':<15} {'Hybrid':>8} {'BM25':>8} {'Vector':>8}")
print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8}")
for K in TOP_K_VALUES:
    hyb = results["recall_at_k"].get(f"hybrid@{K}", 0)
    bm = results["recall_at_k"].get(f"bm25@{K}", 0)
    vec = results["recall_at_k"].get(f"vec@{K}", 0)
    print(f"  R@{K:<13} {hyb/total*100:>7.1f}% {bm/total*100:>7.1f}% {vec/total*100:>7.1f}%")

print(f"\n  MRR (hybrid):  {results['mrr_sum']/total*100:.1f}%")
print(f"  NDCG@10 (hybrid): {results['ndcg_sum']['hybrid']/total*100:.1f}%")

print(f"\n📋 BY QUESTION TYPE (hybrid):")
print(f"  {'Type':<28} {'Count':>6} {'R@5':>8} {'R@10':>8}")
print(f"  {'-'*28} {'-'*6} {'-'*8} {'-'*8}")
for qtype in sorted(results["by_type"].keys()):
    t = results["by_type"][qtype]
    n = t["total"]
    if n == 0:
        continue
    r5 = t["hits"].get("hybrid@5", 0)
    r10 = t["hits"].get("hybrid@10", 0)
    print(f"  {qtype:<28} {n:>6} {r5/n*100:>7.1f}% {r10/n*100:>7.1f}%")

print(f"\n🌍 COMPARISON:")
print(f"  JKG Hybrid (BM25+Vec+RRF):  R@5={results['recall_at_k'].get('hybrid@5',0)/total*100:.1f}%")
print(f"  agentmemory Hybrid:         R@5=95.2%  R@10=98.6%")
print(f"  agentmemory BM25-only:      R@5=86.2%  R@10=94.6%")
print(f"  MemPalace Vector-only:      R@5=96.6%")
print(f"  Zep/Graphiti QA:            71.2% (full QA, разные метрики)")

# Save detailed results
output = {
    "benchmark": "LongMemEval-S",
    "system": "JKG 7.1",
    "architecture": "BM25(FTS5) + Vector(MiniLM-L6-v2) + RRF(k=60)",
    "questions": total,
    "skipped_abstention": results["skipped_abstention"],
    "recall_at_k": {k: results["recall_at_k"][k] for k in sorted(results["recall_at_k"])},
    "mrr": round(results["mrr_sum"] / max(1, total), 4),
    "ndcg10": round(results["ndcg_sum"]["hybrid"] / max(1, total), 4),
    "by_type": {
        qt: {
            "count": results["by_type"][qt]["total"],
            "r5": results["by_type"][qt]["hits"].get("hybrid@5", 0) / max(1, results["by_type"][qt]["total"]),
            "r10": results["by_type"][qt]["hits"].get("hybrid@10", 0) / max(1, results["by_type"][qt]["total"]),
        }
        for qt in sorted(results["by_type"])
    }
}

out_path = os.path.expanduser("~/.hermes/benchmark/longmemeval_jkg_results.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\n📁 Results saved: {out_path}")
