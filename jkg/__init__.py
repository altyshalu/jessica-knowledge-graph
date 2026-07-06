"""
JKG 7.1 — Unified Memory Fabric.
Semantic + Profile + Factual + Episodic + Procedural memory.
Single SQLite database. CLARK retrieval (Value Iteration + A*).
Dynamic session-start context injection. Sync bridges.
"""
from .memory import HybridMemory

__version__ = "7.1.0"
__all__ = ["HybridMemory"]
