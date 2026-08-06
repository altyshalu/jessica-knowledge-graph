"""JKG 8 — local-first, five-layer memory for Python applications."""
from .config import EmbeddingConfigurationError, MemoryConfig
from .lifecycle import ContextBudget, FeedbackSignal, MemoryAction
from .memory import HybridMemory

__version__ = "8.0.0"
__all__ = [
    "HybridMemory", "MemoryConfig", "EmbeddingConfigurationError",
    "ContextBudget", "FeedbackSignal", "MemoryAction",
]
