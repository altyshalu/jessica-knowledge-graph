"""Runtime configuration and physical namespace isolation for JKG."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path


class EmbeddingConfigurationError(RuntimeError):
    """Raised when an existing database uses a different embedding index."""


def normalize_namespace(value: str | None) -> str:
    namespace = (value or os.environ.get("JKG_NAMESPACE") or "default").strip().lower()
    namespace = re.sub(r"[^a-z0-9_.-]+", "-", namespace).strip("-.")
    return namespace or "default"


def namespace_db_path(db_path: str, namespace: str) -> str:
    """Map non-default namespaces to separate SQLite files.

    File-level isolation prevents accidental cross-tenant reads even if a future
    query forgets a logical namespace predicate.
    """
    if db_path in (":memory:", "") or namespace == "default":
        return db_path
    path = Path(os.path.expanduser(db_path))
    digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:10]
    suffix = path.suffix or ".sqlite3"
    stem = path.name[:-len(path.suffix)] if path.suffix else path.name
    return str(path.with_name(f"{stem}.{namespace[:32]}.{digest}{suffix}"))


@dataclass(frozen=True)
class MemoryConfig:
    namespace: str = "default"
    db_path: str = ""
    embedding_provider: str = "hash"
    embedding_model: str = "hash-v1"
    embedding_dimensions: int = 384

    @property
    def embedding_signature(self) -> str:
        return ":".join([
            self.embedding_provider,
            self.embedding_model,
            str(self.embedding_dimensions),
        ])
