"""Explicit embedding backends. Never mix failed model calls with hash vectors."""
from functools import lru_cache
import hashlib
import os
import re

import numpy as np

from .config import EmbeddingConfigurationError

DEFAULT_LOCAL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=2)
def _local_model(name, cache, offline):
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(name, cache_folder=cache, local_files_only=offline)
    except Exception as exc:
        raise EmbeddingConfigurationError(
            f"Cannot load local embedding model {name!r}. Install the project dependencies "
            "and download the model once before enabling JKG_OFFLINE=1. "
            "For explicit lexical-only operation use JKG_EMBEDDING_PROVIDER=hash; "
            "switching an existing database requires reindexing."
        ) from exc


class EmbeddingEngine:
    def __init__(self, provider=None, model=None, dimensions=None):
        self.provider = provider or os.environ.get("JKG_EMBEDDING_PROVIDER") or (
            "gemini" if os.environ.get("GEMINI_API_KEY") else "sentence-transformers"
        )
        defaults = {"hash": "hash-v1", "gemini": "gemini-embedding-001",
                    "sentence-transformers": DEFAULT_LOCAL_MODEL}
        if self.provider not in defaults:
            raise EmbeddingConfigurationError(f"Unknown embedding provider: {self.provider!r}")
        self.model = model or os.environ.get("JKG_EMBEDDING_MODEL") or defaults[self.provider]
        configured_dim = dimensions or os.environ.get("JKG_EMBEDDING_DIMENSIONS")
        self.backend = None
        if self.provider == "sentence-transformers":
            self.backend = _local_model(
                self.model, os.environ.get("JKG_MODEL_CACHE") or None,
                os.environ.get("JKG_OFFLINE", "").lower() in {"1", "true", "yes"},
            )
            self.dimensions = int(self.backend.get_sentence_embedding_dimension())
            if configured_dim and int(configured_dim) != self.dimensions:
                raise EmbeddingConfigurationError("Configured dimensions do not match the local model")
        else:
            self.dimensions = int(configured_dim or (768 if self.provider == "gemini" else 384))
        if self.dimensions <= 0:
            raise EmbeddingConfigurationError("Embedding dimensions must be positive")
        if self.provider == "hash" and self.model != "hash-v1":
            raise EmbeddingConfigurationError("The hash provider only supports hash-v1")
        if self.provider == "gemini" and not os.environ.get("GEMINI_API_KEY"):
            raise EmbeddingConfigurationError("GEMINI_API_KEY is required for the gemini provider")

    def encode(self, texts, **kwargs):
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        if not batch:
            return np.empty((0, self.dimensions), dtype=np.float32)
        try:
            if self.provider == "sentence-transformers":
                vectors = self.backend.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            elif self.provider == "hash":
                vectors = [self._hash(text) for text in batch]
            else:
                vectors = [self._gemini(text) for text in batch]
            array = np.asarray(vectors, dtype=np.float32)
            if array.shape != (len(batch), self.dimensions) or not np.isfinite(array).all():
                raise ValueError("Invalid embedding shape or non-finite values")
            norms = np.linalg.norm(array, axis=1, keepdims=True)
            if np.any(norms <= 0):
                raise ValueError("Embedding provider returned a zero vector")
            array = array / norms
            return array[0] if single else array
        except EmbeddingConfigurationError:
            raise
        except Exception as exc:
            raise EmbeddingConfigurationError(
                f"Embedding failed for {self.provider}/{self.model}; no fallback vectors were stored"
            ) from exc

    def _hash(self, text):
        vector = np.zeros(self.dimensions, dtype=np.float32)
        tokens = re.findall(r"[^\s,?.!;:\"'()\[\]{}]+", str(text).lower()) or [str(text)]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for index in range(0, len(digest), 4):
                vector[int.from_bytes(digest[index:index + 4], "big") % self.dimensions] += 1
        return vector

    def _gemini(self, text):
        import requests
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent",
            headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
            json={"model": f"models/{self.model}", "content": {"parts": [{"text": text}]},
                  "outputDimensionality": self.dimensions},
            timeout=float(os.environ.get("JKG_EMBEDDING_TIMEOUT_SECONDS", "8")),
        )
        response.raise_for_status()
        return response.json()["embedding"]["values"]
