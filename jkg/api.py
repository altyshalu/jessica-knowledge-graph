"""Bounded HTTP API for a single JKG memory store."""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import uvicorn

from . import HybridMemory, __version__ as JKG_VERSION


DB_PATH = os.environ.get("JKG_DB_PATH", "/root/.hermes/jkg_memory.db")
memory: HybridMemory | None = None
memory_lock = threading.RLock()
health_snapshot: dict[str, object] = {"status": "starting"}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global memory, health_snapshot
    with memory_lock:
        memory = HybridMemory(db_path=DB_PATH)
        stats = memory.stats()
        health_snapshot = {
            "status": "ok",
            "version": JKG_VERSION,
            "entities": stats.get("entities", 0),
            "facts": stats.get("facts", 0),
            "embedding_backend": os.environ.get("JKG_EMBEDDING_BACKEND", "local"),
        }
    yield


app = FastAPI(title="JKG API", version=JKG_VERSION, docs_url=None, redoc_url=None, lifespan=lifespan)


class RememberPayload(BaseModel):
    text: str
    ground_to: str = "sovereign-os"
    source: str = "api"


def _memory() -> HybridMemory:
    if memory is None:
        raise HTTPException(503, "JKG is not ready")
    return memory


@app.get("/health")
def health() -> dict[str, object]:
    """Liveness only: it must remain available while a bounded query runs."""
    return dict(health_snapshot)


@app.get("/stats")
def stats() -> dict:
    with memory_lock:
        return _memory().stats()


@app.get("/query")
def query(
    q: str = Query(..., min_length=1, max_length=512),
    layers: str = Query("semantic,profile,factual,episodic,procedural"),
    limit: int = Query(10, ge=1, le=20),
) -> dict:
    selected_layers = [layer.strip() for layer in layers.split(",") if layer.strip()]
    try:
        with memory_lock:
            return _memory().query(q, layers=selected_layers, limit=limit)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, "JKG query failed") from exc


@app.post("/remember")
def remember(payload: RememberPayload) -> dict:
    try:
        with memory_lock:
            result = _memory().remember(payload.text, ground_to=payload.ground_to, source=payload.source)
        return {"status": "ok", **result}
    except Exception as exc:
        raise HTTPException(500, "JKG write failed") from exc


def main() -> None:
    port = int(os.environ.get("JKG_PORT", "8410"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
