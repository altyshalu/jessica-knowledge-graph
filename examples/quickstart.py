#!/usr/bin/env python3
"""A no-API-key example. The first run downloads the local embedding model.

Install with `python -m pip install -e .`, then run `python examples/quickstart.py`.
The example uses a temporary database and does not modify your personal memory.
"""
import tempfile
from pathlib import Path

from jkg import HybridMemory


def main():
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "example.sqlite3")
        memory = HybridMemory(db_path=path)
        try:
            fact = memory.remember("Алина работает в компании Acme с 2023 года")
            profile = memory.remember_profile("Alice prefers concise answers")
            memory.remember_rule(
                "Create a backup copy before modifying important files.", rule_name="backups"
            )
            assert fact["facts"] == 1
            assert profile["stored"] == 1
            print("Employment:", memory.query("Где работает Алина?", layers=["factual"])["results"])
            print("Preference:", memory.query("How does Alice prefer to communicate?", layers=["profile"])["results"])
            print("Cross-language search:", memory.query("Что делать перед изменением важных файлов?", layers=["semantic"])["results"])
        finally:
            memory.conn.close()

        reopened = HybridMemory(db_path=path)
        try:
            assert reopened.query("Alice", layers=["profile"])["results"]
            print("Memory survived reopening the database.")
        finally:
            reopened.conn.close()


if __name__ == "__main__":
    main()
