# Contributing

Thank you for helping improve Jessica Knowledge Graph.

## Development setup

```bash
git clone https://github.com/altyshalu/jessica-knowledge-graph.git
cd jessica-knowledge-graph
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Before opening a pull request

1. Keep changes focused and document user-visible behavior.
2. Add or update tests for behavior changes.
3. Do not commit databases, generated artifacts, `.env` files, or credentials.
4. Run the verification suite:

```bash
python -m unittest discover -v
python -m compileall -q jkg examples
```

## Pull requests

Explain the problem, the chosen approach, verification performed, and any compatibility or migration impact. Keep public APIs backward compatible where practical; call out breaking changes explicitly.

## Bug reports

Include the Python and SQLite versions, operating system, a minimal reproduction, expected behavior, actual behavior, and a sanitized traceback. Never include API keys or private memory data.
