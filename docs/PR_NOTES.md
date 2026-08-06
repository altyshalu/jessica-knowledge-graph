# Release Notes — JKG 7.1

## Scope

JKG 7.1 introduces the unified five-layer memory model and prepares the package for public beta use.

Included in this release:
- semantic memory for durable project, tool, and rule knowledge
- unified retrieval across semantic, profile, factual, episodic, and procedural layers
- optional Gemini embeddings with deterministic local fallback
- package metadata, quickstart, and public documentation aligned to JKG 7.1
- local artifact and secret ignore rules

## Verification
Suggested local verification before push/PR:

```bash
python -m compileall jkg
python run_tests.py   # helper included in this branch
```

Current extended-test status:
- PASS: multi-layer fusion
- PASS: bi-temporal validity
- PASS: CLARK confidence update
- PASS: conflict resolution
- PASS: intentional forgetting
- PASS: self-evolving schema
- PASS: cascading deletion across all memory layers
- FAIL: semantic noise isolation

## Known limitations still present
The remaining unresolved scenario should be treated as follow-up work, not as solved behavior:

- semantic disambiguation can still mix contexts in noisy queries when using deterministic local fallback embeddings

See `docs/BUG_REPORT_AND_FIXES.md` for the detailed audit context.
