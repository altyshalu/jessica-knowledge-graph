# PR Notes

## Scope
This branch is intended for release preparation of the `fix/architecture-flaws` work already present in the repository.

Included in this final cleanup:
- packaging metadata alignment to JKG 7.1
- README clarification for optional Gemini embeddings setup
- local artifact ignore rules for verification outputs
- quickstart copy cleanup to match current package version

## Verification
Suggested local verification before push/PR:

```bash
python -m compileall jkg
python run_tests.py   # helper included in this branch
```

Current local status on this branch:
- PASS: multi-layer fusion
- FAIL: bi-temporal validity
- PASS: CLARK confidence update
- PASS: conflict resolution
- PASS: intentional forgetting
- PASS: self-evolving schema
- FAIL: GDPR cascading delete completeness
- FAIL: semantic noise isolation

## Known limitations still present
These remain unresolved on this branch and should be treated as follow-up work, not as solved problems:
- temporal reasoning still misses the 2024 Bishkek case in the current test harness
- GDPR delete still leaves traces in at least some query paths
- semantic disambiguation still mixes contexts in noisy queries

See `docs/BUG_REPORT_AND_FIXES.md` for the detailed audit context.
