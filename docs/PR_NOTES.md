# Release Notes — JKG 8.0

JKG 8.0 focuses on memory correctness and production boundaries.

- query-dependent cross-layer ranking with preserved vector KNN order
- physically isolated namespaces
- embedding provider/model/dimension registry with explicit full reindexing
- persisted vector indexes for semantic, profile, factual, episodic, and procedural memory
- truth confidence separated from retrieval salience and access history
- explicit feedback, entity aliases/merge, and safe consolidation preview
- bounded session-start context
- corrected semantic-noise and retrieval-stability evaluation scenarios

Verified locally with 10 unit/invariant tests, 8 extended scenarios, compilation,
package build, and metadata validation.

Known limits: the package remains beta; multi-process write load, large migrations,
and full public long-context benchmark reproduction need further work.
