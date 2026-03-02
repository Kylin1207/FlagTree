# TLE Design Docs

## Scope

This directory stores design intent, API contracts, and lowering behavior.

## Index

- `distributed_api.md`: distributed API detailed contracts, lowering model, and test strategy.
- `distributed_barrier_submesh.md`: `TLE-DIST-002` subgroup barrier lowering phased plan.
- `distributed_remote_gemm.md`: cluster-local GEMM optimization design using `remote` + `distributed_barrier`.
- `language_gpu_api.md`: `tle.language.gpu` function/type API contracts and gaps.
- `reviews/README.md`: design review index.
- `reviews/distributed_api_design_review_2026-02-24.md`: distributed API design review findings and decisions.

## Design Review Checklist

1. Problem statement and non-goals are explicit.
2. Python API contract is concrete.
3. IR/lowering impact is listed with file-level ownership.
4. Backend-specific behavior is clearly separated from hardware-agnostic API.
5. Test plan references concrete test files and acceptance criteria.
