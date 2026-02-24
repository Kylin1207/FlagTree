# TLE Design Docs

## Scope

This directory stores design intent, API contracts, and lowering behavior.

## Index

- `distributed_api.md`: distributed API contracts, implementation mapping, and gaps.
- `language_gpu_api.md`: `tle.language.gpu` function/type API contracts and gaps.

## Design Review Checklist

1. Problem statement and non-goals are explicit.
2. Python API contract is concrete.
3. IR/lowering impact is listed with file-level ownership.
4. Backend-specific behavior is clearly separated from hardware-agnostic API.
5. Test plan references concrete test files and acceptance criteria.
