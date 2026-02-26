# Distributed API Design Review (2026-02-24)

## 1. Review Scope

- Reviewed docs:
  - `docs/tle/design/distributed_api.md`
  - `docs/tle/backlog/distributed_api_backlog.md`
- Reviewed implementation context:
  - `python/triton/experimental/tle/distributed.py`
  - `python/triton/experimental/tle/language/gpu/core.py`
  - `python/test/tle/unit/test_tle_distributed.py`
  - `python/test/tle/integration/test_tle_distributed.py`

## 2. Review Findings

### F1 (High): Sub-mesh barrier semantics are not fully executable in implementation

- Observation:
  - API accepts `mesh`, but selective synchronization behavior for sub-mesh groups is not complete.
- Impact:
  - High risk of semantic mismatch and user confusion in grouped synchronization patterns.
- Recommendation:
  - Implement `TLE-DIST-001` and `TLE-DIST-002` before enabling more distributed collectives.
- Backlog link:
  - `TLE-DIST-001`, `TLE-DIST-002`, `TLE-API-D06`.
- Update:
  - Follow-up phased implementation plan documented in `docs/tle/design/distributed_barrier_submesh.md`.

### F2 (High): `reshard` is undefined operationally beyond placeholder

- Observation:
  - Transition matrix exists conceptually, but no executable IR contract or backend mapping yet.
- Impact:
  - Blocks distributed workflow expansion; sharding metadata cannot transition safely.
- Recommendation:
  - Prioritize `TLE-DIST-101` through `TLE-DIST-104` in M4 and enforce legality checks first.
- Backlog link:
  - `TLE-DIST-101`, `TLE-DIST-102`, `TLE-DIST-103`, `TLE-DIST-104`.

### F3 (Medium): Remote access contract depends on implicit memory-order expectations

- Observation:
  - Correct remote DSMEM access requires explicit barrier ordering, but misuse diagnostics are limited.
- Impact:
  - Can cause intermittent wrong reads (stale or uninitialized values).
- Recommendation:
  - Add API guidance and misuse-focused tests (`producer -> barrier -> remote read` pattern).
- Backlog link:
  - `TLE-DIST-003`, `TLE-DIST-005`.

### F4 (Medium): Backend capability model needs first-class compiler contract

- Observation:
  - Current capability checks are partly implicit in backend/runtime behavior.
- Impact:
  - Unsupported operations may fail late or with unclear diagnostics.
- Recommendation:
  - Introduce explicit capability checks in semantic/lowering stage with targeted error messages.
- Backlog link:
  - `TLE-DIST-104`, `TLE-API-D08`, `TLE-API-D10`.

### F5 (Low): Distributed dot entrypoint lacks phased pattern boundary

- Observation:
  - `distributed_dot` is postponed, but initial supported pattern is not codified in existing docs/tests.
- Impact:
  - Scope creep risk in M5 planning.
- Recommendation:
  - Freeze one initial sharding pattern and reject others with deterministic diagnostics.
- Backlog link:
  - `TLE-DIST-201`, `TLE-DIST-203`.

## 3. Decisions

1. Keep API hardware-agnostic and place backend specifics behind lowering capability gates.
2. Sequence remains:
   - M3 hardening first (barrier/remote correctness),
   - then M4 (`reshard`),
   - then M5 (`distributed_dot`).
3. `remote(buffered_tensor)` + `local_ptr(...)` materialization model remains the canonical path for remote shared-memory pointer acquisition.

## 4. Review Checklist Status

| Item | Status | Notes |
| --- | --- | --- |
| API semantics explicit | Partial | `reshard` and `distributed_dot` now documented as phased design targets. |
| Backend abstraction boundary clear | Partial | Capability model defined; implementation to follow. |
| Testability of each API | Partial | M3 mostly covered; M4/M5 tests planned but not implemented. |
| Failure-mode diagnostics defined | Partial | Defined in design; enforcement backlog remains. |

## 5. Sources Used in Review

- CUDA C Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- CUDA Runtime API: https://docs.nvidia.com/cuda/archive/13.0.0/cuda-runtime-api/group__CUDART__HIGHLEVEL.html
- PTX ISA: https://docs.nvidia.com/cuda/archive/12.4.0/parallel-thread-execution/index.html
- PyTorch DTensor: https://docs.pytorch.org/docs/stable/distributed.tensor.html
- PyTorch DeviceMesh recipe: https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html
- JAX shard_map: https://docs.jax.dev/en/latest/notebooks/shard_map.html
- OpenXLA Shardy: https://openxla.org/shardy/sharding_representation
- NCCL collectives: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html
