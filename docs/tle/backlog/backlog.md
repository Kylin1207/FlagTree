# TLE Backlog

## Milestones

- M3: distributed barrier and remote access on cluster DSMEM.
- M4: reshard and collective communication.
- M5: distributed dot / distributed GEMM contract and lowering.

Detailed execution plan for distributed APIs:
- `docs/tle/backlog/distributed_api_backlog.md`

## API-Level Backlog

### Distributed API (`triton.experimental.tle`)

| ID | API | Milestone | Priority | Status | Owner | Next Action |
| --- | --- | --- | --- | --- | --- | --- |
| TLE-API-D01 | `device_mesh` | M3 | P0 | Done | TLE | Keep shape/slice semantics stable; add regression tests when topology parser changes. |
| TLE-API-D02 | `S/P/B` + `sharding` | M3 | P0 | Done | TLE | Extend diagnostics when sharding rank is inferred from graph metadata. |
| TLE-API-D03 | `make_sharded_tensor` | M3 | P0 | Done | TLE | Add richer validation once reshard consumes shape/split metadata. |
| TLE-API-D04 | `remote` pointer path | M3 | P0 | Done | TLE + NVIDIA | Maintain runtime `int32` shard-id carrier lowering path. |
| TLE-API-D05 | `remote` buffered-tensor path | M3 | P0 | Done | TLE + NVIDIA | Keep contract: `remote(buffered_tensor)` + `local_ptr(...)` materialization. |
| TLE-API-D06 | `distributed_barrier(mesh)` sub-mesh semantics | M3 | P1 | In Progress | TLE + backend | Implement selective sync for sub-mesh groups (not whole cluster). |
| TLE-API-D07 | `reshard` API semantics | M4 | P0 | Todo | TLE | Finalize legality matrix (`S/B/P` transitions) and Python diagnostics. |
| TLE-API-D08 | `reshard` lowering/runtime collectives | M4 | P0 | Todo | TLE + backend | Add scatter/gather/reduce/all-gather/all-reduce lowering strategy by backend capability. |
| TLE-API-D09 | `distributed_dot` semantic contract | M5 | P0 | Todo | TLE | Define shape rules, accumulator behavior, and sharding inference. |
| TLE-API-D10 | `distributed_dot` backend lowering | M5 | P0 | Todo | NVIDIA backend | Lower to cluster DSMEM multicast/reduction path and validate numerics. |

### Language GPU API (`triton.experimental.tle.language.gpu`)

| ID | API | Milestone | Priority | Status | Owner | Next Action |
| --- | --- | --- | --- | --- | --- | --- |
| TLE-API-G01 | `pipeline` | M3 | P1 | Done | TLE | Keep parity with `tl.range` options and e2e coverage. |
| TLE-API-G02 | `memory_space` | M3 | P2 | Todo | TLE | Add capability-driven validation for accepted memory-space names. |
| TLE-API-G03 | `alloc` (`scope=smem`) | M3 | P0 | Done | TLE | Keep stable; retain layout defaults and diagnostics. |
| TLE-API-G04 | `alloc` (`scope=tmem` / non-SMEM) | M4 | P1 | Todo | TLE + backend | Implement non-SMEM allocation path and backend feature gating. |
| TLE-API-G05 | `copy` normal + TMA paths | M3 | P0 | In Progress | TLE + NVIDIA | Clarify async/boundary semantics and extend negative tests. |
| TLE-API-G06 | `local_ptr` | M3 | P0 | Done | TLE | Preserve explicit index tensor contract; expand diagnostics for shape mismatches. |
| TLE-API-G07 | `scope/layout/buffered_tensor` type contracts | M3 | P1 | In Progress | TLE | Add docs + targeted unit tests for type equality/IR conversion behavior. |

## Cross-Cutting Items

| ID | Milestone | Priority | Status | Scope | Owner | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| TLE-X01 | M3-M5 | P0 | In Progress | Keep hardware-agnostic API in `python/triton/experimental/tle` and backend specifics in `third_party/*`. | TLE + backend | Required for all new APIs. |
| TLE-X02 | M3-M5 | P0 | In Progress | Keep docs and tests synchronized with implementation state. | TLE | Update `docs/tle/design/*` + this backlog in each milestone. |

## Completed Items

| ID | Date | Summary |
| --- | --- | --- |
| TLE-M3-C01 | 2026-02 | Mesh-driven cluster launch path added for distributed APIs. |
| TLE-M3-C02 | 2026-02 | Integration tests added for peer block DSMEM read in one cluster. |
