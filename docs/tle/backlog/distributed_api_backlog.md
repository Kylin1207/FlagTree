# Distributed API Backlog (Detailed)

## 1. Planning Assumptions

- Planning date: 2026-02-24.
- Scope: `triton.experimental.tle` distributed APIs.
- Implementation principle: hardware-agnostic API in TLE, backend-specific lowering/runtime in `third_party/*`.

## 2. Milestone Overview

| Milestone | Window (Proposed) | Outcome |
| --- | --- | --- |
| M3-hardening | 2026-03-02 to 2026-03-20 | Close semantic gaps in `distributed_barrier` and `remote`; improve diagnostics and tests. |
| M4 | 2026-03-23 to 2026-04-24 | Deliver `reshard` legality + backend-agnostic collective lowering interface and first backend implementation. |
| M5 | 2026-04-27 to 2026-05-29 | Deliver `distributed_dot` contract, first supported sharding pattern, and backend lowering. |

## 3. Detailed Backlog

### 3.1 M3-Hardening

Progress update (2026-02-25):
- `TLE-DIST-001` partial completion: sub-mesh barrier path now has explicit fail-fast (`NotImplementedError`) to avoid silent semantic mismatch.
- `TLE-DIST-002A` completed: `DistributedBarrierOp` carries subgroup attrs + verifier contract, and source build passes for touched targets.
- `TLE-DIST-002B` completed: frontend infers subgroup descriptor, emits group metadata through group-aware builder API, and keeps legacy-builder fail-fast diagnostics.
- `TLE-DIST-002C` completed: NVIDIA backend lowers static `group_mask` sub-mesh barrier via software subgroup barrier (cluster-shared scratch + atomics); row/column subgroup independence tests now pass on sm90+.
- `TLE-DIST-002T` completed: unit + integration subgroup barrier coverage passes with no deadlock/timeout and no distributed regression.
- `TLE-DIST-003` completed: `remote(buffered_tensor)` now validates shard-id early, rejects duplicate remote annotation, and has explicit misuse tests for the required `local_ptr` materialization flow.

| ID | Priority | Task | Deliverable | Dependencies | Acceptance |
| --- | --- | --- | --- | --- | --- |
| TLE-DIST-001 | P0 | Define sub-mesh membership semantics for `distributed_barrier(mesh)` | Design update + API behavior spec | None | Updated design doc with group semantics and invalid usage diagnostics. |
| TLE-DIST-002 | P0 | Implement selective barrier lowering for sub-mesh | Compiler and backend changes | TLE-DIST-001 | Integration test verifies row/column group barrier independence. |
| TLE-DIST-003 | P0 | Harden `remote` buffered-tensor contract | Diagnostic and docs update | None | `remote(buffered_tensor)` requires `local_ptr` materialization path and has tests for misuse. |
| TLE-DIST-004 | P1 | Validate runtime `shard_id` constraints | Type/rank checks + error messages | None | Unit tests for scalar int32 pass and invalid types fail with clear messages. |
| TLE-DIST-005 | P1 | Add memory-ordering usage guidance in docs | Lessons + design update | TLE-DIST-003 | Docs include producer/barrier/consumer examples and pitfalls. |

`TLE-DIST-002` execution split:

| Sub-ID | Priority | Scope | Deliverable | Acceptance |
| --- | --- | --- | --- | --- |
| TLE-DIST-002A | P0 | IR contract | `Tle_DistributedBarrierOp` subgroup attrs + verifier | Invalid subgroup metadata fails in verifier with explicit error. |
| TLE-DIST-002B | P0 | Frontend inference | sub-mesh -> subgroup descriptor inference and unsupported-pattern diagnostics | Supported static subgroup patterns compile; unsupported patterns fail fast. |
| TLE-DIST-002C | P0 | NVIDIA lowering | subgroup barrier lowering path (non-cluster-wide) | row/column subgroup barrier tests pass on sm90+. |
| TLE-DIST-002T | P0 | Tests | unit + integration coverage for subgroup barriers | No deadlock/timeouts; existing distributed tests remain green. |

### 3.2 M4 (`reshard`)

| ID | Priority | Task | Deliverable | Dependencies | Acceptance |
| --- | --- | --- | --- | --- | --- |
| TLE-DIST-101 | P0 | Finalize `reshard` legality matrix | Spec with full transition table | None | All transition classes documented and unit-tested (`legal`/`illegal`). |
| TLE-DIST-102 | P0 | Introduce backend-agnostic collective IR abstraction | TLE IR op + verifier | TLE-DIST-101 | IR verifier enforces sharding and tensor-rank compatibility. |
| TLE-DIST-103 | P0 | Implement lowering for Scatter/Gather/Reduce/AllGather/AllReduce | First backend implementation | TLE-DIST-102 | Integration tests pass for each collective family. |
| TLE-DIST-104 | P1 | Capability-gated fallback strategy | Backend capability checks | TLE-DIST-103 | Unsupported backend emits actionable compile-time diagnostic. |
| TLE-DIST-105 | P1 | Add reshard cost model metadata hook | IR annotation + pass plumbing | TLE-DIST-102 | IR contains cost metadata used by pass pipeline (even if heuristic). |

### 3.3 M5 (`distributed_dot`)

Progress update (2026-02-26):
- `M5A` completed: added `tled.shard_id(mesh, axis)` with launch-axis validation and integration coverage.
- `M5B` completed: `remote(tl.ptr, shard_id, ...)` now prefers `tle.remote_pointers` emission to preserve shard-id through lowering.
- `M5C` completed: remote cluster GEMM prework kernel validated (`num_ctas=1`, `cluster_dims=(2,1,1)`) with lowering evidence and numeric checks.
- `M5D` completed: unit/integration distributed suites pass after remote GEMM + shard-id additions.

| ID | Priority | Task | Deliverable | Dependencies | Acceptance |
| --- | --- | --- | --- | --- | --- |
| TLE-DIST-201 | P0 | Define initial supported sharding patterns for `distributed_dot` | API spec + verifier | TLE-DIST-101 | Unsupported pattern errors are deterministic and include corrective hints. |
| TLE-DIST-202 | P0 | Add `distributed_dot` IR representation | Op definition + verifier | TLE-DIST-201 | IR captures operand sharding + output sharding inference. |
| TLE-DIST-203 | P0 | NVIDIA cluster DSMEM lowering for first pattern | Backend lowering pass | TLE-DIST-202 | Numeric correctness matches reference GEMM for supported pattern. |
| TLE-DIST-204 | P1 | Add accumulation/reduction correctness tests | Integration + stress tests | TLE-DIST-203 | Deterministic test pass on sm90+ for shape and shard permutations. |
| TLE-DIST-205 | P2 | Benchmark and profile path | Perf report + optimization tasks | TLE-DIST-203 | Report includes instruction-level bottleneck evidence and tracked optimization deltas. |

`M5` execution split (prework path):

| Sub-ID | Priority | Scope | Deliverable | Acceptance |
| --- | --- | --- | --- | --- |
| M5A | P0 | Mesh-rank query API | `tled.shard_id(mesh, axis)` + axis validation | Unit/integration tests verify axis decode and error paths. |
| M5B | P0 | Remote shard-id lowering robustness | `remote` pointer path emits `tle.remote_pointers` when available | Remote shard-id survives lowering in compile/runtime forms. |
| M5C | P0 | Remote GEMM prework kernel | cluster producer/consumer GEMM tile reuse test | Numeric parity + remote lowering evidence in integration test. |
| M5D | P0 | Regression + documentation | test matrix + design/backlog updates | `python/test/tle/unit` and `python/test/tle/integration` distributed suites pass. |

`TLE-DIST-205` execution split (2026-02-27 update):

| Sub-ID | Priority | Scope | Deliverable | Status |
| --- | --- | --- | --- | --- |
| TLE-DIST-205A | P1 | Benchmark harness hardening | stable benchmark command + metadata capture (ptx/llir/resource usage) | Completed |
| TLE-DIST-205B | P1 | Bottleneck attribution | documented evidence for remote hot path (`mapa + ld.shared::cluster.b16`) | Completed |
| TLE-DIST-205C | P0 | Lowering fast path prototype | remote-to-dot path that avoids scalar DSMEM gather staging for supported pattern | In Progress |
| TLE-DIST-205D | P1 | Regression/perf gate | perf delta report with pass/fail threshold and CI-ready command set | Pending |
| TLE-DIST-205E | P0 | Tutorial-kernel structured optimization execution | phase plan + incremental validation for stage2/stage3 | In Progress |

`TLE-DIST-205C` progress update (2026-02-28):

- completed increment #1: local-pointer encoding assignment now considers load consumers and `tle.remote_pointers` propagation,
- TTGIR on remote GEMM no longer inserts pointer-side remote `ttg.convert_layout` before `tt.load` for A remote path,
- benchmark delta: remote GEMM improved by ~2.3% (`~98.6 ms -> ~96.4 ms`),
- tutorial retune (`BM=64`, `BN=128`, `BK=32`) further improves remote path to about `~48.9 ms` on the sample benchmark,
- remaining gap unchanged: hot path still dominated by scalar `ld.shared::cluster.b16`,
- known issue: remote path with `DOT_K=32` still has numeric mismatch; keep remote `DOT_K=16` until lowering root cause is fixed.

`TLE-DIST-205E` progress update (2026-02-28):

- optimization execution plan documented at `docs/tle/design/distributed_remote_gemm_optimization_plan.md`,
- S0 (baseline freeze) completed with stage2/stage3 reference numbers and IR evidence,
- S1 (hoist slot-index construction out of hot `ks` loop) completed with no regression:
  - `python/test/tle/integration/test_tle_distributed.py -k remote_dot_dotk32_num_stages` passes,
  - tutorial stage2/stage3 runs remain stable.
- S2 (A consumer pointer stream compaction) completed:
  - stage pointers are materialized once per stage and advanced by `DOT_K` in-loop,
  - `remote` lowering evidence check updated to include `tle.remote_pointers`,
  - `python/test/tle/integration/test_tle_distributed.py -q` passes (`13 passed`),
  - tutorial benchmark updated (`BM=64, BN=256, BK=32`):
    - stage2 remote: `~26.33 ms -> ~17.99 ms`,
    - stage3 remote: `~43.41 ms -> ~11.56 ms`.
- S3A (A-buffer NV-MMA shared layout tuning) completed:
  - switched `a_buf0/a_buf1/a_buf` to `nv_mma_shared_layout=True`,
  - `python/test/tle/integration/test_tle_distributed.py -q` still passes (`13 passed`),
  - tutorial benchmark (repeat x3, `BM=64, BN=256, BK=32`) now stabilizes at:
    - stage2 remote: `~7.14 ms`,
    - stage3 remote: `~6.98 ms`.

## 4. Test Work Items

| ID | Scope | Task | Target File(s) |
| --- | --- | --- | --- |
| TLE-DIST-T01 | Unit | `distributed_barrier` sub-mesh legality tests | `python/test/tle/unit/test_tle_distributed.py` |
| TLE-DIST-T02 | Integration | Sub-mesh barrier row/column sync test | `python/test/tle/integration/test_tle_distributed.py` |
| TLE-DIST-T03 | Unit | Runtime `shard_id` type/rank validation tests | `python/test/tle/unit/test_tle_distributed.py` |
| TLE-DIST-T04 | Integration | `reshard` collective behavior tests | `python/test/tle/integration/test_tle_distributed.py` |
| TLE-DIST-T05 | Integration | `distributed_dot` correctness tests | `python/test/tle/integration/test_tle_distributed.py` or dedicated file |

## 5. Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Sub-mesh barrier semantics mismatch with hardware primitive | Deadlock or false synchronization | Compile-time group validation + targeted integration tests with disjoint groups. |
| Runtime shard-id carrier break across lowering passes | Wrong remote target | Keep IR marker checks in tests and add pass-level invariants. |
| `reshard` abstraction too backend-specific | API lock-in | Keep collective IR backend-agnostic and gate backend-specific behavior by capability. |
| `distributed_dot` scope creep | M5 slip | Start from one supported sharding pattern and expand incrementally. |

## 6. Exit Criteria

- M3-hardening exit:
  - `distributed_barrier` sub-mesh semantics implemented and tested.
  - `remote` contract/documentation consistency completed.
- M4 exit:
  - `reshard` implemented for core transitions with diagnostics and integration tests.
- M5 exit:
  - `distributed_dot` first production-grade path implemented with correctness and performance baseline.

## 7. References

- `docs/tle/design/distributed_api.md`
- `docs/tle/design/reviews/distributed_api_design_review_2026-02-24.md`
