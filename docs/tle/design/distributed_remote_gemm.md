# TLE Remote GEMM Design (M5A-D)

## 1. Goal

Deliver a strict, testable M5 prework path for remote DSMEM GEMM reuse before full `distributed_dot` lands:

- expose mesh-rank query (`tled.shard_id`) for axis-local control flow,
- ensure remote shard-id survives lowering,
- validate cluster-local GEMM reuse pattern with correctness + lowering evidence.

This remains prework for `distributed_dot` (M5), not the final API.

## 2. M5A-D Plan and Status

| Phase | Scope | Acceptance | Status |
| --- | --- | --- | --- |
| M5A | `tled.shard_id(mesh, axis)` API + launch-axis mapping | axis name/index accepted; invalid axis fails fast; integration proves rank pattern | Completed |
| M5B | Robust remote pointer materialization | remote shard-id preserved through lowering (`tle.remote_pointers` path) for compile-time/runtime shard ids | Completed |
| M5C | Cluster remote GEMM integration kernel | producer/consumer pattern compiles with cluster launch and emits remote DSMEM evidence | Completed |
| M5D | Regression matrix and docs sync | unit + integration tests pass; design/backlog updated with implementation facts | Completed |

## 3. Target Pattern

First supported shape:

- cluster shape `(2, 1, 1)`,
- one program maps to one block (`num_ctas=1`),
- block 0 loads `A` tile to SMEM once,
- peer block consumes the same `A` tile via `tled.remote(..., scope=mesh)`.

## 4. Kernel Dataflow

For each `k_tile`:

1. `cluster_rank == 0` loads `A[m, k_tile]` from GMEM into `a_buf`.
2. all blocks wait at `tled.distributed_barrier(mesh)`.
3. producer reads `a_buf` locally; consumer reads remote `a_buf` via remote pointer path.
4. all blocks accumulate with their own `B[k_tile, n_rank]`.
5. second barrier protects single-buffer overwrite.

## 5. Key Implementation Notes

- `shard_id` decodes per-axis coordinate from launch PID using mesh `launch_shape`.
- `remote(tl.ptr, shard_id, ...)` now prefers dedicated `tle.remote_pointers` op emission.
- fallback remains for older extensions (attr carrier/addptr path).
- GEMM kernel uses explicit pointer-level remote annotation (`local_ptr` then `remote`) in consumer branch to keep remote semantics visible in IR/lowering.

## 6. Test Contract

Compile-time checks:

- inferred `cluster_dims == (2, 1, 1)`,
- TTGIR/PTX contains remote evidence (`mapa.shared::cluster` or remote marker path),
- `ttg.num-ctas == 1`.

Runtime checks:

- output matches `torch.matmul` for each cluster result slice.

Regression checks:

- `python/test/tle/unit/test_tle_distributed.py`,
- `python/test/tle/integration/test_tle_distributed.py`.

## 7. Remaining Gap to Final M5

Still pending for full `distributed_dot` milestone:

- explicit `distributed_dot` IR op + verifier (`TLE-DIST-202`),
- backend lowering from `distributed_dot` op (`TLE-DIST-203`),
- larger sharding-pattern coverage and stress tests (`TLE-DIST-204`),
- perf characterization and optimization (`TLE-DIST-205`).

## 8. Performance Status (2026-02-27)

Measured on current branch with:

```bash
conda run -n flagtree python python/tutorials/tle/04-cluster-gemm.py --check --check-lowering --warmup 5 --rep 30
```

Observed:

- `triton_gemm`: ~3.57 ms
- `cluster_tle_remote_gemm`: ~98.6 ms
- effective ratio: ~0.04x (remote path is much slower than baseline)

## 9. Root-Cause Evidence

TTGIR/PTX/LLIR inspection shows the key gap is instruction shape on operand A:

- baseline kernel A path: `cp.async` + `ldmatrix` pipeline,
- current remote consumer path: repeated `mapa.shared::cluster` + scalar `ld.shared::cluster.b16`,
- then extra staging before `tt.dot` lowering (`local_alloc/local_load` chain).

This means remote data reuse currently removes some GMEM traffic but introduces a much slower DSMEM access pattern that is not tensor-core-friendly.

Additional evidence:

- remote branch emits 8 scalar `ld.shared::cluster.b16` loads per fragment group,
- no `ld.shared::cluster.v*` vector forms observed in the hot path,
- register pressure rises (`REG:97` remote vs `REG:74` baseline by `cuobjdump --dump-resource-usage`), but this alone does not explain the full gap.

## 10. Tried-and-Rejected Low-Risk Tweaks

The following were tested and did not materially improve throughput:

- mask-free fast path (`USE_MASK=False` for aligned shapes),
- alternate SMEM layout toggle for `tleg.alloc(... nv_mma_shared_layout=...)`,
- index-expression reshaping to remove pointer `convert_layout`.

Conclusion: remaining gap is structural in lowering, not a small kernel-side tuning issue.

## 11. Next Execution Plan (for TLE-DIST-205)

1. Add IR/ptx profiler harness for remote GEMM hot blocks (instruction counts + timeline markers).
2. Introduce a remote-to-dot fast path in lowering:
   detect remote shared pointer patterns that can bypass scalar load staging.
3. Prototype a pattern-specific remote tile path:
   load peer tile once per `k0` into dot-friendly shared layout, then consume via `ttg.local_load`.
4. Re-benchmark and gate:
   require measurable uplift vs current ~98 ms and no correctness regression.

## 12. TLE-DIST-205C Increment #1 (2026-02-28)

Implemented change:

- `TleAssignLocalPointersEncoding` now infers local pointer encoding from both load/store consumers (and propagates through `tle.remote_pointers`), instead of store-only inference.

Observed effect on remote GEMM kernel IR:

- removed pointer-side remote `ttg.convert_layout` before `tt.load` in the `a_ptr_local/a_ptr_remote` path,
- `ttg.convert_layout` count reduced (from 4 to 2 in sampled TTGIR),
- remote path still uses `mapa.shared::cluster + ld.shared::cluster.b16`, so the main bottleneck remains.

Measured delta (same benchmark command as Section 8):

- remote GEMM improved from about `98.6 ms` to about `96.4 ms` (~2.3%).

Status:

- correctness unchanged,
- distributed unit/integration tests remain green,
- this is a partial gain and not yet the final remote-to-dot fast path.

## 13. Tutorial Tuning Update (2026-02-28)

`python/tutorials/tle/04-cluster-gemm.py` defaults were retuned for better remote-path throughput:

- `BM=64`, `BN=128`, `BK=32` (previously `128/64/32`),
- baseline kernel now prefers `DOT_K=32` when `BK` is divisible by 32,
- remote kernel is pinned to `DOT_K=16`.

Reason for remote `DOT_K=16` pin:

- `DOT_K=32` currently compiles and lowers, but can produce incorrect numeric results on the remote pointer path.
- This is treated as a known limitation until remote-to-dot fast path work removes the current instability source.

## 14. Execution Plan Link (2026-02-28)

Detailed optimization execution plan and step-by-step status:

- `docs/tle/design/distributed_remote_gemm_optimization_plan.md`
