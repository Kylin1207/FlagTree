# TLE-DIST-002: Sub-Mesh Barrier Lowering Design and Status

## 1. Problem

`distributed_barrier(mesh=sub_mesh)` requires only subgroup participants to synchronize.

Historical issue:
- `tle.distributed_barrier` previously lowered subgroup usage to cluster-wide barrier.
- Predicating cluster barrier by subgroup membership is incorrect and may deadlock.

Current status:
- Frontend + IR contract + NVIDIA subgroup lowering are implemented for static subgroup descriptors.
- Validation includes dedicated row/column subgroup independence integration tests on sm90+.

## 2. Hard Constraints

1. Public API remains hardware-agnostic (`python/triton/experimental/tle` only).
2. Backend-specific behavior stays in `third_party/*`.
3. Unsupported subgroup patterns must fail at compile time with explicit diagnostics.
4. No silent fallback to full-cluster barrier for sub-mesh inputs.

## 3. Target Semantics

For `distributed_barrier(sub_mesh)`:
- Participants: CTAs whose launch-cluster coordinates belong to `sub_mesh`.
- Non-participants: semantic no-op.
- Visibility/order guarantees apply only inside the subgroup.

## 4. Implementation Status (Phased)

## Phase 0 (Completed)

- Fail-fast for sub-mesh path (explicit `NotImplementedError`), preventing semantic mismatch.

## Phase 1 (IR Contract) - Completed

- Extend `Tle_DistributedBarrierOp` with optional subgroup metadata attrs:
  - `group_kind` (`"cluster"` or `"submesh"`).
  - `group_rank` (i32).
  - `group_shape` (DenseI32ArrayAttr).
  - `group_axes` (DenseI32ArrayAttr): cluster-axis indices used by subgroup.
  - `group_mask` (DenseI32ArrayAttr, optional): static linear member IDs for small static groups.
- Add verifier rules in `third_party/tle/dialect/lib/IR/Ops.cpp`.

## Phase 2 (Frontend + Analysis) - Completed

- In `python/triton/experimental/tle/distributed.py`:
  - Infer subgroup descriptor from `device_mesh` slicing info.
  - Emit subgroup metadata only for supported static patterns.
  - Reject dynamic/irregular subgroup patterns early with actionable error.
- Supported initial pattern:
  - 1D cluster subgroup with contiguous members.
  - Example: `(2, 2, 1)` cluster where subgroup is one row or one column.

## Phase 3 (Backend Lowering) - Completed (Static Group Path)

- Implemented in `third_party/tle/dialect/lib/Conversion/TleToLLVM/DistributedBarrierOpToLLVM.cpp`:
  - `group_kind=cluster`: lower to `NVVM::ClusterArriveOp` + `NVVM::ClusterWaitOp`.
  - `group_kind=submesh`:
    - reserve shared scratch (`ttg.shared`) once per module for `counter + phase`.
    - map subgroup-leader scratch pointer with `mapa.shared::cluster`.
    - participant CTA0 performs atomic arrive/spin/release on mapped pointers.
    - non-participants skip subgroup synchronization path.
    - CTA entry/exit `gpu.barrier` ensures per-CTA thread consistency.

Lowering preconditions:
- static `group_mask` is required.
- `group_shape` product must match `group_mask` size.

## Phase 4 (Validation) - Completed

- Unit:
  - subgroup descriptor inference.
  - verifier rejection for unsupported subgroup metadata.
- Integration (sm90+):
  - compile/lowering coverage for subgroup barrier path with PTX checks is implemented.
  - existing full-cluster barrier and remote DSMEM regressions remain green.
  - dedicated row/column subgroup independence tests validate selective synchronization for both axis slices.

## 5. Ownership Map

- API + descriptor inference:
  - `python/triton/experimental/tle/distributed.py`
- Dialect op + verifier:
  - `third_party/tle/dialect/include/IR/TleOps.td`
  - `third_party/tle/dialect/lib/IR/Ops.cpp`
- Conversion:
  - `third_party/tle/dialect/lib/Conversion/TleToLLVM/DistributedBarrierOpToLLVM.cpp`
- NVIDIA helper/lowering:
  - `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/*`
- Tests:
  - `python/test/tle/unit/test_tle_distributed.py`
  - `python/test/tle/integration/test_tle_distributed.py`

## 6. Go/No-Go Gates

- Gate A: IR verifier and frontend diagnostics are complete.
- Gate B: subgroup barrier correctness test passes without timeout/deadlock on sm90+.
- Gate C: existing cluster barrier and remote DSMEM tests remain green.

## 7. Open Risks

1. DSMEM software barrier latency may be higher than native cluster barrier.
2. Shared-memory footprint for subgroup state may limit occupancy.
3. Atomic ordering semantics must be audited carefully to avoid rare hangs.
