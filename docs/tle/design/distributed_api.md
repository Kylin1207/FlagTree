# TLE Distributed API Design

## Goal

Provide a hardware-agnostic distributed API in `triton.experimental.tle` and keep backend specifics in lowering/runtime.

## Non-Goals

- Do not expose vendor-specific launch primitives in Python API.
- Do not commit to `reshard` collectives and `distributed_dot` lowering before M4/M5.

## API Contract Matrix

| API | Contract | Current Status | Code | Tests | Gap / Notes |
| --- | --- | --- | --- | --- | --- |
| `device_mesh(topology)` | Build a logical mesh from named topology axes; supports slicing to sub-mesh and reshape/flatten views. | Implemented | `python/triton/experimental/tle/distributed.py` | `python/test/tle/unit/test_tle_distributed.py` | Axis naming and shape validation are strict; behavior is deterministic row-major linearization. |
| `S(axis)`, `P(axis)`, `B` | Mark split/partial/broadcast state for sharding specs. | Implemented | `python/triton/experimental/tle/distributed.py` | `python/test/tle/unit/test_tle_distributed.py` | Pure metadata markers. |
| `sharding(mesh, split, partial)` | Build validated `ShardingSpec` with split/partial/broadcast axes. | Implemented (metadata only) | `python/triton/experimental/tle/distributed.py` | `python/test/tle/unit/test_tle_distributed.py` | No communication generated at this stage. |
| `make_sharded_tensor(handle, sharding, shape)` | Wrap logical handle + sharding metadata into `ShardedTensor`. | Implemented | `python/triton/experimental/tle/distributed.py` | `python/test/tle/unit/test_tle_distributed.py` | Rank check is enforced only when `shape` is provided. |
| `distributed_barrier(mesh=None)` | Synchronize participants in current distributed scope. | Implemented as cluster barrier | `python/triton/experimental/tle/distributed.py` | `python/test/tle/integration/test_tle_distributed.py` | Sub-mesh selective barrier is not implemented yet. |
| `remote(tensor_or_buffer, shard_id, scope=None)` | Mark access to remote shard/block; supports pointer tensor and buffered tensor paths. | Implemented | `python/triton/experimental/tle/distributed.py` and `python/triton/experimental/tle/language/gpu/core.py` | `python/test/tle/integration/test_tle_distributed.py` | `buffered_tensor` flow requires `tleg.local_ptr(...)` to materialize pointer; runtime scalar `int32` shard id is supported. |
| `reshard(tensor, spec)` | Transform sharding state; compiler inserts collectives/scatter/gather as needed. | Not implemented | `python/triton/experimental/tle/distributed.py` | `python/test/tle/unit/test_tle_distributed.py` (`NotImplementedError`) | M4. |
| `distributed_dot(a, b, c=None)` | Cluster-level distributed GEMM abstraction over shard specs. | Not implemented | `python/triton/experimental/tle/distributed.py` | N/A | M5. |

## Launch and Scope Rules

1. Passing `mesh`/`scope` into distributed APIs enables mesh-driven cluster launch inference.
2. Current contract requires `num_ctas=1` when mesh-driven cluster launch is used.
3. Cluster dims are inferred from mesh axes in priority order: `cluster*` -> `block*` -> full mesh.

## Lowering and Runtime Mapping

- Frontend API:
  - `python/triton/experimental/tle/distributed.py`
- Pointer materialization (buffered tensor -> pointer tensor):
  - `python/triton/experimental/tle/language/gpu/core.py`
- TLE dialect ops and conversion:
  - `third_party/tle/dialect/include/IR/TleOps.td`
  - `third_party/tle/dialect/lib/IR/Ops.cpp`
  - `third_party/tle/dialect/lib/Conversion/TleToLLVM/*.cpp`
- NVIDIA backend remote/shared lowering:
  - `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp`
  - `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TargetInfo.cpp`
- NVIDIA launch/runtime plumbing:
  - `third_party/nvidia/backend/compiler.py`
  - `third_party/nvidia/backend/driver.py`

## Design Gaps (Backlog Links)

1. `distributed_barrier` sub-mesh synchronization semantics (`TLE-API-D06`).
2. `reshard` legality + collective mapping (`TLE-API-D07`, `TLE-API-D08`).
3. `distributed_dot` shape/sharding inference and backend lowering (`TLE-API-D09`, `TLE-API-D10`).
