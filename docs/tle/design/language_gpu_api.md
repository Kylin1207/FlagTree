# TLE `language.gpu` API Design

## Scope

This document defines the contract for APIs in `triton.experimental.tle.language.gpu` and the core data types they expose.

## Function API Matrix

| API | Contract | Current Status | Code | Tests | Gap / Notes |
| --- | --- | --- | --- | --- | --- |
| `pipeline(start, stop, step, ...)` | Loop iterator with pipeline-friendly metadata for JIT lowering. | Implemented | `python/triton/experimental/tle/language/gpu/core.py` | `python/test/tle/integration/test_tle_pipeline_e2e.py` | Extends `tl.range`; advanced scheduling semantics are backend-dependent. |
| `memory_space(input, space)` | Attach `tt.memory_space` attribute to tensor value. | Implemented | `python/triton/experimental/tle/language/gpu/core.py` | No dedicated unit test | Validation of allowed `space` values is currently minimal. |
| `alloc(shape, dtype, layout=None, scope=smem, ...)` | Allocate local buffer (`buffered_tensor`) in target memory space. | Partially implemented | `python/triton/experimental/tle/language/gpu/core.py` | `python/test/tle/integration/test_tle_pipeline_e2e.py`, `python/test/tle/integration/test_tle_local_store.py`, `python/test/tle/unit/test_tle_gpu_local_ptr.py` | `scope=tmem` is declared but not yet supported in allocation path. |
| `copy(src, dst, shape, offsets=None)` | Move data between global/tensor-descriptor and local buffered tensor. | Partially implemented | `python/triton/experimental/tle/language/gpu/core.py` | `python/test/tle/integration/test_tle_local_store.py`, `python/test/tle/integration/test_tle_tma_copy.py`, `python/test/tle/unit/test_tle_gpu_local_ptr.py` | Supports normal and TMA copy forms; async/event contract is not defined yet. |
| `local_ptr(buffer, indices)` | Materialize shared-memory pointer tensor view for `tl.load`/`tl.store`. | Implemented | `python/triton/experimental/tle/language/gpu/core.py` | `python/test/tle/unit/test_tle_gpu_local_ptr.py` | Requires explicit index tensors with identical shapes and buffer-rank arity. |

## Type API Matrix

| Type API | Contract | Current Status | Code | Notes |
| --- | --- | --- | --- | --- |
| `scope`, `smem`, `tmem` | Storage scope descriptors for buffer allocation. | Partially implemented | `python/triton/experimental/tle/language/gpu/types.py` | `smem` path is active; `tmem` is forward-declared but not operational in `alloc`. |
| `layout`, `shared_layout` | Base layout classes for local memory encoding. | Implemented | `python/triton/experimental/tle/language/gpu/types.py` | Abstract base, backend lowers concrete layout attrs. |
| `swizzled_shared_layout` | Generic swizzled shared-memory layout. | Implemented | `python/triton/experimental/tle/language/gpu/types.py` | Default row-major order. |
| `nv_mma_shared_layout` | NVIDIA MMA-optimized shared-memory layout. | Implemented | `python/triton/experimental/tle/language/gpu/types.py` | NVIDIA-specific encoding carried via generic API layer. |
| `tensor_memory_layout` | Tensor memory layout descriptor. | Partially implemented | `python/triton/experimental/tle/language/gpu/types.py` | Construction exists; allocation backend path still incomplete for non-SMEM scopes. |
| `buffered_tensor`, `buffered_tensor_type` | Symbolic local-buffer value/type used by TLE APIs. | Implemented | `python/triton/experimental/tle/language/gpu/types.py` | `remote(...)` can annotate buffered tensor metadata before `local_ptr` materialization. |

## Interaction with Distributed APIs

1. `remote(buffered_tensor, shard_id, scope)` does not directly return pointers.
2. Remote metadata is attached to `buffered_tensor`.
3. `local_ptr(...)` is the materialization point that generates remote-capable shared pointers.

Code path:
- `python/triton/experimental/tle/distributed.py`
- `python/triton/experimental/tle/language/gpu/core.py`

## Design Gaps (Backlog Links)

1. Define capability checks and diagnostics for `memory_space` values (`TLE-API-G02`).
2. Implement `tmem`/non-SMEM allocation and tests (`TLE-API-G04`).
3. Tighten `copy` contract around async and boundary behavior (`TLE-API-G05`).
