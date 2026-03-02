# Distributed Remote GEMM Optimization Plan (2026-02-28)

## 1. Goal

Improve `python/tutorials/tle/04-cluster-gemm.py` remote path while keeping:

- stage-2 fast path performance,
- stage-3 correctness and stability,
- existing distributed integration tests green.

## 2. Baseline Snapshot

Config:

```bash
python python/tutorials/tle/04-cluster-gemm.py \
  --m 4096 --n 4096 --k 4096 \
  --bm 64 --bn 256 --bk 32 --num-warps 8
```

Observed:

- stage-2: `cluster_tle_remote_gemm ~26.33 ms`
- stage-3: `cluster_tle_remote_gemm ~43.41 ms`

TTGIR/PTX evidence highlights:

- remote path has repeated `tle.distributed_barrier`,
- remote stage relies on `mapa.shared::cluster + ld.shared::cluster`,
- remote path has more `convert_layout` and `local_alloc` than baseline.

## 3. Work Breakdown

| Step | Scope | Acceptance | Status |
| --- | --- | --- | --- |
| S0 | Freeze baseline and IR evidence | benchmark + TTGIR/PTX counters recorded in docs | Completed |
| S1 | Remove hot-loop redundant index construction | no correctness regression; stage-2/3 runnable | Completed |
| S2 | A consumer pointer stream compaction | per-stage pointer materialization + `DOT_K` pointer advance; correctness pass | Completed |
| S3A | A-buffer layout tuning (NV MMA shared layout) | no correctness regression; stage-2/3 perf improves and remains stable | Completed |
| S3B | Remote A view reshaping (`(DOT_K, BM)` + transpose) | TTGIR shows reduced conversion pressure and benchmark uplift with full regression pass | Completed |
| S3 | A producer async-copy prototype (`cp.async` style path) | TTGIR shows async-copy for A producer path; correctness pass | Planned |
| S4 | Re-tune launch params for stage-3 | stage-3 improves vs baseline and remains stable | Planned |

## 4. S1 Implementation Record

Changes:

- hoisted `slot_dot` / `slot_dot1` creation outside the inner `ks` loop in remote slot-buffer path.

File:

- `python/tutorials/tle/04-cluster-gemm.py`

Validation:

```bash
conda run -n flagtree pytest python/test/tle/integration/test_tle_distributed.py -k remote_dot_dotk32_num_stages -q
```

Result:

- `2 passed`

Tutorial run checks:

```bash
python python/tutorials/tle/04-cluster-gemm.py \
  --m 4096 --n 4096 --k 4096 \
  --bm 64 --bn 256 --bk 32 --num-warps 8 \
  --num-stages 2 --no-check --check-lowering
python python/tutorials/tle/04-cluster-gemm.py \
  --m 4096 --n 4096 --k 4096 \
  --bm 64 --bn 256 --bk 32 --num-warps 8 \
  --num-stages 3 --no-check --check-lowering
```

Result:

- stage-2 remote: `~26.33 ms` (stable)
- stage-3 remote: `~43.41 ms` (stable)

## 5. S2 Implementation Record

Changes:

- materialize A-stage pointers once per stage (instead of each `ks` chunk):
  - stage0: `a_ptr_stage0 = local_ptr(...)` then `a_ptr_stage0 += DOT_K`,
  - stage1: `a_ptr_stage1 = local_ptr(...)` then `a_ptr_stage1 += DOT_K`,
- apply `remote(...)` on the stage pointer once per stage (non-owner CTA), not inside hot `ks` reconstruction,
- update tutorial lowering check to accept `tle.remote_pointers` as valid remote evidence.

Files:

- `python/tutorials/tle/04-cluster-gemm.py`

Validation:

```bash
conda run -n flagtree pytest python/test/tle/integration/test_tle_distributed.py -q
```

Result:

- `13 passed`

Tutorial run checks:

```bash
python python/tutorials/tle/04-cluster-gemm.py \
  --m 4096 --n 4096 --k 4096 \
  --bm 64 --bn 256 --bk 32 --num-warps 8 \
  --num-stages 2 --no-check --check-lowering
python python/tutorials/tle/04-cluster-gemm.py \
  --m 4096 --n 4096 --k 4096 \
  --bm 64 --bn 256 --bk 32 --num-warps 8 \
  --num-stages 3 --no-check --check-lowering
```

Result:

- stage-2 remote: `~17.99 ms` (improved from `~26.33 ms`),
- stage-3 remote: `~11.56 ms` (improved from `~43.41 ms`),
- lowering evidence check: PASS.

## 6. S3A Implementation Record

Changes:

- switched A-side shared buffers to NV MMA shared layout:
  - `tleg.alloc(..., nv_mma_shared_layout=True)` for `a_buf0`, `a_buf1`, and slot-buffer `a_buf`.
- retained S2 pointer-stream compaction and current remote-pointer lowering checks.

Files:

- `python/tutorials/tle/04-cluster-gemm.py`

Validation:

```bash
conda run -n flagtree pytest python/test/tle/integration/test_tle_distributed.py -q
```

Result:

- `13 passed`

Tutorial run checks (repeat x3 to avoid one-shot noise):

```bash
python python/tutorials/tle/04-cluster-gemm.py \
  --m 4096 --n 4096 --k 4096 \
  --bm 64 --bn 256 --bk 32 --num-warps 8 \
  --num-stages 2 --no-check --check-lowering
python python/tutorials/tle/04-cluster-gemm.py \
  --m 4096 --n 4096 --k 4096 \
  --bm 64 --bn 256 --bk 32 --num-warps 8 \
  --num-stages 3 --no-check --check-lowering
```

Stable result (3 runs):

- stage-2 remote: `~7.14 ms` (improved from S2 `~17.99 ms`),
- stage-3 remote: `~6.98 ms` (improved from S2 `~11.56 ms`),
- lowering evidence check: PASS.

## 7. Current Snapshot (2026-03-02)

Validated best config in current implementation:

- `KernelConfig(64, 256, 64, 8, 2)` (cluster size = 2, `DOT_K=16`, `A_SLOTS=2`)

Observed on 4096^3 benchmark:

- no-autotune default path: `cluster_tle_remote_gemm ~35.9-36.0 TFLOPS`
- autotune best in current candidate set: `~36.0 TFLOPS`
- TTGIR/PTX evidence remains valid (`tle.remote_pointers`, `mapa.shared::cluster`)

What was explicitly rejected in this round:

- `CLUSTER_SIZE=4` mapping (regression vs current 2-block mapping),
- `remote(buffered_tensor)+local_ptr` replacement in this kernel (correctness mismatch),
- `DOT_K > 16` and `BK=256 with num_stages=3/4` (performance regression).

## 8. S3B Implementation Record

Changes:

- restructured remote A tile view from `(BM, DOT_K)` to `(DOT_K, BM)` for
  `local_ptr` materialization, then transposed loaded fragment before `tl.dot`.
- this reduces hot-loop layout conversion pressure in TTGIR and improves remote
  path throughput in current lowering pipeline.
- updated no-autotune default remote config to `KernelConfig(64, 256, 64, 8, 2)`.

Files:

- `python/tutorials/tle/04-cluster-gemm.py`
- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp`

Validation:

```bash
python python/tutorials/tle/04-cluster-gemm.py \
  --m 4096 --n 4096 --k 4096 \
  --no-autotune --check --check-lowering --warmup 10 --rep 30
python python/tutorials/tle/04-cluster-gemm.py \
  --m 4096 --n 4096 --k 4096 \
  --no-autotune --no-check --check-lowering --analyze-ir --warmup 10 --rep 30
python python/tutorials/tle/04-cluster-gemm.py \
  --m 4096 --n 4096 --k 4096 \
  --autotune --no-check --check-lowering --warmup 10 --rep 30 \
  --tune-warmup 3 --tune-rep 8
conda run -n flagtree pytest python/test/tle/integration/test_tle_distributed.py -q
conda run -n flagtree pytest python/test/tle/unit/test_tle_distributed.py -q
```

Result:

- no-autotune default remote: `~3.82 ms (~35.98 TFLOPS)` on 4096^3
- autotune remote best (current set): `~3.82 ms (~36.00 TFLOPS)`
- integration: `33 passed`
- unit: `22 passed`

## 9. Next Step

Execute backend-focused S3:

- keep API unchanged,
- improve producer-side A staging overlap without weakening distributed barrier semantics,
- require TTGIR evidence + regression tests before adopting.
