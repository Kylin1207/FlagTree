# Remote GEMM Optimization Notes (2026-03-02)

## Scope

- Kernel: `python/tutorials/tle/04-cluster-gemm.py`
- Lowering hotspot: `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp`
- Goal: improve cluster+`tled.remote` GEMM without regressing correctness/stability.

## Process (Condensed SOP)

1. Reproduce with a fixed benchmark:
   - `python python/tutorials/tle/04-cluster-gemm.py --m 4096 --n 4096 --k 4096 --no-autotune --analyze-ir`
2. Capture TTGIR/PTX evidence:
   - compare `ttg.convert_layout`, `ttg.local_load`, `tle.remote_pointers`,
     `mapa.shared::cluster`, `ld.shared::cluster`.
3. Apply one minimal optimization at a time:
   - keep semantic barriers intact,
   - avoid mixed local/remote access experiments unless proven stable.
4. Validate in two layers:
   - targeted remote tests,
   - full TLE unit/integration regression.
5. Re-benchmark with same CLI and record before/after:
   - no-autotune result,
   - autotune best result.

## Template Mapping (for future entries)

- Context: this kernel + lowering path.
- Root Cause: TTGIR-guided bottleneck in remote A view/lowering shape.
- What Changed: view reshape + lowering/vectorization adjustments.
- Validation: tutorial benchmark + `python/test/tle/*`.
- Keep/Avoid: rules listed below.
- Follow-up: next backend-focused overlap work.

## Confirmed Good Changes

1. Remote vector-width recovery should use remote hint pointer for unmasked loads.
- Change location: `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp`
- Effect: keeps remote DSMEM load vectorization from being overly conservative when metadata carriers degrade base pointer axis info.

2. Non-autotune default remote config should be remote-path friendly.
- Change location: `python/tutorials/tle/04-cluster-gemm.py` (`_default_remote_config_for_no_autotune`)
- Default remote cfg moved to `KernelConfig(64, 256, 64, 8, 2)` when user does not explicitly set launch params.
- Observed result on 4096^3 benchmark: remote path around `36 TFLOPS` for out-of-box no-autotune run in current kernel revision.

3. TTGIR-guided remote A view reshaping (`(DOT_K, BM)` + transpose) can materially improve throughput.
- Change location: `python/tutorials/tle/04-cluster-gemm.py` (`_cluster_remote_gemm_kernel`)
- Key change:
  - build remote `local_ptr` with indices shaped `(DOT_K, BM)`,
  - `a = tl.trans(tl.load(remote_ptr))` before `tl.dot`.
- Observed result on 4096^3 benchmark:
  - no-autotune default remote path around `6.8 -> 36.0 TFLOPS`,
  - autotune best around `36 TFLOPS` in current candidate set.
- TTGIR signal:
  - remote path `ttg.convert_layout` count reduced (`4 -> 3` in measured config),
  - `ttg.local_load` increased (`1 -> 2`), indicating better lowering shape for MMA feed.

## Failed / Risky Ideas (Do Not Repeat Blindly)

1. Removing the final per-iteration cluster barrier.
- Symptom: intermittent `CUDA error: unspecified launch failure` during config sweep/autotune.
- Root cause: producer/consumer CTAs lost DSMEM lifetime synchronization in tail stage.
- Rule: do not remove barrier points unless full lifecycle proof and full config matrix are both satisfied.

2. Rank-conditional local-vs-remote load split in the hot loop (`cluster_rank == 0` local load, others remote).
- Symptom: specific configs (for example `BK=64, num_stages=3`) became unstable.
- Root cause: mixed access path inside pipelined loop introduced fragile scheduling/ordering behavior.
- Rule: keep one uniform remote load path in loop unless a stronger invariant + full matrix validation exists.

3. Replacing main K-loop with `tl.range(..., num_stages=...)` hint.
- Result: no measurable gain in this kernel; added complexity only.
- Rule: keep loop form simple unless TTGIR/PTX or benchmark clearly improves.

4. For `BK=128`, enabling `nv_mma_shared_layout` is not beneficial.
- Symptom: throughput regressed (`~37.7 -> ~33.6 TFLOPS` in tested setup).
- Rule: keep `USE_NV_MMA_SMEM_LAYOUT=False` for `BK=128` in this kernel family unless lowering changes.

5. Raising remote `DOT_K` to `32/64` degrades performance in this kernel.
- Symptom: `DOT_K=16` stayed best; `DOT_K=32/64` dropped significantly.
- Rule: keep remote `DOT_K=16` policy for current implementation.

6. Rotating producer CTA by iteration (`producer = iter_idx % cluster_size`) regresses.
- Symptom: throughput dropped from about `40.6` to about `37.3 TFLOPS`.
- Rule: keep fixed producer (`rank0`) for current kernel; do not rotate without new synchronization strategy.

7. Building one remote base pointer and using pointer arithmetic (`remote_col0 + k`) breaks correctness.
- Symptom: large numeric mismatch (around 50% elements mismatched in 4096^3 check).
- Likely reason: local pointer encoding is not a plain linear row-major byte view for this path.
- Rule: keep per-slice `local_ptr + remote` materialization; do not assume arithmetic closure on remote pointers.

8. Increasing stages beyond 3 (`num_stages=4`) heavily regresses.
- Symptom: for current best tile, performance drops to around `27 TFLOPS`.
- Rule: keep `num_stages=3` for `BK=128` remote path in this kernel family.

9. For `BK=256`, `num_stages=2` is best in current kernel.
- Symptom: `KernelConfig(32, 512, 256, 4, 2)` reaches around `41.5 TFLOPS`, while `num_stages=3` drops sharply.
- Rule: keep `BK=256` with `num_stages=2` unless pipeline/lowering changes.

10. `remote(buffered_tensor)` + `local_ptr(...)` had a metadata-loss bug in loop/live-in cloning paths (fixed).
- Historical symptom: switching to buffered-tensor remote form caused around 50% element mismatch on 4096^3 and dropped `tle.remote_pointers`.
- Root cause: `_tle_*` remote metadata was lost when Triton cloned live-in values for control-flow/loop scopes.
- Fix:
  - preserve `_tle_*` metadata in `python/triton/compiler/code_generator.py` `_clone_triton_value`,
  - keep remote metadata carriers on buffered value/type in `python/triton/experimental/tle/distributed.py` + `python/triton/experimental/tle/language/gpu/types.py`,
  - resolve metadata in `tleg.local_ptr` from type/value/handle-id cache.
- New rule: prefer `remote(buffered_tensor)` + `local_ptr` as the primary API form; use pointer-level `remote(...)` as compatibility path.

11. Cluster size `4` regresses vs cluster size `2` for current mapping.
- Symptom: tested 4-block cluster configs reached about `36-37 TFLOPS`, below 2-block best (`41+ TFLOPS`).
- Rule: keep `CLUSTER_SIZE=2` unless tile mapping/synchronization strategy is redesigned.

12. Rank-0 local-load split in the hot loop may compile and pass, but not always improve.
- Symptom: forcing `cluster_rank == 0` to read local buffer while peers read remote buffer did not improve measured throughput in this kernel revision.
- Rule: keep a single remote-load path unless new profiling evidence proves stable speedup across the target matrix.

## Practical Guardrails for Next Iteration

1. Any barrier/scheduling change must pass all entries in `REMOTE_TUNE_CONFIGS` before keeping the patch.
2. Always run:
- `python/tutorials/tle/04-cluster-gemm.py --no-autotune --analyze-ir`
- `python/tutorials/tle/04-cluster-gemm.py --autotune --tune-warmup <small> --tune-rep <small>`
- `pytest python/test/tle/unit/test_tle_distributed.py`
- `pytest python/test/tle/integration/test_tle_distributed.py`
3. TTGIR/PTX review priority:
- check remote evidence still exists (`tle.remote_pointers`, `mapa.shared::cluster`)
- compare `cp.async`, `ldmatrix.sync`, `mma.sync`, and cluster load form before/after.

## Next Focus (High Probability)

1. Reduce global->shared A staging overhead on producer CTA (current dominant gap vs baseline).
2. Improve overlap between producer fill and consumer compute without relaxing required cluster barriers.
