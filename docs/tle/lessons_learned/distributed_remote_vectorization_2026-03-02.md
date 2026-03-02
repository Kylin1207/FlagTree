# Lessons Learned: Remote Vectorization Debug (2026-03-02)

## 1. Context

- Scope: TLE remote pointer path (`tle.remote_pointers`) to NVIDIA DSMEM load lowering.
- Trigger: Remote loads stayed scalar (`vec=1`) in TTGIR/PTX despite vector-friendly shapes.
- Date: 2026-03-02

## 2. Process (Condensed)

1. Reproduced with smallest stable kernels in `python/test/tle/integration/test_tle_distributed.py`:
   - `_remote_const_shard_load_kernel`
   - `_remote_const_shard_vectorized_load_kernel`
2. Collected TTGIR/PTX evidence:
   - checked `#blocked/#blocked1`, `ttg.convert_layout`, `tle.remote_pointers`
   - checked `ld.shared::cluster.*` forms in PTX.
3. Applied minimal changes in order:
   - improved remote vector-size inference path,
   - fixed remote load vector unpack bug,
   - fixed pointer encoding selection to prefer load consumers,
   - added AxisInfo visitor for `RemotePointersOp`.
4. Re-ran targeted + regression tests.

## 3. Attempt Log (What Worked / What Failed)

1. Attempt: widen vector hint using load pointer layout directly in `LoadStoreOpToLLVM`.
- Observation: PTX could emit `ld.shared::cluster.v2/.v4`, but correctness regressed for larger blocks.
- Decision: reverted this unsafe shortcut; do not force vectorization from layout hint only.

2. Attempt: enable vectorized remote load path.
- Observation: hit lowering crash due to type mismatch when unpacking vector result as struct.
- Root cause: remote path used `unpackLLElements` for vector return.
- Fix: use `unpackLLVector` in remote vectorized load branch.

3. Attempt: inspect encoding flow for `local_pointers -> remote_pointers -> tt.load`.
- Observation: pointer tensors could degrade to `#blocked1` and then recover via `ttg.convert_layout` before load.
- Root cause: encoding assignment pass favored store-side encodings or ambiguous consumer resolution.
- Fix: prioritize load consumer encodings, insert value-side `convert_layout` for stores when needed.

4. Attempt: inspect AxisInfo for remote pointers.
- Observation: `RemotePointersOp` had no dedicated AxisInfo visitor, so info could drop to pessimistic state.
- Fix: added `TleRemotePointersOpAxisInfoVisitor` and propagated source axis info for uniform shard-id usage.

## 4. Root Cause

- Remote vectorization was blocked by a combination of metadata/analysis loss (`RemotePointersOp` AxisInfo gap) and encoding-selection drift (`#blocked1` pointer chains), not by a single backend PTX emission rule.

## 5. What Changed

- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp`
  - remote load path now safely unpacks vector returns via `unpackLLVector`.
  - remote vector-size recovery uses TLE remote hint pointer + AxisInfo-based inference (without forcing layout-only vectorization).
- `third_party/tle/dialect/lib/Transforms/TleAssignLocalPointersEncoding.cpp`
  - consumer encoding collection split into load/store.
  - pointer encoding selection now prefers load consumers.
  - store value gets bridging `ttg.convert_layout` when pointer/value encodings differ.
  - remote pointer chain types are updated recursively.
- `third_party/tle/dialect/lib/Analysis/AxisInfoExt.cpp`
  - added `TleRemotePointersOpAxisInfoVisitor`.
- `third_party/tle/dialect/lib/Conversion/TleToLLVM/RemotePointerUtils.cpp`
  - removed over-restrictive TLE-op-name gate in vector-size hint helpers.
- `python/test/tle/integration/test_tle_distributed.py`
  - added `test_remote_const_shard_load_high_block_encoding_no_regression` to lock encoding-shape regression (`#blocked1 -> #blocked` convert chain).

## 6. Validation

- `conda run --no-capture-output -n flagtree pytest python/test/tle/integration/test_tle_distributed.py -q`
  - `34 passed`
- `conda run --no-capture-output -n flagtree pytest python/test/tle/unit/test_tle_distributed.py -q`
  - `22 passed`

## 7. Keep / Avoid

- Keep:
  - Diagnose remote vectorization from both TTGIR and PTX (not PTX only).
  - Keep encoding assignment logic close to TLE (`third_party/tle`) and add minimal backend hooks only when necessary.
  - Prefer compile-level regression checks when runtime behavior is architecture-sensitive.
- Avoid:
  - Forcing vectorization purely from layout hint; require consistent AxisInfo + encoding + correctness evidence.
  - Mixing multiple speculative optimizations in one patch (vector hint + encoding + runtime behavior) without stepwise validation.

## 8. Follow-up

1. Add a dedicated deterministic runtime test for high-block remote load when platform stability permits (currently compile-level guard is in place).
2. Add a TTGIR parser helper to assert pointer encoding transitions directly (to reduce brittle string matching in tests).
