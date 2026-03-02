# TLE Cluster Remote GEMM Backend Optimization Plan

## Goal Description

Optimize the TLE cluster remote GEMM kernel (`_cluster_remote_gemm_kernel`) in `python/tutorials/tle/04-cluster-gemm.py` through backend-focused analysis and improvements. The optimization targets the performance gap between the current remote GEMM path (~36 TFLOPS) and baseline Triton GEMM (~50 TFLOPS on H100 for 4096³ matrices).

Focus areas:
- TTGIR/PTX analysis to identify inefficiencies in remote pointer lowering
- Improve remote A staging and fetch overhead
- Better vectorization for cluster-shared loads
- Maintain all correctness guarantees (cluster barriers, numeric accuracy)

This plan continues from existing optimization work (phases S0→S3A completed) and targets phases S3B and beyond.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Performance Improvement
  - Positive Tests (expected to PASS):
    - Remote GEMM achieves higher TFLOPS than current baseline (~36 TFLOPS) for 4096³ matrices
    - At least one tuning configuration shows measurable improvement (>5% TFLOPS gain)
    - `python python/tutorials/tle/04-cluster-gemm.py --m 4096 --n 4096 --k 4096 --autotune` shows improved best config throughput
  - Negative Tests (expected to FAIL):
    - Any configuration that regresses below 30 TFLOPS for standard 4096³ workload should not be accepted as default
    - Configurations requiring >50% more registers than baseline should be flagged

- AC-2: Correctness Preservation
  - Positive Tests (expected to PASS):
    - `python python/tutorials/tle/04-cluster-gemm.py --check` passes for all supported configurations
    - `pytest python/test/tle/unit/test_tle_distributed.py -q` passes
    - `pytest python/test/tle/integration/test_tle_distributed.py -q` passes
    - `torch.testing.assert_close(c_remote, ref, atol=1e-1, rtol=1e-1)` succeeds
  - Negative Tests (expected to FAIL):
    - Any optimization that causes numeric mismatch >1e-1 absolute tolerance
    - Any optimization that triggers `CUDA error: unspecified launch failure`

- AC-3: Remote Lowering Verification
  - Positive Tests (expected to PASS):
    - `--check-lowering` flag passes (verifies `mapa.shared::cluster` or equivalent markers in PTX)
    - `cluster_dims == (2, 1, 1)` in compiled metadata
    - TTGIR contains `tle.remote_pointers` or `tle.remote_cta_id` markers
  - Negative Tests (expected to FAIL):
    - Lowering that loses remote semantics (falls back to non-cluster path)
    - Lowering that generates incorrect cluster barrier patterns

- AC-4: Documentation
  - Positive Tests (expected to PASS):
    - Optimization changes documented in `docs/tle/lessons_learned/`
    - IR analysis findings recorded with before/after TTGIR/PTX patterns
    - Failed optimization attempts documented to prevent rework
  - Negative Tests (expected to FAIL):
    - Undocumented changes to kernel or lowering code
    - Missing record of performance measurements

- AC-5: Code Quality
  - Positive Tests (expected to PASS):
    - No hardcoded magic numbers without explanatory comments
    - New lowering paths have unit test coverage
    - Changes follow existing code patterns in the repository
  - Negative Tests (expected to FAIL):
    - Introduction of new compiler warnings or errors
    - Breaking changes to existing TLE distributed API

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)

The implementation achieves near-parity with baseline Triton GEMM (~50 TFLOPS) through comprehensive backend optimizations:
- Producer A path uses `cp.async` for async global-to-shared copy with compute overlap
- Remote loads upgraded to vectorized forms (`ld.shared::cluster.v4.b32` or similar)
- Optimal shared memory layout automatically selected based on tile configuration
- Complete TTGIR/PTX analysis documented with optimization rationale
- All supported tile configurations (BM, BN, BK variants) benefit from improvements
- New lowering optimizations have dedicated unit tests

### Lower Bound (Minimum Acceptable Scope)

The implementation shows measurable improvement over current ~36 TFLOPS baseline:
- At least one identified backend inefficiency addressed
- Default configuration (`KernelConfig(64, 256, 64, 8, 2)`) shows improvement
- Changes documented in lessons learned
- Existing tests continue to pass
- No performance regressions in other configurations

### Allowed Choices

**Can use:**
- TTGIR pattern analysis and modification
- PTX-level inspection for optimization opportunities
- Shared memory layout optimization (`nv_mma_shared_layout`, `swizzled_shared_layout`)
- Producer/consumer overlap strategies within cluster barrier constraints
- Vectorization improvements for remote pointer loads
- Register allocation hints if supported by backend
- New compile-time constexpr parameters for tile tuning

**Cannot use:**
- Removal or weakening of cluster barriers (known to cause launch failures)
- `DOT_K > 16` for remote path (confirmed regression)
- Cluster size > 2 for this optimization scope
- `num_stages > 3` (confirmed regression)
- Rotating producer CTA (confirmed regression)
- Remote base pointer arithmetic (causes numeric mismatch)
- Skip hooks or bypass verification in tests
- Force vectorization from layout hint only without AxisInfo + encoding consistency
- Multiple speculative optimizations in single patch without stepwise validation

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

The optimization follows a systematic analysis-measure-optimize cycle:

```
1. BASELINE MEASUREMENT
   - Run benchmark with --analyze-ir to capture current TTGIR/PTX
   - Record key metrics: TFLOPS, register count, instruction counts

2. BOTTLENECK IDENTIFICATION
   - Compare remote vs baseline IR for discrepancies:
     * Remote path: scalar ld.shared::cluster.b16 vs baseline vectorized loads
     * Extra ttg.convert_layout ops in remote path
     * REG:97 (remote) vs REG:74 (baseline)

3. TARGETED OPTIMIZATION
   Option A: Producer Async-Copy Path (S3B from existing plan) - PRIMARY FOCUS
     - Modify producer A load to use cp.async style staging
     - Requires: careful barrier placement, pipeline depth adjustment

   Option B: Remote Load Vectorization - PARTIALLY ADDRESSED
     - AxisInfo visitor added (`TleRemotePointersOpAxisInfoVisitor`)
     - Vector unpack fixed (`unpackLLVector`)
     - Remaining: verify effective in cluster GEMM, measure impact

   Option C: Layout Optimization - PARTIALLY ADDRESSED
     - Load consumer priority implemented in `TleAssignLocalPointersEncoding.cpp`
     - Remaining: verify encoding chain optimal in cluster GEMM context

4. VALIDATION
   - Re-run benchmark, verify improvement
   - Run full test suite
   - Document findings
```

### Known Bottlenecks (Priority Order)

1. **Remote A staging overhead**: Current dominant gap
   - Producer loads A from global, all CTAs read via `mapa.shared::cluster`
   - Opportunity: async copy overlap

2. **Scalar cluster loads**: ~~`ld.shared::cluster.b16` instead of vector forms~~ **(Partially addressed)**
   - Location: `LoadStoreOpToLLVM.cpp` remote pointer path
   - Status: Fixed via `TleRemotePointersOpAxisInfoVisitor` + `unpackLLVector` for remote loads
   - Remaining: Verify vectorization effective in cluster GEMM context

3. **Layout conversion pressure**: ~~Extra `ttg.convert_layout` ops~~ **(Partially addressed)**
   - Location: TTGIR pipeline after TLE lowering
   - Status: Fixed via load consumer encoding priority in `TleAssignLocalPointersEncoding.cpp`
   - Remaining: Verify encoding chain optimal in cluster GEMM context

### Relevant References

- `python/tutorials/tle/04-cluster-gemm.py` - Target kernel implementation
- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp` - Remote load lowering
- `third_party/tle/dialect/lib/Conversion/TleToLLVM/RemotePointerUtils.cpp` - Remote pointer utilities
- `third_party/tle/dialect/lib/Transforms/TleAssignLocalPointersEncoding.cpp` - Layout assignment
- `third_party/tle/dialect/lib/Analysis/AxisInfoExt.cpp` - TLE AxisInfo visitors (including `TleRemotePointersOpAxisInfoVisitor`)
- `docs/tle/design/distributed_remote_gemm_optimization_plan.md` - Existing optimization plan
- `docs/tle/lessons_learned/distributed_remote_gemm_optimization_2026-03-02.md` - What works/fails
- `docs/tle/lessons_learned/distributed_remote_vectorization_2026-03-02.md` - Remote vectorization debug lessons

### Validation Commands

```bash
# Quick benchmark (no autotune)
python python/tutorials/tle/04-cluster-gemm.py \
  --m 4096 --n 4096 --k 4096 \
  --no-autotune --check --check-lowering --warmup 10 --rep 30

# IR analysis
python python/tutorials/tle/04-cluster-gemm.py \
  --m 4096 --n 4096 --k 4096 \
  --no-autotune --no-check --check-lowering --analyze-ir --warmup 10 --rep 30

# Full autotune benchmark
python python/tutorials/tle/04-cluster-gemm.py \
  --m 4096 --n 4096 --k 4096 \
  --autotune --no-check --check-lowering --warmup 10 --rep 30 \
  --tune-warmup 3 --tune-rep 8

# Unit tests
pytest python/test/tle/unit/test_tle_distributed.py -q

# Integration tests
pytest python/test/tle/integration/test_tle_distributed.py -q
```

## Dependencies and Sequence

### Milestones

1. **Analysis and Profiling**
   - Phase A: Establish baseline measurements with `--analyze-ir`
   - Phase B: Identify specific TTGIR/PTX patterns causing overhead
   - Phase C: Document findings in analysis notes

2. **Backend Optimization Implementation**
   - Step 1: Select optimization target based on analysis (producer async-copy / vectorization / layout)
   - Step 2: Implement changes in appropriate backend files
   - Step 3: Validate correctness (tests must pass)
   - Step 4: Measure performance impact

3. **Iteration and Documentation**
   - Phase A: If improvement achieved, document in lessons learned
   - Phase B: If no improvement or regression, document failed attempt and try next option
   - Phase C: Update default configurations if new best found

### Dependency Graph

```
Analysis → Optimization Selection → Implementation → Validation
                                                       ↓
                              ┌─────────── Success ───┴─── Failure ──────────┐
                              ↓                                               ↓
                        Document & Deploy                              Document & Iterate
                                                                              ↓
                                                                    Try next optimization
```

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead

### Known Anti-Patterns (Do Not Repeat)

Based on documented lessons learned, avoid these patterns:

| Anti-Pattern | Consequence |
|--------------|-------------|
| Remove final per-iteration cluster barrier | CUDA launch failure |
| `DOT_K > 16` for remote path | Significant performance regression |
| `num_stages > 3` | ~27 TFLOPS (regression) |
| Cluster size 4 | ~36-37 TFLOPS vs 2-block ~41+ |
| Rotating producer CTA by iteration | ~40.6 → ~37.3 TFLOPS |
| Remote base pointer + arithmetic | 50% numeric mismatch |
| `nv_mma_shared_layout` for `BK=128` | ~37.7 → ~33.6 TFLOPS |
| Force vectorization from layout hint only | Correctness regression for larger blocks |
| Mix multiple speculative optimizations in one patch | Hard to isolate root cause of failures |
| Use `unpackLLElements` for remote vector returns | Type mismatch crash; use `unpackLLVector` instead |
| Diagnose vectorization from PTX only | Miss TTGIR-level encoding/AxisInfo issues |

### Success Patterns (Reference)

| Pattern | Effect |
|---------|--------|
| Remote A view `(DOT_K, BM)` + transpose | 6.8 → 36.0 TFLOPS |
| NV MMA shared layout for BK≤64 | 17.99 → 7.14 ms |
| Per-stage pointer materialization | Reduced hot-loop overhead |
| Vector-width recovery via hint pointer | Prevents vectorization degradation |
| `TleRemotePointersOpAxisInfoVisitor` for AxisInfo propagation | Enables proper vectorization inference |
| Prioritize load consumer encodings in pointer encoding selection | Reduces `#blocked1` degradation chains |
| Use `unpackLLVector` for remote vectorized loads | Correct type handling for vector returns |
| Diagnose from both TTGIR and PTX | Catches encoding/AxisInfo issues early |
| Stepwise validation for speculative optimizations | Isolates root cause of failures |

### Follow-up Items (From Recent Vectorization Work)

1. Add dedicated deterministic runtime test for high-block remote load when platform stability permits
2. Add TTGIR parser helper to assert pointer encoding transitions directly (reduce brittle string matching)
3. Verify vectorization improvements effective in cluster GEMM benchmark context
4. Re-measure performance after vectorization fixes to establish new baseline

---

## Original Design Draft

> 优化python/tutorials/tle/04-cluster-gemm.py中的tle cluster gemm，尽量从后端优化考虑，可以分析ttgir/ptx等，用上你丰富的cuda和triton/mlir经验，做好优化记录，避免返工。

**Translation:** Optimize the TLE cluster GEMM in python/tutorials/tle/04-cluster-gemm.py, focusing on backend optimizations, analyzing ttgir/ptx etc., using rich cuda and triton/mlir experience, keeping good optimization records to avoid rework.
