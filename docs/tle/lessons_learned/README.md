# TLE Lessons Learned

## Index

- `distributed_m3.md`: lessons from M3 distributed API and cluster DSMEM work.
- `distributed_remote_gemm_optimization_2026-03-02.md`: concise do/don't notes from TTGIR-driven remote GEMM optimization.
- `distributed_remote_vectorization_2026-03-02.md`: root-cause and attempt log for remote load vectorization regression/debug.

## Rules

1. Record concrete issue, root cause, fix, and prevention.
2. Prefer file-level references over generic summaries.
3. Add a new entry per major incident or behavior change.
4. For every major fix/optimization, include a condensed process section:
   - Repro -> Evidence -> Minimal Change -> Validation -> Benchmark.
5. Use `docs/tle/templates/lessons_entry.md` as the default format.
