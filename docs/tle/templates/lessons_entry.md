# TLE Lessons Entry Template

Use this template for every major fix or optimization to keep lessons concise,
searchable, and actionable.

## 1. Context

- Scope: `<component/kernel/pass>`
- Trigger: `<bug/perf regression/new optimization>`
- Date: `<YYYY-MM-DD>`

## 2. Process (Condensed)

1. Reproduce with the smallest stable case.
2. Capture IR/PTX evidence (`ttgir`/`ptx` counters or key snippets).
3. Apply the minimal safe change.
4. Re-run targeted tests, then module regression.
5. Re-benchmark with fixed command line and report before/after.

## 3. Root Cause

- `<one sentence on mechanism>`

## 4. What Changed

- `<path:line>`: `<key change>`
- `<path:line>`: `<key change>`

## 5. Validation

- `<command>`
  - `<result>`
- `<command>`
  - `<result>`

## 6. Keep / Avoid

- Keep:
  - `<rule 1>`
  - `<rule 2>`
- Avoid:
  - `<anti-pattern 1>`
  - `<anti-pattern 2>`

## 7. Follow-up

- `<next optimization or cleanup>`
