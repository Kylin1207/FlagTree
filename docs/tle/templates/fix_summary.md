# TLE Fix Summary Template

After each bug fix, provide a concise summary (recommended in the PR description
or task thread).

## Required Fields

1. `Root Cause`
   - One sentence describing the triggering mechanism (not just the symptom).

2. `Changes`
   - List modified files and key changes.
   - Recommended format: `path:line` + one short sentence.

3. `Validation`
   - List commands and outcomes (pass/fail counts).
   - Must include at least:
     - smallest affected test case(s)
     - impacted module regression run

4. `Risk and Follow-up`
   - Residual risk (`None` if not applicable).
   - Follow-up optimization/cleanup items (`None` if not applicable).

## Example

```text
Root Cause
- Missing cross-CTA synchronization after remote load; subsequent shared-scratch
  reuse overwrote data still being read by peer CTAs.

Changes
- python/test/tle/integration/test_tle_distributed.py:133
  - Added tled.distributed_barrier(mesh) after remote 2D load.

Validation
- conda run -n flagtree pytest python/test/tle/integration/test_tle_distributed.py -k remote_* -vv -s
  - 4 passed
- conda run -n flagtree pytest python/test/tle/integration/test_tle_distributed.py -vv -s
  - 33 passed

Risk and Follow-up
- Risk: None
- Follow-up: Evaluate compiler-side automatic barrier insertion to reduce
  manual synchronization in user kernels.
```
