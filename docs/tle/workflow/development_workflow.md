# TLE Development Workflow

## 1. Intake

1. Capture requirements with `docs/tle/templates/requirement_intake.md`.
2. Assign owner, target milestone, and acceptance tests.

## 2. Design

1. Add or update design doc in `docs/tle/design/`.
2. List impacted components:
   - Python API (`python/triton/experimental/tle`)
   - TLE dialect/lowering (`third_party/tle`)
   - backend-specific lowering (`third_party/<backend>`)

## 3. Implementation

1. Implement hardware-agnostic API in TLE Python first.
2. Implement dialect and conversion changes in `third_party/tle`.
3. Add backend-specific lowering only when required by hardware execution.
4. Keep behavior-gating explicit (feature checks, capability checks).

## 4. Validation

Run at least:

1. `conda run -n flagtree pytest python/test/tle/unit -s`
2. `conda run -n flagtree pytest python/test/tle/integration -s`
3. If C++ changed: `conda run -n flagtree ninja -C build/cmake.linux-x86_64-cpython-3.10`

## 5. Documentation and Handoff

1. Update `docs/tle/backlog/backlog.md` status.
2. Record lessons in `docs/tle/lessons_learned/`.
3. Update `AGENTS.md` if process or entry points changed.

