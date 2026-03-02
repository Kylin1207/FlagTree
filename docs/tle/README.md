# TLE Documentation Hub

This folder is the canonical workspace for TLE planning and execution docs.

## Directory Index

- `design/`: architecture and API design docs.
- `workflow/`: development and review workflow docs.
- `backlog/`: prioritized tasks and milestones.
- `lessons_learned/`: postmortems and implementation notes.
- `templates/`: reusable templates for requirement intake and reviews.

## Entry Points

- Design overview: `docs/tle/design/README.md`
- Distributed API design: `docs/tle/design/distributed_api.md`
- Distributed sub-mesh barrier plan: `docs/tle/design/distributed_barrier_submesh.md`
- Distributed remote GEMM design: `docs/tle/design/distributed_remote_gemm.md`
- Distributed remote GEMM optimization plan: `docs/tle/design/distributed_remote_gemm_optimization_plan.md`
- Design reviews index: `docs/tle/design/reviews/README.md`
- Distributed API review: `docs/tle/design/reviews/distributed_api_design_review_2026-02-24.md`
- Language GPU API design: `docs/tle/design/language_gpu_api.md`
- Workflow overview: `docs/tle/workflow/README.md`
- Agentic playbook: `docs/tle/workflow/agentic_development_playbook.md`
- Agentic workflow flowchart: `docs/tle/workflow/agentic_standard_workflow.drawio`
- Active backlog: `docs/tle/backlog/backlog.md`
- Distributed API detailed backlog: `docs/tle/backlog/distributed_api_backlog.md`
- Lessons learned index: `docs/tle/lessons_learned/README.md`
- Fix summary template: `docs/tle/templates/fix_summary.md`
- Lessons entry template: `docs/tle/templates/lessons_entry.md`

## Document Lifecycle

1. Requirement intake: start from `docs/tle/templates/requirement_intake.md`.
2. Design proposal: add/update docs under `docs/tle/design/`.
3. Execution plan: add tasks to `docs/tle/backlog/backlog.md`.
4. Implementation workflow: follow `docs/tle/workflow/development_workflow.md`.
5. Post-fix summary: fill `docs/tle/templates/fix_summary.md` after every fix.
6. Lessons entry: fill `docs/tle/templates/lessons_entry.md` for every major fix/optimization.
7. Retrospective: record outcomes in `docs/tle/lessons_learned/`.

## Ownership Rules

- Every major TLE change should update at least one of:
  - `docs/tle/design/*`
  - `docs/tle/backlog/backlog.md`
  - `docs/tle/lessons_learned/*`
- Keep links in this file valid.
- Prefer short docs with explicit file references to code and tests.
