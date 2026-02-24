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
- Language GPU API design: `docs/tle/design/language_gpu_api.md`
- Workflow overview: `docs/tle/workflow/README.md`
- Active backlog: `docs/tle/backlog/backlog.md`
- Lessons learned index: `docs/tle/lessons_learned/README.md`

## Document Lifecycle

1. Requirement intake: start from `docs/tle/templates/requirement_intake.md`.
2. Design proposal: add/update docs under `docs/tle/design/`.
3. Execution plan: add tasks to `docs/tle/backlog/backlog.md`.
4. Implementation workflow: follow `docs/tle/workflow/development_workflow.md`.
5. Retrospective: record outcomes in `docs/tle/lessons_learned/`.

## Ownership Rules

- Every major TLE change should update at least one of:
  - `docs/tle/design/*`
  - `docs/tle/backlog/backlog.md`
  - `docs/tle/lessons_learned/*`
- Keep links in this file valid.
- Prefer short docs with explicit file references to code and tests.
