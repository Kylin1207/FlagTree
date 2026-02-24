# Lessons Learned: Distributed M3

## Context

Work scope: distributed barrier, remote DSMEM access, and cluster launch behavior for TLE.

## Lesson 1: API semantics and launch semantics must be explicit

- Issue:
  - Confusion between `num_ctas` and "peer block in cluster" semantics.
- Root cause:
  - Runtime launch behavior was implicitly tied to `num_ctas`, while API intent was mesh-driven.
- Resolution:
  - Mesh-driven cluster launch path was added for distributed API usage.
- Prevention:
  - Document launch contract in design docs and integration tests.

## Lesson 2: Runtime shard-id path is the critical correctness path

- Issue:
  - Pointer-attribute-only remote paths can be brittle through lowering.
- Root cause:
  - Attribute propagation across conversion boundaries is easy to break.
- Resolution:
  - Use runtime shard-id carrier (`tle.remote_shard_id_carrier`) and test for backend instruction evidence.
- Prevention:
  - Keep integration tests asserting both TTGIR markers and PTX cluster instructions.

## Lesson 3: Test shape must reflect cluster scheduling reality

- Issue:
  - Tests that assume peer writes without proper synchronization can produce misleading zeros.
- Root cause:
  - CTA partitioning and cluster scheduling behavior were not reflected in test setup.
- Resolution:
  - Use explicit write/barrier/read sequence and deterministic shard mapping in integration tests.
- Prevention:
  - All distributed tests should specify expected ownership and synchronization boundaries.

