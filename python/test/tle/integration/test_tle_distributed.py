# flagtree tle
"""
TLE Distributed (M3) Integration Tests

These tests validate:
- `tle.distributed_barrier` can be used in JIT kernels.
- `tle.remote` can annotate shared-memory pointers and participate in load/store.

Current lowering targets NVIDIA Hopper cluster instructions, so tests run only
on CUDA devices with compute capability >= 9.0.
"""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle as tled
import triton.experimental.tle.language.gpu as tleg

BLOCK_CLUSTER_MESH = tled.device_mesh({"block_cluster": [("cluster_x", 2)]})
BLOCK_CLUSTER_MESH_2X2 = tled.device_mesh({"block_cluster": [("cluster_x", 2), ("cluster_y", 2)]})
BLOCK_CLUSTER_SUBMESH_ROW0 = BLOCK_CLUSTER_MESH_2X2[0, :]
BLOCK_CLUSTER_SUBMESH_ROW1 = BLOCK_CLUSTER_MESH_2X2[1, :]
BLOCK_CLUSTER_SUBMESH_COL0 = BLOCK_CLUSTER_MESH_2X2[:, 0]
BLOCK_CLUSTER_SUBMESH_COL1 = BLOCK_CLUSTER_MESH_2X2[:, 1]


def _require_hopper_cuda():
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA GPU")
    major, _minor = torch.cuda.get_device_capability()
    if major < 9:
        pytest.skip("Requires NVIDIA Hopper (sm90+) for cluster instructions")


@pytest.fixture(scope="module", autouse=True)
def _cuda_guard():
    _require_hopper_cuda()


@triton.jit
def _distributed_barrier_copy_kernel(x_ptr, out_ptr, numel, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < numel
    vals = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tled.distributed_barrier()
    tl.store(out_ptr + offs, vals, mask=mask)


@triton.jit
def _remote_roundtrip_kernel(x_ptr, out_ptr, numel, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < numel
    vals = tl.load(x_ptr + offs, mask=mask, other=0.0)

    smem = tleg.alloc([BLOCK], dtype=tl.float32, layout=None, scope=tleg.smem, nv_mma_shared_layout=False)
    remote_smem = tled.remote(smem, 0)
    local_ptr = tleg.local_ptr(remote_smem, (tl.arange(0, BLOCK), ))
    tl.store(local_ptr, vals, mask=mask)

    out_vals = tl.load(local_ptr, mask=mask, other=0.0)
    tl.store(out_ptr + offs, out_vals, mask=mask)


@triton.jit
def _remote_peer_smem_kernel(out_ptr, shard_id_ptr, mesh: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    pid = tl.program_id(0)
    vals = tl.cast(offs + pid * BLOCK, tl.float32)

    smem = tleg.alloc([BLOCK], dtype=tl.float32, layout=None, scope=tleg.smem, nv_mma_shared_layout=False)
    local_ptr = tleg.local_ptr(smem, (offs, ))
    tl.store(local_ptr, vals)
    tled.distributed_barrier(mesh)

    shard_id = tl.load(shard_id_ptr + pid)
    remote_smem = tled.remote(smem, shard_id, scope=mesh)
    peer_ptr = tleg.local_ptr(remote_smem, (offs, ))
    peer_vals = tl.load(peer_ptr)
    tl.store(out_ptr + pid * BLOCK + offs, peer_vals)


@triton.jit
def _submesh_barrier_lowering_kernel(out_ptr, mesh: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    pid = tl.program_id(0)
    vals = tl.full((BLOCK, ), pid, tl.int32)
    tled.distributed_barrier(mesh)
    tl.store(out_ptr + pid * BLOCK + offs, vals)


@triton.jit
def _distributed_barrier_multiblock_counter_kernel(counter_ptr, out_ptr, mesh: tl.constexpr):
    pid = tl.program_id(0)
    cluster_id = pid // 2
    counter_lane_ptr = counter_ptr + cluster_id

    tl.atomic_add(counter_lane_ptr, 1)
    tled.distributed_barrier(mesh)

    seen = tl.load(counter_lane_ptr)
    tl.store(out_ptr + pid, seen)


@triton.jit
def _submesh_row_group_barrier_kernel(
    counter_row0_ptr,
    counter_row1_ptr,
    out_row0_ptr,
    out_row1_ptr,
    row0_mesh: tl.constexpr,
    row1_mesh: tl.constexpr,
):
    pid = tl.program_id(0)

    if pid < 2:
        tl.atomic_add(counter_row0_ptr, 1)
        tled.distributed_barrier(row0_mesh)
        seen_row0 = tl.load(counter_row0_ptr)
        tl.store(out_row0_ptr + pid, seen_row0)
    else:
        tl.store(out_row0_ptr + pid, -1)

    if pid >= 2:
        tl.atomic_add(counter_row1_ptr, 3)
        tled.distributed_barrier(row1_mesh)
        seen_row1 = tl.load(counter_row1_ptr)
        tl.store(out_row1_ptr + pid, seen_row1)
    else:
        tl.store(out_row1_ptr + pid, -1)


@triton.jit
def _submesh_col_group_barrier_kernel(
    counter_col0_ptr,
    counter_col1_ptr,
    out_col0_ptr,
    out_col1_ptr,
    col0_mesh: tl.constexpr,
    col1_mesh: tl.constexpr,
):
    pid = tl.program_id(0)
    is_col0 = (pid & 1) == 0

    if is_col0:
        tl.atomic_add(counter_col0_ptr, 1)
        tled.distributed_barrier(col0_mesh)
        seen_col0 = tl.load(counter_col0_ptr)
        tl.store(out_col0_ptr + pid, seen_col0)
    else:
        tl.store(out_col0_ptr + pid, -1)

    if not is_col0:
        tl.atomic_add(counter_col1_ptr, 5)
        tled.distributed_barrier(col1_mesh)
        seen_col1 = tl.load(counter_col1_ptr)
        tl.store(out_col1_ptr + pid, seen_col1)
    else:
        tl.store(out_col1_ptr + pid, -1)


class TestTLEDistributed:

    def test_distributed_barrier_copy(self):
        block = 128
        numel = block
        x = torch.randn(numel, device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)

        _distributed_barrier_copy_kernel[(1, )](x, out, numel, BLOCK=block)
        torch.testing.assert_close(out, x, atol=1e-6, rtol=1e-6)

    def test_remote_roundtrip(self):
        block = 128
        numel = block
        x = torch.randn(numel, device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)

        _remote_roundtrip_kernel[(1, )](x, out, numel, BLOCK=block)
        torch.testing.assert_close(out, x, atol=1e-6, rtol=1e-6)

    def test_remote_read_peer_smem_same_cluster(self):
        block = 64
        grid = 2
        cluster_size = 2
        num_programs = grid * cluster_size
        out = torch.empty((num_programs * block, ), device="cuda", dtype=torch.float32)
        shard_id = torch.tensor([1, 0, 1, 0], device="cuda", dtype=torch.int32)

        compiled = _remote_peer_smem_kernel.warmup(
            out,
            shard_id_ptr=shard_id,
            mesh=BLOCK_CLUSTER_MESH,
            BLOCK=block,
            grid=(grid, ),
            num_ctas=1,
            num_warps=4,
        )
        assert compiled.metadata.cluster_dims == (2, 1, 1)
        assert "tle.remote_shard_id_carrier" in compiled.asm["ttgir"]
        assert "\"ttg.num-ctas\" = 1" in compiled.asm["ttgir"]
        assert "mapa.shared::cluster" in compiled.asm["ptx"]

        _remote_peer_smem_kernel[(grid, )](
            out, shard_id_ptr=shard_id, mesh=BLOCK_CLUSTER_MESH, BLOCK=block, num_ctas=1, num_warps=4
        )
        expected_chunks = []
        for pid in range(num_programs):
            peer_pid = pid ^ 1
            expected_chunks.append(
                torch.arange(peer_pid * block, (peer_pid + 1) * block, device="cuda", dtype=torch.float32)
            )
        expected = torch.cat(expected_chunks, dim=0)
        torch.testing.assert_close(out, expected, atol=0.0, rtol=0.0)

    @pytest.mark.parametrize("submesh", [BLOCK_CLUSTER_SUBMESH_ROW0, BLOCK_CLUSTER_SUBMESH_COL0])
    def test_distributed_barrier_submesh_lowering(self, submesh):
        block = 32
        out = torch.empty((4 * block, ), device="cuda", dtype=torch.int32)

        compiled = _submesh_barrier_lowering_kernel.warmup(
            out,
            mesh=submesh,
            BLOCK=block,
            grid=(1, ),
            num_ctas=1,
            num_warps=4,
        )
        assert compiled.metadata.cluster_dims == (2, 2, 1)
        ptx = compiled.asm["ptx"]
        assert "atom.shared::cluster.add.u32" in ptx

        _submesh_barrier_lowering_kernel[(1, )](
            out, mesh=submesh, BLOCK=block, num_ctas=1, num_warps=4
        )
        torch.cuda.synchronize()

    def test_distributed_barrier_multiblock_counter(self):
        grid = 2
        cluster_size = 2
        num_programs = grid * cluster_size

        counter = torch.zeros((grid, ), device="cuda", dtype=torch.int32)
        out = torch.empty((num_programs, ), device="cuda", dtype=torch.int32)

        compiled = _distributed_barrier_multiblock_counter_kernel.warmup(
            counter,
            out,
            mesh=BLOCK_CLUSTER_MESH,
            grid=(grid, ),
            num_ctas=1,
            num_warps=4,
        )
        assert compiled.metadata.cluster_dims == (2, 1, 1)

        _distributed_barrier_multiblock_counter_kernel[(grid, )](
            counter,
            out,
            mesh=BLOCK_CLUSTER_MESH,
            num_ctas=1,
            num_warps=4,
        )
        torch.cuda.synchronize()

        expected_counter = torch.full_like(counter, cluster_size)
        expected_out = torch.full_like(out, cluster_size)
        torch.testing.assert_close(counter, expected_counter, atol=0, rtol=0)
        torch.testing.assert_close(out, expected_out, atol=0, rtol=0)

    def test_distributed_barrier_row_group_independence(self):
        grid = 2
        counter_row0 = torch.zeros((1, ), device="cuda", dtype=torch.int32)
        counter_row1 = torch.zeros((1, ), device="cuda", dtype=torch.int32)
        out_row0 = torch.empty((4, ), device="cuda", dtype=torch.int32)
        out_row1 = torch.empty((4, ), device="cuda", dtype=torch.int32)

        compiled = _submesh_row_group_barrier_kernel.warmup(
            counter_row0,
            counter_row1,
            out_row0,
            out_row1,
            row0_mesh=BLOCK_CLUSTER_SUBMESH_ROW0,
            row1_mesh=BLOCK_CLUSTER_SUBMESH_ROW1,
            grid=(grid, ),
            num_ctas=1,
            num_warps=4,
        )
        assert compiled.metadata.cluster_dims == (2, 2, 1)
        assert compiled.asm["ptx"].count("atom.shared::cluster.add.u32") >= 2

        _submesh_row_group_barrier_kernel[(grid, )](
            counter_row0,
            counter_row1,
            out_row0,
            out_row1,
            row0_mesh=BLOCK_CLUSTER_SUBMESH_ROW0,
            row1_mesh=BLOCK_CLUSTER_SUBMESH_ROW1,
            num_ctas=1,
            num_warps=4,
        )
        torch.cuda.synchronize()

        row0_count = int(counter_row0.cpu().item())
        row1_count = int(counter_row1.cpu().item())
        assert row0_count > 0
        assert row1_count == 3 * row0_count
        torch.testing.assert_close(
            out_row0, torch.tensor([row0_count, row0_count, -1, -1], device="cuda", dtype=torch.int32)
        )
        torch.testing.assert_close(
            out_row1, torch.tensor([-1, -1, row1_count, row1_count], device="cuda", dtype=torch.int32)
        )

    def test_distributed_barrier_col_group_independence(self):
        grid = 2
        counter_col0 = torch.zeros((1, ), device="cuda", dtype=torch.int32)
        counter_col1 = torch.zeros((1, ), device="cuda", dtype=torch.int32)
        out_col0 = torch.empty((4, ), device="cuda", dtype=torch.int32)
        out_col1 = torch.empty((4, ), device="cuda", dtype=torch.int32)

        compiled = _submesh_col_group_barrier_kernel.warmup(
            counter_col0,
            counter_col1,
            out_col0,
            out_col1,
            col0_mesh=BLOCK_CLUSTER_SUBMESH_COL0,
            col1_mesh=BLOCK_CLUSTER_SUBMESH_COL1,
            grid=(grid, ),
            num_ctas=1,
            num_warps=4,
        )
        assert compiled.metadata.cluster_dims == (2, 2, 1)
        assert compiled.asm["ptx"].count("atom.shared::cluster.add.u32") >= 2

        _submesh_col_group_barrier_kernel[(grid, )](
            counter_col0,
            counter_col1,
            out_col0,
            out_col1,
            col0_mesh=BLOCK_CLUSTER_SUBMESH_COL0,
            col1_mesh=BLOCK_CLUSTER_SUBMESH_COL1,
            num_ctas=1,
            num_warps=4,
        )
        torch.cuda.synchronize()

        col0_count = int(counter_col0.cpu().item())
        col1_count = int(counter_col1.cpu().item())
        assert col0_count > 0
        assert col1_count == 5 * col0_count
        torch.testing.assert_close(
            out_col0, torch.tensor([col0_count, -1, col0_count, -1], device="cuda", dtype=torch.int32)
        )
        torch.testing.assert_close(
            out_col1, torch.tensor([-1, col1_count, -1, col1_count], device="cuda", dtype=torch.int32)
        )
