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
