import numpy as np
import pytest
import torch

import triton
import triton.language as tl


def get_resolution(dtype):
    rtol_resolution_map = {
        torch.float16:
        1e-3,
        torch.bfloat16:
        7.9e-3,
        # f32->fp8 implemention in triton could ran out of registers when block_size > 256*256*64,
        # thus convert f32 to f16 first in triton, then convert f16 to dtype after triton kernel.
        # but this cause some precision problem, thus here use 1.25e-1 as resolution of fp8.
        # torch.float8_e4m3fn: 1e-3,
        torch.float8_e4m3fn:
        1.25e-1,
    }
    return rtol_resolution_map.get(dtype, None)


def get_triton_dtype(dtype):
    dtype_map = {
        torch.float16: tl.float16,
        torch.float32: tl.float32,
        torch.bfloat16: tl.bfloat16,
        torch.float8_e4m3fn: tl.float8e4nv,
    }
    return dtype_map.get(dtype, None)


@triton.jit
def matmul_kernel_tlLoad_sqmma_tlStore_transpose(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    dtype: tl.constexpr,
    save_dtype: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # handle an AttributeError
    dtype = dtype

    # pointers
    A = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    c_block_ptr = C_ptr + N * offs_m[:, None] + 1 * offs_n[None, :]

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(A)
        b = tl.load(B)
        accumulator = tl.dot(a, b, acc=accumulator)
        A += BLOCK_SIZE_K * stride_ak
        B += BLOCK_SIZE_K * stride_bk
    # f32->bf16 implemention in triton could ran out of registers when block_size > 128*128*64,
    # thus convert f32 to f16 first in triton, then convert f16 to dtype after triton kernel.
    accumulator = accumulator.to(save_dtype)
    tl.store(c_block_ptr, accumulator, mask=mask)


@pytest.mark.parametrize("num_stages", [1])
@pytest.mark.parametrize(
    "M, N, K",
    [
        (4096, 3072, 2048),
        (4096, 576, 2048),
        (4096, 4096, 512),
        (4096, 2048, 2048),
        (4096, 5632, 2048),
        (4096, 2048, 2816),
        (4096, 2816, 2048),
        (2048, 2816, 4096),
        (5632, 2048, 4096),
        (2048, 4096, 2048),
        (4096, 512, 4096),
        (4096, 2048, 576),
        (4096, 2048, 3072),
        (3072, 2048, 4096),
        (2048, 2048, 4096),
    ],
)
@pytest.mark.parametrize(
    "BLOCK_M, BLOCK_N, BLOCK_K, num_warps",
    [
        (32, 32, 32, 4),
        (128, 128, 64, 4),
        (32, 512, 32, 4),
        (64, 64, 32, 16),
        # (256, 256, 64, 16),
        (32, 64, 32, 8),
        (64, 32, 32, 8),
        (128, 256, 64, 8),
        (256, 128, 64, 8),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("is_transposeA", [True, False])
@pytest.mark.parametrize("is_transposeB", [True, False])
def test_tlLoad_sqmma_tlStore_transpose(
    num_stages,
    M,
    N,
    K,
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    num_warps,
    dtype,
    is_transposeA,
    is_transposeB,
):
    print(
        f"num_stage: {num_stages}\tM: {M}\tN: {N}\tK: {K}\tBLOCK_M: {BLOCK_M}\tBLOCK_N: {BLOCK_N}\tBLOCK_K: {BLOCK_K}\tdtype: {dtype}",
        end="\t\t",
    )

    device = "musa"
    rtol = get_resolution(dtype)
    # save_dtype = dtype if dtype != torch.bfloat16 else torch.float16
    save_dtype = torch.float16
    torch.manual_seed(42)
    if is_transposeA:
        A = torch.randn((K, M), dtype=torch.float32, device=device).to(dtype)
        A = A.transpose(1, 0)
    else:
        A = torch.randn((M, K), dtype=torch.float32, device=device).to(dtype)
    if is_transposeB:
        B = torch.randn((N, K), dtype=torch.float32, device=device).to(dtype)
        B = B.transpose(1, 0)
    else:
        B = torch.randn((K, N), dtype=torch.float32, device=device).to(dtype)
    C = torch.zeros((M, N), dtype=torch.float32, device=device).to(save_dtype)

    kernel = matmul_kernel_tlLoad_sqmma_tlStore_transpose[(triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, 1)](
        A,
        B,
        C,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        get_triton_dtype(dtype),
        get_triton_dtype(save_dtype),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    # print(kernel.asm['ttir'])
    # print(kernel.asm['ttgir'])
    # print(kernel.asm['llir'])
    # print("C.numel() :", C.numel())
    torch.set_printoptions(threshold=float("inf"), linewidth=800, edgeitems=50, sci_mode=False, precision=1)

    C = C.to(dtype)
    ref_out = torch.matmul(A, B).to(dtype)

    torch.testing.assert_close(ref_out.to(torch.float32), C.to(torch.float32), rtol=rtol, atol=1e-3)
    print("Successfully!")
