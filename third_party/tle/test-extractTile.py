import torch
import triton
import triton.language as tl
import triton.experimental.tle as tle

# This test is for extract_tile

@triton.jit
def extract_tile_kernel(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :])

    # tile_shape[1] 必须是 128 的倍数
    # 修改offsets和tile_shape来检测正确性
    tile = tle.extract_tile(x, offsets=[0, 0], tile_shape=[64, 128])

    out_offs_m = tl.arange(0, 64)
    out_offs_n = tl.arange(0, 128)  # ← 也要改
    tl.store(out_ptr + out_offs_m[:, None] * 128 + out_offs_n[None, :], tile)

M, N = 128, 128
x = torch.randn(M, N, device='cuda', dtype=torch.float32)
out = torch.zeros(64, 128, device='cuda', dtype=torch.float32)  # ← 改成 [64, 128]

print("Running kernel...")
extract_tile_kernel[(1,)](x, out, M, N)

print("Kernel executed!")
# 结果测试，修改完tle.extract参数后需要修改这里
expected = x[0:64, 0:128]
print(f"Match: {torch.allclose(out, expected)}")

if torch.allclose(out, expected):
    print(" Test PASSED!")
else:
    print(" Test FAILED!")
    print(f"Max diff: {(out - expected).abs().max().item()}")

