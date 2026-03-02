"""
MoE Align Block Size (TLE Tutorial)
=================================

This tutorial keeps one shared stage2/3/4 pipeline and compares two stage1
implementations:
- triton stage1: direct global atomic accumulation
- tle stage1: shared-memory accumulation, then write back to global counts
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language.gpu as tle

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


@triton.jit(do_not_specialize=["numel"])
def moe_align_block_size_stage1_triton(
    topk_ids_ptr,
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
    numel,
    tokens_per_thread: tl.constexpr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    numel_sorted_token_ids: tl.constexpr,
    numel_expert_ids: tl.constexpr,
    block_size_sorted: tl.constexpr,
    block_size_expert: tl.constexpr,
):
    pid = tl.program_id(0)

    offsets_sorted = pid * block_size_sorted + tl.arange(0, block_size_sorted)
    mask_sorted = offsets_sorted < numel_sorted_token_ids
    tl.store(sorted_token_ids_ptr + offsets_sorted, numel, mask=mask_sorted)

    offsets_expert = pid * block_size_expert + tl.arange(0, block_size_expert)
    mask_expert = offsets_expert < numel_expert_ids
    tl.store(expert_ids_ptr + offsets_expert, 0, mask=mask_expert)

    start_idx = pid * tokens_per_thread
    off_c = (pid + 1) * num_experts

    offsets = start_idx + tl.arange(0, tokens_per_thread)
    mask = offsets < numel
    expert_id = tl.load(topk_ids_ptr + offsets, mask=mask, other=0).to(tl.int32)
    valid = mask & (expert_id < num_experts)
    expert_id = tl.where(valid, expert_id, 0)
    tl.atomic_add(tokens_cnts_ptr + off_c + expert_id, 1, mask=valid)


@triton.jit(do_not_specialize=["numel"])
def moe_align_block_size_stage1_tle(
    topk_ids_ptr,
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
    numel,
    tokens_per_thread: tl.constexpr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    numel_sorted_token_ids: tl.constexpr,
    numel_expert_ids: tl.constexpr,
    block_size_sorted: tl.constexpr,
    block_size_expert: tl.constexpr,
    BLOCK_EXPERT: tl.constexpr,
):
    pid = tl.program_id(0)

    offsets_sorted = pid * block_size_sorted + tl.arange(0, block_size_sorted)
    mask_sorted = offsets_sorted < numel_sorted_token_ids
    tl.store(sorted_token_ids_ptr + offsets_sorted, numel, mask=mask_sorted)

    offsets_expert = pid * block_size_expert + tl.arange(0, block_size_expert)
    mask_expert = offsets_expert < numel_expert_ids
    tl.store(expert_ids_ptr + offsets_expert, 0, mask=mask_expert)

    start_idx = pid * tokens_per_thread
    off_c = (pid + 1) * num_experts

    offsets = start_idx + tl.arange(0, tokens_per_thread)
    mask = offsets < numel
    expert_id = tl.load(topk_ids_ptr + offsets, mask=mask, other=0).to(tl.int32)
    valid = mask & (expert_id < num_experts)
    expert_id = tl.where(valid, expert_id, 0)

    expert_offsets = tl.arange(0, BLOCK_EXPERT)
    expert_mask = expert_offsets < num_experts

    smem_counts = tle.alloc(
        [BLOCK_EXPERT],
        dtype=tl.int32,
        layout=None,
        scope=tle.smem,
        nv_mma_shared_layout=False,
    )
    smem_ptrs = tle.local_ptr(smem_counts, (expert_offsets, ))
    tl.store(smem_ptrs, 0)
    tl.debug_barrier()

    count_ptrs = tle.local_ptr(smem_counts, (expert_id, ))
    tl.atomic_add(count_ptrs, 1, mask=valid, sem="relaxed", scope="cta")
    tl.debug_barrier()

    counts = tl.load(smem_ptrs, mask=expert_mask, other=0)
    tl.store(tokens_cnts_ptr + off_c + expert_offsets, counts, mask=expert_mask)


@triton.jit
def moe_align_block_size_stage2_vec(
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
):
    pid = tl.program_id(0)

    offset = tl.arange(0, num_experts) + 1
    token_cnt = tl.load(tokens_cnts_ptr + offset * num_experts + pid)
    cnt = tl.cumsum(token_cnt, axis=0)
    tl.store(tokens_cnts_ptr + offset * num_experts + pid, cnt)


@triton.jit
def moe_align_block_size_stage2(
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
):
    pid = tl.program_id(0)

    last_cnt = 0
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + i * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + i * num_experts + pid, last_cnt)


@triton.jit
def moe_align_block_size_stage3(
    total_tokens_post_pad_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    num_experts_next_power_of_2: tl.constexpr,
    block_size: tl.constexpr,
):
    off_cnt = num_experts * num_experts

    expert_offsets = tl.arange(0, num_experts_next_power_of_2)
    mask = expert_offsets < num_experts
    token_cnts = tl.load(tokens_cnts_ptr + off_cnt + expert_offsets, mask=mask, other=0)
    aligned_cnts = tl.cdiv(token_cnts, block_size) * block_size

    cumsum_values = tl.cumsum(aligned_cnts, axis=0)
    tl.store(cumsum_ptr + 1 + expert_offsets, cumsum_values, mask=mask)

    total_tokens = tl.sum(aligned_cnts, axis=0)
    tl.store(total_tokens_post_pad_ptr, total_tokens)


@triton.jit(do_not_specialize=["numel"])
def moe_align_block_size_stage4(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    numel,
    tokens_per_thread: tl.constexpr,
):
    pid = tl.program_id(0)
    start_idx = tl.load(cumsum_ptr + pid)
    end_idx = tl.load(cumsum_ptr + pid + 1)

    for i in range(start_idx, end_idx, block_size):
        tl.store(expert_ids_ptr + i // block_size, pid)

    start_idx = pid * tokens_per_thread
    off_t = pid * num_experts

    offset = tl.arange(0, tokens_per_thread) + start_idx
    mask = offset < numel
    expert_id = tl.load(topk_ids_ptr + offset, mask=mask)
    token_idx_in_expert = tl.atomic_add(tokens_cnts_ptr + off_t + expert_id, 1, mask=mask)
    rank_post_pad = token_idx_in_expert + tl.load(cumsum_ptr + expert_id, mask=mask)
    tl.store(sorted_token_ids_ptr + rank_post_pad, offset, mask=mask)


def _allocate_outputs(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    pad_sorted_ids: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    sorted_ids = torch.empty((max_num_tokens_padded, ), dtype=torch.int32, device=topk_ids.device)
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty((max_num_m_blocks, ), dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1, ), dtype=torch.int32, device=topk_ids.device)
    return sorted_ids, expert_ids, num_tokens_post_pad


def _make_bench_runner(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    provider: str = "triton",
):
    numel = topk_ids.numel()
    numel_sorted_token_ids = sorted_ids.numel()
    numel_expert_ids = expert_ids.numel()
    grid = (num_experts, )
    tokens_cnts = torch.empty((num_experts + 1, num_experts), dtype=torch.int32, device=topk_ids.device)
    cumsum = torch.empty((num_experts + 1, ), dtype=torch.int32, device=topk_ids.device)
    tokens_per_thread = triton.next_power_of_2(ceil_div(numel, num_experts))
    num_experts_next_power_of_2 = triton.next_power_of_2(num_experts)
    block_size_sorted = triton.next_power_of_2(ceil_div(numel_sorted_token_ids, num_experts))
    block_size_expert = triton.next_power_of_2(ceil_div(numel_expert_ids, num_experts))
    block_expert = triton.cdiv(num_experts, 32) * 32

    def init_fn():
        tokens_cnts.zero_()
        cumsum.zero_()

    def kernel_fn():
        if provider == "triton":
            moe_align_block_size_stage1_triton[grid](
                topk_ids,
                tokens_cnts,
                num_experts,
                numel,
                tokens_per_thread,
                sorted_ids,
                expert_ids,
                numel_sorted_token_ids,
                numel_expert_ids,
                block_size_sorted,
                block_size_expert,
            )
        elif provider == "tle":
            moe_align_block_size_stage1_tle[grid](
                topk_ids,
                tokens_cnts,
                num_experts,
                numel,
                tokens_per_thread,
                sorted_ids,
                expert_ids,
                numel_sorted_token_ids,
                numel_expert_ids,
                block_size_sorted,
                block_size_expert,
                BLOCK_EXPERT=block_expert,
            )
        else:
            raise ValueError(f"unknown provider: {provider}")
        if num_experts == triton.next_power_of_2(num_experts):
            moe_align_block_size_stage2_vec[grid](tokens_cnts, num_experts)
        else:
            moe_align_block_size_stage2[grid](tokens_cnts, num_experts)
        moe_align_block_size_stage3[(1, )](
            num_tokens_post_pad,
            tokens_cnts,
            cumsum,
            num_experts,
            num_experts_next_power_of_2,
            block_size,
        )
        moe_align_block_size_stage4[grid](
            topk_ids,
            sorted_ids,
            expert_ids,
            tokens_cnts,
            cumsum,
            num_experts,
            block_size,
            numel,
            tokens_per_thread,
        )

    return init_fn, kernel_fn


def moe_align_block_size_triton_impl(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    init_fn, kernel_fn = _make_bench_runner(
        topk_ids, block_size, num_experts, sorted_token_ids, expert_ids, num_tokens_post_pad, provider="triton"
    )
    init_fn()
    kernel_fn()


def moe_align_block_size_tle_impl(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    init_fn, kernel_fn = _make_bench_runner(
        topk_ids, block_size, num_experts, sorted_token_ids, expert_ids, num_tokens_post_pad, provider="tle"
    )
    init_fn()
    kernel_fn()


def moe_align_block_size_triton(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    pad_sorted_ids: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sorted_ids, expert_ids, num_tokens_post_pad = _allocate_outputs(topk_ids, num_experts, block_size, pad_sorted_ids)
    moe_align_block_size_triton_impl(topk_ids, num_experts, block_size, sorted_ids, expert_ids, num_tokens_post_pad)
    return sorted_ids, expert_ids, num_tokens_post_pad


def moe_align_block_size_tle(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    pad_sorted_ids: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sorted_ids, expert_ids, num_tokens_post_pad = _allocate_outputs(topk_ids, num_experts, block_size, pad_sorted_ids)
    moe_align_block_size_tle_impl(topk_ids, num_experts, block_size, sorted_ids, expert_ids, num_tokens_post_pad)
    return sorted_ids, expert_ids, num_tokens_post_pad


def _bench_kernel_only(init_fn, kernel_fn, warmup_ms: int = 20, rep_ms: int = 100) -> Tuple[float, float, float]:
    def run():
        init_fn()
        kernel_fn()

    ms, p20, p80 = triton.testing.do_bench(
        run,
        warmup=warmup_ms,
        rep=rep_ms,
        quantiles=[0.5, 0.2, 0.8],
    )
    return float(ms), float(p20), float(p80)


def _rand_topk_ids(num_tokens: int, num_experts: int) -> torch.Tensor:
    return torch.randint(0, num_experts, (num_tokens, ), device=DEVICE, dtype=torch.int32)


def run_correctness(
    num_tokens: int,
    num_experts: int,
    block_size: int,
):
    torch.manual_seed(0)
    topk_ids = _rand_topk_ids(num_tokens, num_experts)
    triton_sorted, triton_expert, triton_num_post = moe_align_block_size_triton(topk_ids, block_size, num_experts)
    tle_sorted, tle_expert, tle_num_post = moe_align_block_size_tle(topk_ids, block_size, num_experts)

    counts = torch.bincount(topk_ids, minlength=num_experts)
    aligned = torch.div(counts + (block_size - 1), block_size, rounding_mode="floor") * block_size
    cumsum = torch.cumsum(aligned, dim=0).to(torch.int32)
    torch.testing.assert_close(triton_num_post, cumsum[-1:])
    torch.testing.assert_close(tle_num_post, cumsum[-1:])

    valid_length = int(triton_num_post.item())
    num_blocks = valid_length // block_size
    expected_expert_ids = torch.repeat_interleave(
        torch.arange(num_experts, device=DEVICE, dtype=torch.int32),
        aligned.to(torch.int32) // block_size,
    )
    torch.testing.assert_close(triton_expert[:num_blocks], expected_expert_ids)
    torch.testing.assert_close(tle_expert[:num_blocks], expected_expert_ids)

    for sorted_ids in (triton_sorted, tle_sorted):
        start = 0
        for expert_id in range(num_experts):
            end = int(cumsum[expert_id].item())
            tokens = sorted_ids[start:end]
            valid_tokens = tokens[tokens < num_tokens]
            if counts[expert_id] > 0:
                assert valid_tokens.numel() == int(counts[expert_id].item())
                torch.testing.assert_close(
                    topk_ids[valid_tokens],
                    torch.full_like(valid_tokens, expert_id),
                )
            start = end

        if valid_length < sorted_ids.numel():
            assert torch.all(sorted_ids[valid_length:] >= num_tokens)

    print("Correctness check passed (stage1 triton vs tle).")


def _moe_realistic_shapes(num_experts: int) -> List[Tuple[int, int]]:
    return [
        (256, num_experts),
        (512, num_experts),
        (1024, num_experts),
        (2048, num_experts),
        (4096, num_experts),
        (8192, num_experts),
        (16384, num_experts),
        (32768, num_experts),
        (65536, num_experts),
        (163840, num_experts),
    ]


def _zipf_probs(num_experts: int, alpha: float) -> torch.Tensor:
    ranks = torch.arange(1, num_experts + 1, device=DEVICE, dtype=torch.float32)
    probs = 1.0 / (ranks**alpha)
    return probs / probs.sum()


def _sample_topk_ids(num_tokens: int, num_experts: int, probs: torch.Tensor) -> torch.Tensor:
    ids = torch.multinomial(probs, num_tokens, replacement=True)
    return ids.to(torch.int32)


def run_realistic_benchmark(block_size: int, num_experts: int) -> None:
    print("num_tokens,num_experts,source,triton_ms,tle_ms")
    probs = _zipf_probs(num_experts, alpha=1.2)
    for num_tokens, _ in _moe_realistic_shapes(num_experts):
        topk_ids = _sample_topk_ids(num_tokens, num_experts, probs)
        sorted_ids_t, expert_ids_t, num_tokens_post_pad_t = _allocate_outputs(topk_ids, num_experts, block_size, False)
        init_fn_t, kernel_fn_t = _make_bench_runner(
            topk_ids, block_size, num_experts, sorted_ids_t, expert_ids_t, num_tokens_post_pad_t, provider="triton"
        )
        sorted_ids_l, expert_ids_l, num_tokens_post_pad_l = _allocate_outputs(topk_ids, num_experts, block_size, False)
        init_fn_l, kernel_fn_l = _make_bench_runner(
            topk_ids, block_size, num_experts, sorted_ids_l, expert_ids_l, num_tokens_post_pad_l, provider="tle"
        )
        ms_t, _, _ = _bench_kernel_only(init_fn_t, kernel_fn_t)
        ms_l, _, _ = _bench_kernel_only(init_fn_l, kernel_fn_l)
        print(f"{num_tokens},{num_experts},zipf,{ms_t:.4f},{ms_l:.4f}")


def _load_real_topk_ids(path: str) -> torch.Tensor:
    path_obj = Path(path)
    if path_obj.is_dir():
        path_obj = path_obj / "topk_ids.pt"
    topk_ids = torch.load(path_obj, map_location=DEVICE)
    if topk_ids.device != DEVICE:
        topk_ids = topk_ids.to(DEVICE)
    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)
    return topk_ids.contiguous()


def run_real_data_benchmark(topk_ids_path: str, num_experts: int, block_size: int) -> None:
    topk_ids = _load_real_topk_ids(topk_ids_path)
    num_tokens = topk_ids.numel()
    max_id = int(topk_ids.max().item()) if num_tokens > 0 else -1
    if max_id >= num_experts:
        print(f"warning: max topk_id {max_id} >= num_experts {num_experts}")

    sorted_ids_t, expert_ids_t, num_tokens_post_pad_t = _allocate_outputs(topk_ids, num_experts, block_size, False)
    init_fn_t, kernel_fn_t = _make_bench_runner(
        topk_ids, block_size, num_experts, sorted_ids_t, expert_ids_t, num_tokens_post_pad_t, provider="triton"
    )
    sorted_ids_l, expert_ids_l, num_tokens_post_pad_l = _allocate_outputs(topk_ids, num_experts, block_size, False)
    init_fn_l, kernel_fn_l = _make_bench_runner(
        topk_ids, block_size, num_experts, sorted_ids_l, expert_ids_l, num_tokens_post_pad_l, provider="tle"
    )
    ms_t, _, _ = _bench_kernel_only(init_fn_t, kernel_fn_t)
    ms_l, _, _ = _bench_kernel_only(init_fn_l, kernel_fn_l)

    print(f"num_tokens={num_tokens}, num_experts={num_experts}, block_size={block_size}, source=real")
    print("provider,ms")
    print(f"triton,{ms_t:.4f}")
    print(f"tle,{ms_l:.4f}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--block_size", type=int, default=16, help="MoE block size")
    parser.add_argument("--num_tokens", type=int, default=8192, help="num tokens")
    parser.add_argument("--num_experts", type=int, default=64, help="num experts")
    parser.add_argument("--skip_correctness", action="store_true", help="skip correctness checks")
    parser.add_argument("--real_data", type=str, default="", help="path to topk_ids.pt")
    args = parser.parse_args(argv)

    if not args.skip_correctness:
        run_correctness(args.num_tokens, args.num_experts, args.block_size)

    if args.real_data:
        run_real_data_benchmark(args.real_data, args.num_experts, args.block_size)
    else:
        run_realistic_benchmark(args.block_size, args.num_experts)


if __name__ == "__main__":
    main()
