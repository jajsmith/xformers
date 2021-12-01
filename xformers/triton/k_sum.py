# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import triton
import triton.language as tl


# fmt: off
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 4}, num_warps=1),
        triton.Config({"BLOCK_N": 4}, num_warps=4),
        triton.Config({"BLOCK_N": 8}, num_warps=2),
        triton.Config({"BLOCK_N": 8}, num_warps=4),
        triton.Config({"BLOCK_N": 16}, num_warps=4),
        triton.Config({"BLOCK_N": 32}, num_warps=8),
        triton.Config({"BLOCK_N": 64}, num_warps=16),
        triton.Config({"BLOCK_N": 64}, num_warps=16),
    ],
    key=["M", "N"],
)
@triton.heuristics(
    values={"BLOCK_SIZE": lambda *args, **_: triton.next_power_of_2(args[-2])}
)
@triton.jit
def k_sum_0(
    Y,
    X,
    stride_xm,
    M,
    N,
    **meta,  # extra parameters which can be automatically filled in given some heuristics
):
    # fmt: om

    """
    Sum a 2d tensor over the first (strided) dimension
    """

    # row indices. We'll reduce over this dimension
    m = tl.arange(0, meta["BLOCK_SIZE"])

    # To get some extra parallelization, we can try to handle several columns in the same thread block
    n = tl.program_id(0)
    rn = n * meta["BLOCK_N"] + tl.arange(0, meta["BLOCK_N"])

    # the memory address of all the elements that we want to load can be computed as follows
    x_ptrs = X + m[:, None] * stride_xm + rn[None, :]

    # load input data; pad out-of-bounds elements with 0
    # NOTE: make sure to accumulate in fp32 to prevent a trivial overflow
    mask = (m[:, None] < M) & (rn[None, :] < N)
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    x = tl.where(mask, x, 0.0)
    x_sum = tl.sum(x, 0)
    tl.store(Y + rn, x_sum, mask=rn < N)
