# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import torch
import triton

from xformers.triton.k_sum import k_sum_0


def sum_2d_dim_0(x: torch.Tensor):
    """
    Sum a 2D tensor across the first dimension
    """

    out = torch.empty(x.shape[1], device=x.device, dtype=x.dtype)

    assert (
        x.ndim == 2
    ), "This is a very specific kernel, only for 2-dim tensors and summing along dim 0"
    M, N = x.shape

    assert (
        M >= 4
    ), "This is a very specific kernel, requires the reduction dimension to be bigger than 4"

    assert x.stride(1) == 1, (
        "We're expecting x to be contiguous along dim 1, and non contiguous along dim 0.\n"
        " You would probably be better served with torch.sum()"
    )

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_N"]),)

    # fmt: off
    k_sum_0[grid](
        out, x,
        x.stride(0),
        M, N,
        x.dtype == torch.float16,
    )
    # fmt: on

    return out
