# Copyright (c) Meta, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch

from xformers.triton.k_causal_product import k_causal_product


def causal_product(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    # We need to preallocate the output
    output = torch.empty_like(v)

    assert q.is_cuda and k.is_cuda and v.is_cuda and output.is_cuda
    assert q.size() == k.size()
    assert q.size()[:-1] == v.size()[:-1], (q.size(), v.size())
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1

    B, SEQ, DIM = output.size()
    DIMv = v.size(-1)

    # grid is one per batch dim, extra paralellism can be brought in through the number of warps
    def grid(meta):
        return (B,)

    block_size = int(2 ** math.ceil(math.log2(max(DIM, DIMv))))

    # fmt: off
    k_causal_product[grid](
        output,
        q, k, v,
        B, SEQ, DIM, DIMv, num_warps=1, BLOCK_SIZE=block_size
    )
    # fmt: on

    return output
