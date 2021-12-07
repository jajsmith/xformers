# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


# CREDITS: This comes almost as-is from the Triton dropout tutorial
# https://raw.githubusercontent.com/openai/triton/master/python/tutorials/04-low-memory-dropout.py

import triton
import triton.language as tl

_k_configs = [
    triton.Config({"BLOCK_SIZE": 128}, num_warps=1),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=2),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=16),
]


def get_depth(*args, **_):
    return triton.next_power_of_2(args[-1])


# autotune: Triton will test out these configurations, and automatically pick the fastest one.
# heuristic: add arguments to the kernel call automatically given some heuristics. These arguments are passed in "meta"
# fmt: off
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 1}, num_warps=4),
        triton.Config({"BLOCK_N": 2}, num_warps=4),
        triton.Config({"BLOCK_N": 8}, num_warps=4),
        triton.Config({"BLOCK_N": 1}, num_warps=8),
        triton.Config({"BLOCK_N": 2}, num_warps=8),
        triton.Config({"BLOCK_N": 8}, num_warps=8),
    ],
    key=["M", "N"],
)
@triton.heuristics(values={"depth": get_depth})
@triton.jit
def k_sum_0(
    Y, X,
    stride_xm,
    M, N,
    **meta,  # extra parameters which can be automatically filled in given some heuristics
):
    # fmt: om

    """
    Sum a 2d tensor over the first dimension
    """

    n = tl.program_id(0)

    # row indices
    m = tl.arange(0, meta["depth"])
    rn = n * meta["BLOCK_N"] + tl.arange(0, meta["BLOCK_N"])

    # the memory address of all the elements that we want to load can be computed as follows
    x_ptrs = X + m[:, None] * stride_xm + rn[None, :]

    # load input data; pad out-of-bounds elements with 0
    x = tl.load(x_ptrs, mask=(m[:, None] < M) & (rn[None, :] < N), other=0.)

    x_sum = tl.sum(x, 0)
    tl.store(Y + rn, x_sum, mask=rn < N)


@triton.jit
def _drop_and_scale(SEEDS, row, p, offsets, x):
    # randomly prune the weights
    seed = SEEDS + row
    random = tl.rand(seed.to(tl.int32), offsets)
    x_keep = random > p

    zero = 0.0
    zero = zero.to(x.dtype)

    # prune and normalize in one go
    return tl.where(x_keep, (x / (1 - p)).to(x.dtype), zero)


# fmt: off
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16)
    ],
    key=["N"],
)
@triton.jit
def k_dropout_fw(
    Y, X, BIAS, SEEDS,
    stride,
    N,
    p,
    **META,
):
    """
    Apply dropout on an input tensor
    Y : Output (M, N)
    X : Input (M, N)
    S : Seeds (M,)
    p : dropout probability
    """
    # fmt: on

    BLOCK_SIZE = META["BLOCK_SIZE"]
    row = tl.program_id(axis=0)
    rn = tl.arange(0, BLOCK_SIZE)

    # compute memory offsets of elements handled by this instance
    offsets = row * stride + rn
    mask = rn < N

    # load data from x
    x_ptrs = X + offsets
    x = tl.load(x_ptrs, mask=mask)

    # optionally apply a fused bias
    if META["USE_BIAS"]:
        b_ptrs = BIAS + rn
        b = tl.load(b_ptrs, mask=mask)
        x += b

    # optional: fused activation (while the data is in shared memory)
    if META["ACTIVATION"]:
        x = META["ACTIVATION"](x)

    # randomly prune it
    if p > 0.:
        output = _drop_and_scale(SEEDS, row, p, offsets, x)
    else:
        output = x

    y_ptrs = Y + offsets
    tl.store(y_ptrs, output, mask=mask)


# fmt: off
@triton.autotune(
    configs=_k_configs,
    key=["N"],
)
@triton.jit
def k_dropout_bw(
    GRAD_IN, GRAD_OUT, INPUTS, BIAS, SEEDS,
    stride_grad, stride_inputs,
    N,
    p,
    **META,
):
    """
    Apply dropout on an input tensor
    GRAD_OUT    (M, N)
    GRAD_IN     (M, N)
    BIAS        (N,)
    SEEDS       (M,)
    p : dropout probability
    """
    # fmt: on

    BLOCK_SIZE = META["BLOCK_SIZE"]
    row = tl.program_id(axis=0)
    col = tl.program_id(axis=1)

    # compute memory offsets of elements handled by this instance
    grad_offsets = row * stride_grad + col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) < N

    # load data from x
    grad_out_ptrs = GRAD_OUT + grad_offsets
    grad_out = tl.load(grad_out_ptrs, mask=mask)

    # optional: fused activation (while the data is in shared memory)
    if META["ACTIVATION_GRAD"]:
        input_ptrs = INPUTS + row * stride_inputs + col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        inputs = tl.load(input_ptrs, mask=mask)

        # optionally apply a fused bias
        if META["USE_BIAS"]:
            b_ptrs = BIAS + col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            b = tl.load(b_ptrs, mask=mask)
            inputs += b

        act_grad = META["ACTIVATION_GRAD"](inputs)
        grad_out *= act_grad

    # randomly prune it
    if p > 0.:
        output = _drop_and_scale(SEEDS, row, p, grad_offsets, grad_out)
    else:
        output = grad_out

    # write-back
    y_ptrs = GRAD_IN + grad_offsets
    tl.store(y_ptrs, output, mask=mask)
