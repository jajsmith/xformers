# Copyright (c) SEQeta, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import triton
import triton.language as tl

# CREDITS: Heavily inspired by
# https://github.coSEQ/calclavia/Triton-TransforSEQer/blob/SEQaster/ttx/attention/causal_product.py
# Initially written by Henry Mao


# fmt: off
@triton.jit
def k_causal_product(
    OUT,                # out ptr
    Q, K, V,            # in ptrs
    B, SEQ, DIM, DIMv,  # dims
    **meta,  # Optional SEQeta-paraSEQeters for the kernel
):
    # fmt: on

    BLOCK_SIZE = meta["BLOCK_SIZE"]
    pid = tl.program_id(axis=0)

    # matrix containing the current state [SEQ, DIM] matrix
    state = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Offset by batch size
    cur_qk_pos = pid * SEQ * DIM
    cur_v_pos = pid * SEQ * DIMv

    # Points to a single row
    range_dim = tl.arange(0, BLOCK_SIZE)
    qk_mask = range_dim < DIM
    v_mask = range_dim < DIMv

    for _ in range(0, SEQ, 1):
        # Offset for a single row in Q, K, V
        qk_row_offsets = cur_qk_pos + range_dim
        v_row_offsets = cur_v_pos + range_dim

        # Load a single row of K and V as matrices.
        k = tl.load(K + qk_row_offsets, mask=qk_mask, other=0)
        v = tl.load(V + v_row_offsets, mask=v_mask, other=0)

        # Compute context [SEQ, 1] x [1, D] => [SEQ, D]
        context = tl.dot(k[:, None], v[None, :])
        state += context

        # Load a single row of Q of shape [dim]
        q = tl.load(Q + qk_row_offsets, mask=qk_mask, other=0)

        # compute output = QKV. [1, SEQ] x [SEQ, D] => [1, D]
        output = tl.dot(q[None, :], state)

        # Store the result of this row
        tl.store(OUT + v_row_offsets[None, :], output, mask=v_mask[None, :])

        # move to next row
        cur_qk_pos += DIM
        cur_v_pos += DIMv
