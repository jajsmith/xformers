# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


# CREDITS: This comes almost as-is from the Triton dropout tutorial
# https://raw.githubusercontent.com/openai/triton/master/python/tutorials/04-low-memory-dropout.py

from typing import Optional

import torch
import triton
from torch.cuda.amp import custom_bwd, custom_fwd

from xformers.components.activations import Activation
from xformers.triton.k_activations import (
    get_triton_activation_bwd_kernel,
    get_triton_activation_kernel,
)
from xformers.triton.k_dropout import k_dropout_bw, k_dropout_fw


# Helper to handle the SPMD launch grid and error cases
class _dropout(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, x, p, bias, activation, activation_grad, trainable_bias):
        # Soft-flatten an hypothetical 3rd dimension
        x_ = x.reshape(-1, x.shape[-1]).contiguous()
        y = torch.empty_like(x_)
        M, N = x_.shape

        assert bias is None or (bias.dtype == x.dtype and bias.shape[0] == N)

        # Generate one seed per sample
        # seed max is int32 max for positive numbers: 2**16
        seeds = torch.randint(65536, (N,), device=x.device).to(torch.int32)

        def grid(meta):
            return (triton.cdiv(N, meta["BLOCK_N"]),)

        # fmt: off
        k_dropout_fw[grid](
            y, x_,
            bias if bias is not None else x_,
            seeds,
            y.stride(0),
            M, N,
            p,
            USE_BIAS=bias is not None,
            ACTIVATION=activation
        )
        # fmt: on

        if activation is not None:
            ctx.save_for_backward(seeds, bias, x)
        else:
            ctx.save_for_backward(seeds, bias, None)

        ctx.trainable_bias = bias is not None and trainable_bias
        ctx.activation_grad = activation_grad
        ctx.p = p

        return y.reshape_as(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        (seeds, bias, inputs) = ctx.saved_tensors

        # Soft-flatten an hypothetical 3rd dimension
        grad_out_ = grad_out.reshape(-1, grad_out.shape[-1]).contiguous()
        grad_in = torch.empty_like(grad_out_)

        M, N = grad_out_.shape

        # Optional inputs to compute the activation contribution to the gradient
        assert inputs is not None or ctx.activation_grad is None

        if inputs is None:
            inputs = grad_out_
        elif inputs.ndim > 2:
            inputs = inputs.reshape(-1, N)

        if ctx.trainable_bias:
            grad_bias = torch.empty((N,), device=grad_in.device, dtype=grad_in.dtype)
        else:
            grad_bias = grad_in  # will not be used

        def grid(meta):
            return (triton.cdiv(N, meta["BLOCK_N"]),)

        # fmt: off
        k_dropout_bw[grid](
            grad_in, grad_bias, grad_out_,
            inputs, bias if bias is not None else inputs,
            seeds,
            grad_out_.stride(0), inputs.stride(0),
            M, N,
            ctx.p,
            USE_BIAS=bias is not None,
            ACTIVATION_GRAD=ctx.activation_grad,
            TRAINABLE_BIAS=ctx.trainable_bias
        )
        # fmt: on

        return (
            grad_in.reshape_as(grad_out),
            None,
            grad_bias if ctx.trainable_bias else None,
            None,
            None,
            None,
        )


def dropout(
    x: torch.Tensor,
    p: float,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[Activation] = None,
):
    """
    Apply dropout on the input tensor.
    Optionally add a bias, the computation will be fused.
    """

    # Micro optim, skip dropout
    if p == 0.0 and activation is None:
        return x + bias if bias is not None else x

    act_kernel = get_triton_activation_kernel(activation)
    act_grad_kernel = get_triton_activation_bwd_kernel(activation)
    return _dropout.apply(
        x,
        p,
        bias,
        act_kernel,
        act_grad_kernel,
        bias is not None and bias.requires_grad,
    )


class FusedDropoutBias(torch.nn.Module):
    def __init__(
        self,
        p: float,
        bias_shape: Optional[int],
        activation: Optional[Activation] = None,
    ) -> None:
        super().__init__()
        self.p = p
        self.activation = activation
        self.bias = (
            torch.zeros(bias_shape, requires_grad=True)
            if bias_shape is not None
            else None
        )
        self.activation = get_triton_activation_kernel(activation)
        self.activation_grad = get_triton_activation_bwd_kernel(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convenience, catch a possible type or device mismatch
        if self.bias is not None:  # type: ignore
            self.bias = self.bias.to(dtype=x.dtype, device=x.device)  # type: ignore

        p = self.p if self.training else 0.0
        return _dropout.apply(
            x, p, self.bias, self.activation, self.activation_grad, True
        )
