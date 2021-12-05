# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from xformers import _is_triton_available
from xformers.components.attention import Attention, AttentionConfig, register_attention
from xformers.components.attention.feature_maps import (
    FeatureMap,
    FeatureMapType,
    SMHyperbolic,
    SMOrf,
    SMReg,
)

if _is_triton_available:
    from xformers.triton.causal_product import causal_product


@dataclass
class FavorAttentionConfig(AttentionConfig):
    causal: Optional[bool]
    dim_features: Optional[int] = None  # The dimensions of the random features
    dim_head: Optional[
        int
    ] = None  # The embedding dimension of the inputs. Only useful to get a dim_features estimate
    iter_before_redraw: Optional[
        int
    ] = None  # The number of iterations before the random features are re-drawn from scratch
    feature_map: Optional[FeatureMapType] = None


@torch.jit.script
def outer_prod(a, b):
    return torch.einsum("be,bf->bef", a, b)


@torch.jit.script
def line_multiply(a, b):
    return torch.einsum("bf, bfe->be", a, b)


# @torch.jit.script
def causal_attention(
    k_prime: torch.Tensor, q_prime: torch.Tensor, v: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    if _is_triton_available:
        att_raw = causal_product(q_prime, k_prime, v)
        att_norm = causal_product(q_prime, k_prime, torch.ones_like(v))
        return att_raw, att_norm

    else:
        # Cleaner take, but too memory intensive:
        # Algorithm 1 in the paper
        ref_v = torch.ones_like(v.unsqueeze(2))  # BATCH x SEQ x 1 x EMB
        Gps = k_prime.unsqueeze(3) * v.unsqueeze(2)
        Grenorm = k_prime.unsqueeze(3) * ref_v

        # Consolidate against the feature dimension
        att_raw = torch.einsum("bcfe,bcf->bce", Gps, q_prime)
        att_norm = torch.einsum("bcfe,bcf->bce", Grenorm, q_prime)

        # Cumulative sum over the sequence
        att_raw = att_raw.cumsum(2)
        att_norm = att_norm.cumsum(2)

        # # TODO(@lefaudeux): Rewrite as an optimized Triton kernel ?
        # See for instance https://github.com/calclavia/Triton-Transformer/blob/master/ttx/attention/causal_product.py
        # ref_v = torch.ones_like(v[:, 0, :])

        # Gps = outer_prod(k_prime[:, 0, :], v[:, 0, :])
        # Grenorm = outer_prod(k_prime[:, 0, :], ref_v)

        # _, M, N = k_prime.shape
        # att_raw = torch.empty(
        #     (k_prime.shape[0], M * N),
        #     device=k_prime.device,
        #     dtype=k_prime.dtype,
        # )
        # att_norm = torch.empty_like(att_raw)

        # for i in range(M):
        #     start, stop = i * N, (i + 1) * N
        #     att_raw[:, start:stop] = line_multiply(q_prime[:, i, :], Gps)
        #     att_norm[:, start:stop] = line_multiply(q_prime[:, i, :], Grenorm)

        #     Gps += outer_prod(k_prime[:, i, :], v[:, i, :])
        #     Grenorm += outer_prod(k_prime[:, i, :], ref_v)

        return att_raw, att_norm


@register_attention("favor", FavorAttentionConfig)
class FavorAttention(Attention):
    def __init__(
        self,
        causal: bool = False,
        dropout: float = 0.0,
        dim_features: Optional[int] = None,
        dim_head: Optional[int] = None,
        iter_before_redraw: Optional[int] = None,
        feature_map_type: FeatureMapType = FeatureMapType.SMReg,
        normalize_inputs: bool = False,
        *_,
        **__,
    ):
        r"""
        Kernelized attention, as proposed in Performers_.

        FAVOR stands for "Fast Attention Via positive Orthogonal Random features"

        Args:
            dropout (float): the probability of an output to be randomly dropped at training time
            dim_features (int): the dimension of the random features space
            iter_before_redraw (int): the number of iterations before a redraw of the features
            feature_map_type (FeatureMapType): the type of feature map being used,
            for instance orthogonal random features.

        .. _Performers: "Rethinking attention with performers." K. Choromanski et al. (2020).
            https://arxiv.org/pdf/2009.14794v1.pdf
        """
        super().__init__()

        self.causal = causal
        self.iter_before_redraw = iter_before_redraw
        self.normalize_inputs = normalize_inputs
        self.feature_map_type = feature_map_type
        self.attn_drop = nn.Dropout(dropout, inplace=True)

        # Setup dimension-dependent variables
        # Reasonable dimension default
        if dim_features is None:
            assert dim_head is not None, "dim_features or dim_head needs to be passed"
            self.dim_features = math.ceil(dim_head * (1 + math.log2(dim_head)))
            self.dim_features = 2 * (
                self.dim_features // 2
            )  # needs to be even for some variants
            logging.info(
                f"FAVOR: Automatically setting the random mapping dimension to {self.dim_features} from {dim_head}"
            )
        else:
            self.dim_features = dim_features

        feature_map_constructor = {
            FeatureMapType.SMHyp: SMHyperbolic,
            FeatureMapType.SMReg: SMReg,
            FeatureMapType.SMOrf: SMOrf,
        }[self.feature_map_type]

        feature_settings = {
            "dim_features": self.dim_features,
            "iter_before_redraw": self.iter_before_redraw,
            "normalize_inputs": self.normalize_inputs,
        }

        self.feature_map_query: FeatureMap = feature_map_constructor(**feature_settings)  # type: ignore
        self.feature_map_key: FeatureMap = feature_map_constructor(**feature_settings)  # type: ignore

    @staticmethod
    def _maybe_promote(x: torch.Tensor) -> torch.Tensor:
        # Only promote fp16 buffers, bfloat16 would be fine for instance
        return x.float() if x.dtype == torch.float16 else x

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *_,
        **__,
    ):

        # Project key and queries onto the feature map space
        k_prime = self.feature_map_key(k)
        q_prime = self.feature_map_query(q)

        with autocast(enabled=False):
            # The softmax kernel approximation for Favor will easily overflow
            # Force the computations here to stay in fp32 for numerical stability
            # Note that the dimensions are vastly reduced when compared to scaled_dot_product
            k_prime = self._maybe_promote(k_prime)
            q_prime = self._maybe_promote(q_prime)
            v = self._maybe_promote(v)

            if not self.causal:
                att_normalization = q_prime @ (
                    k_prime.transpose(-2, -1) @ torch.ones_like(v)
                )
                att_raw = q_prime @ (k_prime.transpose(-2, -1) @ v)
            else:
                # Actually compute attention
                att_raw, att_normalization = causal_attention(k_prime, q_prime, v)

            # Normalize
            att = att_raw / att_normalization

        if self.attn_drop is not None:
            att = self.attn_drop(att)

        return att
