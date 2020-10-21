# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from fairseq import utils


class MultiheadPooling(nn.Module):
    """Multi-headed pooling.
    """

    def __init__(self, embed_dim, num_heads, dropout=0., use_final_linear=True):
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.linear_keys = nn.Linear(embed_dim, num_heads)
        self.linear_values = nn.Linear(embed_dim, embed_dim)

        self.softmax = nn.Softmax(dim=0)
        if use_final_linear:
            self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.use_final_linear = use_final_linear
        self.reset_parameters()

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.xavier_uniform_(self.linear_keys.weight)
        nn.init.xavier_uniform_(self.linear_values.weight)

    def forward(self, key, value, mask=None):
        """Input shape: `(n_tokens, batch * n_blocks, embed_dim)`

        pooling can be implemented by passing in the same arguments for
        key and value. Timesteps can be masked by supplying a `(batch * n_blocks, n_tokens)` mask in the
        `mask` argument.
        """

        batch_size = key.size(1)
        head_dim = self.head_dim
        num_heads = self.num_heads
        scores = self.linear_keys(key)
        value = self.linear_values(value)

        scores = scores.view(-1, batch_size, num_heads)   # `(n_tokens, batch * n_blocks, num_heads)`
        value = value.view(-1, batch_size, num_heads, head_dim)

        if mask is not None:
            mask = mask.transpose(0, 1).unsqueeze(-1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e18)

        # apply attention dropout and compute context vectors.
        attn = self.dropout(self.softmax(scores))   # `(n_tokens, batch * n_blocks, num_heads)`
        context = torch.sum(attn.unsqueeze(-1) * value, 0)   # `(batch * n_blocks, num_heads, head_dim)`

        if self.use_final_linear:
            context = context.view(batch_size, self.embed_dim)   # `(batch * n_blocks, num_heads * head_dim)`
            context = self.out_proj(context)

        return context