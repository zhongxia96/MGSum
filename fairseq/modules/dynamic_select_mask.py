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


class DynamicSelectMask(nn.Module):
    """Multi-headed pooling.
    """

    def __init__(self, embed_dim, dropout=0.):
        super().__init__()
        self.embed_dim = embed_dim

        self.dropout = nn.Dropout(dropout)

        self.scores = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.SELU(), nn.Linear(embed_dim, 2))
        self.scores2 = nn.Linear(embed_dim, 1, bias=False)
        self.relu = nn.ReLU()
        self.threshold = torch.nn.Threshold(threshold=0, value=-2 ** 32 + 1)

        self.reset_parameters()


    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.scores.weight)
        nn.init.xavier_uniform_(self.scores2.weight)


    def forward(self, encoder_out, decoder_input, encoder_padding_mask=None, decoder_mask=None):
        """encoder_out shape: `(src_len, batch, embed_dim)`
          decoder_out shape: `(tgt_len, batch, embed_dim)`
          encoder_padding_mask: `(batch, src_len)`
          attn_mask: `(tgt_len, tgt_len)`

        :return
        encoder_gate_mask shape: `(batch, tgt_len, src_len)`
        """
        src_len, batch = encoder_out.size(0), encoder_out.size(1)
        tgt_len, embed_dim = decoder_input.size(0), decoder_input.size(-1)

        encoder_out_expand = encoder_out.transpose(0, 1).unsqueeze(1).repeat(1, tgt_len, 1, 1)

        decoder_input_expand = decoder_input.transpose(0, 1).unsqueeze(2)\
                        .repeat(1, 1, tgt_len, 1) #`(batch, tgt_len, tgt_len, embed_dim)`
        decoder_input_expand = decoder_input_expand.float().masked_fill(
                    decoder_mask.unsqueeze(0).unsqueeze(-1),
                    0,
                ).type_as(decoder_input_expand)
        decoder_input_expand = torch.mean(decoder_input_expand, dim=2) #`(batch, tgt_len, embed_dim)`

        encoder_out_expand = encoder_out_expand + decoder_input_expand.unsqueeze(2)
        encoder_out_expand = self.dropout(encoder_out_expand)

        encoder_gate_mask = self.relu(F.softmax(self.scores2(encoder_out_expand).squeeze(-1), dim=-1)-0.1) #`(batch, tgt_len, src_len)`

        # encoder_gate_mask = self.scores(encoder_out_expand)[:, :, :, 0].squeeze(-1)  # `(batch, tgt_len, src_len)`
        # encoder_gate_mask = encoder_gate_mask < 0.5

        # if encoder_padding_mask is not None:
        #     encoder_gate_mask = encoder_gate_mask.masked_fill(encoder_padding_mask.unsqueeze(1), -2 ** 32 + 1)

        # return encoder_gate_mask.unsqueeze(1).repeat(1, tgt_len, 1)


        # encoder_out_expand2 = encoder_out_expand.unsqueeze(1).repeat(1, tgt_len, 1, 1)
        # encoder_gate_mask2 = F.gumbel_softmax(self.scores(encoder_out_expand2), tau=1, hard=True,
        #                                      dim=-1)
        # encoder_gate_mask2 = encoder_gate_mask2[:, :, :, 1].squeeze(-1)  # `(batch, tgt_len, src_len)`
        # return encoder_gate_mask.view(batch, tgt_len, src_len)

        # encoder_out_expand3 = encoder_out_expand
        # encoder_gate_mask = F.gumbel_softmax(self.scores(encoder_out_expand), tau=1, hard=True,
        #                                      dim=-1)[:, :, :, 1].squeeze(-1)  # `(batch, tgt_len, src_len)`
        return encoder_gate_mask