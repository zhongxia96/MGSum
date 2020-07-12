# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import torch

from fairseq import search, utils
from fairseq.models import FairseqIncrementalDecoder


class SelectGenerator(object):
    def __init__(
        self,
        retain_dropout=False,
    ):
        """Generates translations of a given source sentence.

        Args:
            retain_dropout (bool, optional): use dropout when generating
                (default: False)

        """
        self.retain_dropout = retain_dropout

    @torch.no_grad()
    def generate(
        self,
        models,
        sample,
            prefix_tokens=None,
    ):
        """Generate a batch of translations.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
        """
        model = EnsembleModel(models)
        if not self.retain_dropout:
            model.eval()

        encoder_input = {
            k: v for k, v in sample['net_input'].items()
            if k != 'prev_output_tokens'
        }

        encoder_outs = model.forward_encoder(encoder_input)
        lprobs = model.forward_decoder(
            encoder_outs,
        )  # (batch_size, seq_len, 2)

        scores = lprobs[:, :, 1]

        return scores


class EnsembleModel(torch.nn.Module):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.incremental_states = None
        # if all(isinstance(m.decoder, FairseqIncrementalDecoder) for m in models):
        #     self.incremental_states = {m: {} for m in models}

    def has_encoder(self):
        return hasattr(self.models[0], 'encoder')

    @torch.no_grad()
    def forward_encoder(self, encoder_input):
        if not self.has_encoder():
            return None
        return [model.encoder(**encoder_input) for model in self.models]

    @torch.no_grad()
    def forward_decoder(self, encoder_outs):
        if len(self.models) == 1:
            return self._decode_one(
                self.models[0],
                encoder_outs[0] if self.has_encoder() else None,
                log_probs=True,
            )
        else:
            return None

    def _decode_one(self, model, encoder_out, log_probs):
        decoder_out = model.sentence_decoder(encoder_out)
        probs = model.sentence_decoder.get_normalized_probs(decoder_out, log_probs=False, sample=None).transpose(0, 1)
        # probs = probs[:, -1, :]
        return probs

    def reorder_encoder_out(self, encoder_outs, new_order):
        if not self.has_encoder():
            return
        return [
            model.encoder.reorder_encoder_out(encoder_out, new_order)
            for model, encoder_out in zip(self.models, encoder_outs)
        ]
