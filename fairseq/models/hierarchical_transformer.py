# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqModel,
    register_model,
    register_model_architecture,
)

from .transformer_with_copy import transformer_with_copyDecoder
from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    MultiheadAttention,
    MultiheadGraphAttention,
    TopkMultiheadAttention,
    MultiheadAttentionWithDocmask,
    TopkMultiheadAttentionWithDocmask,
    MultiheadOnlyAttention,
    MultiheadPooling,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TopkMultiheadAttention,
)


@register_model('hierarchical_transformer')
class HierarchicalTransformerModel(FairseqModel):
    """
    hierarchical_transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (hierarchical_transformerEncoder): the encoder
        decoder (hierarchical_transformerDecoder): the decoder

    The hierarchical_transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.hierarchical_transformer_parser
        :prog:
    """

    def __init__(self, encoder, decoder, sentence_decoder, doc_decoder):
        super().__init__(encoder, decoder)
        self.sentence_decoder = sentence_decoder
        self.doc_decoder = doc_decoder

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn', choices=['relu', 'gelu', 'gelu_fast'],
                            help='Which activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--pooling-attention-heads', type=int, metavar='N',
                            help='num pooling attention heads')
        parser.add_argument('--local-encoder-layers', type=int, metavar='N',
                            help='num local encoder layers')
        parser.add_argument('--global-encoder-layers', type=int, metavar='N',
                            help='num global encoder layers')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        # fmt: on

    def forward(self, src_tokens, src_lengths, block_mask, doc_lengths, doc_block_mask, prev_output_tokens):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., input feeding/teacher
        forcing) to the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, n_blocks, n_tokens)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            block_mask (torch.LongTensor): block mask of the source sentences of shape
                `(batch, n_blocks, n_blocks)`
            doc_lengths (torch.LongTensor): doc mask of the source sentences of shape
                `(batch)`
            doc_block_mask (torch.LongTensor): doc mask of the source sentences of shape
                `(batch, n_docs, n_blocks)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing

        Returns:
            the decoder's output, typically of shape `(batch, tgt_len, vocab)`
        """
        encoder_out = self.encoder(src_tokens, src_lengths, block_mask, doc_lengths, doc_block_mask)
        decoder_out = self.decoder(prev_output_tokens, encoder_out)
        sentence_decoder_out = self.sentence_decoder(encoder_out)
        doc_decoder_out = self.doc_decoder(encoder_out)
        return decoder_out, sentence_decoder_out, doc_decoder_out

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = HierarchicalTransformerEncoder(args, src_dict, encoder_embed_tokens)
        decoder = HierarchicalTransformer_with_copyDecoder(args, tgt_dict, decoder_embed_tokens)
        sentence_decoder = HierarchicalTransformer_with_sentenceDecoder(args)
        doc_decoder = HierarchicalTransformer_with_docDecoder(args)
        return HierarchicalTransformerModel(encoder, decoder, sentence_decoder, doc_decoder)


class HierarchicalTransformerEncoder(FairseqEncoder):
    """
    hierarchical_transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`hierarchical_transformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.dropout = args.dropout

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim/2, self.padding_idx,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerLayer(args)
            for _ in range(args.encoder_layers)
        ])

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)
            self.sentence_norm = LayerNorm(embed_dim)
            self.doc_norm = LayerNorm(embed_dim)

    def forward(self, src_tokens, src_lengths, block_mask, doc_lengths, doc_block_mask):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, n_blocks, n_tokens)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            block_mask (torch.LongTensor): block mask of the source sentences of shape
                `(batch, n_blocks, n_blocks)`
            doc_lengths (torch.LongTensor): doc mask of the source sentences of shape
                `(batch)`
            doc_block_mask (torch.LongTensor): doc mask of the source sentences of shape
                `(batch, n_docs, n_blocks)`

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        batch_size, n_blocks, n_tokens = src_tokens.size()
        doc_padding_mask = torch.arange(0, doc_lengths.max())
        doc_padding_mask = doc_padding_mask.repeat(doc_lengths.numel(), 1)
        doc_padding_mask = 1 - doc_padding_mask.lt(doc_lengths.unsqueeze(1).cpu())
        doc_padding_mask = doc_padding_mask.byte().cuda()
        n_docs = doc_padding_mask.size(1)
        x = self.embed_scale * self.embed_tokens(src_tokens)

        # if self.embed_positions is not None:
        local_pos_emb = self.embed_positions(src_tokens.view(batch_size*n_blocks, n_tokens))
        local_pos_emb = local_pos_emb.view(batch_size, n_blocks, n_tokens, -1)
        block_pos_emb = self.embed_positions(torch.sum(src_tokens, 2)).unsqueeze(2).repeat(1, 1, n_tokens, 1)
        combined_pos_emb = torch.cat([local_pos_emb, block_pos_emb], -1)
        x += combined_pos_emb

        x = F.dropout(x, p=self.dropout, training=self.training)

        # compute padding mask
        local_padding_mask = src_tokens.eq(self.padding_idx).view(batch_size * n_blocks, n_tokens)
        block_padding_mask = torch.sum(1-local_padding_mask.view(batch_size, n_blocks, n_tokens), -1) == 0

        x = x.view(batch_size * n_blocks, n_tokens, -1)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        block_vec = torch.zeros(n_blocks, batch_size, self.embed_tokens.embedding_dim).cuda()
        doc_vec = torch.zeros(n_docs, batch_size, self.embed_tokens.embedding_dim).cuda()

        # encoder local layers
        for layer in self.layers:
            x, block_vec, doc_vec = layer(x, block_vec, doc_vec, local_padding_mask, block_padding_mask, doc_padding_mask, block_mask, doc_block_mask, batch_size, n_blocks)

        if self.normalize:
            x = self.layer_norm(x)
            block_vec = self.sentence_norm(block_vec)
            doc_vec = self.doc_norm(doc_vec)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        mask_hier = 1 - local_padding_mask[:, :, None].float()
        src_features = x * mask_hier
        src_features = src_features.view(batch_size, n_blocks * n_tokens, -1)
        src_features = src_features.transpose(0, 1).contiguous()  # src_len, batch_size, hidden_dim
        mask_hier = mask_hier.view(batch_size, n_blocks * n_tokens, -1)
        mask_hier = mask_hier.transpose(0, 1).contiguous()

        unpadded = [torch.masked_select(src_features[:, i], mask_hier[:, i].byte()).view([-1, src_features.size(-1)])
                    for i in range(src_features.size(1))]

        max_l = max([p.size(0) for p in unpadded])

        def sequence_mask(lengths, max_len=None):
            """
            Creates a boolean mask from sequence lengths.
            """
            batch_size = lengths.numel()
            max_len = max_len or lengths.max()
            return (torch.arange(0, max_len)
                    .type_as(lengths)
                    .repeat(batch_size, 1)
                    .lt(lengths.unsqueeze(1)))

        mask_hier = sequence_mask(torch.tensor([p.size(0) for p in unpadded]), max_l).cuda()
        mask_hier = 1 - mask_hier[:, None, :]

        unpadded = torch.stack(
            [torch.cat([p, torch.zeros(max_l - p.size(0), src_features.size(-1)).cuda()]) for p in unpadded], 1
        )

        x = unpadded
        # x = unpadded.transpose(0, 1)
        encoder_padding_mask = mask_hier.squeeze(1)

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
            'sentence_out': block_vec,  # T x B x C
            'sentence_padding_mask': block_padding_mask,  # B x T
            'doc_out': doc_vec,  # T x B x C
            'doc_padding_mask': doc_padding_mask,  # B x T
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        if encoder_out['sentence_out'] is not None:
            encoder_out['sentence_out'] = \
                encoder_out['sentence_out'].index_select(1, new_order)
        if encoder_out['sentence_padding_mask'] is not None:
            encoder_out['sentence_padding_mask'] = \
                encoder_out['sentence_padding_mask'].index_select(0, new_order)
        if encoder_out['doc_out'] is not None:
            encoder_out['doc_out'] = \
                encoder_out['doc_out'].index_select(1, new_order)
        if encoder_out['doc_padding_mask'] is not None:
            encoder_out['doc_padding_mask'] = \
                encoder_out['doc_padding_mask'].index_select(0, new_order)
        return encoder_out

    def reorder_encoder_input(self, encoder_input, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        # print('reorder')
        if encoder_input['src_tokens'] is not None:
            encoder_input['src_tokens'] = \
                encoder_input['src_tokens'].index_select(0, new_order)
        if encoder_input['src_lengths'] is not None:
            encoder_input['src_lengths'] = \
                encoder_input['src_lengths'].index_select(0, new_order)
        if encoder_input['block_mask'] is not None:
            encoder_input['block_mask'] = \
                encoder_input['block_mask'].index_select(0, new_order)
        if encoder_input['doc_block_mask'] is not None:
            encoder_input['doc_block_mask'] = \
                encoder_input['doc_block_mask'].index_select(0, new_order)
        if encoder_input['doc_lengths'] is not None:
            encoder_input['doc_lengths'] = \
                encoder_input['doc_lengths'].index_select(0, new_order)
        return encoder_input

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)
        for i in range(len(self.layers)):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(state_dict, f"{name}.layers.{i}")

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class TransformerLayer(nn.Module):
    """Global layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_block_attn = TopkMultiheadAttentionWithDocmask(
            self.embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.self_word_attn = TopkMultiheadAttention(
            self.embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.self_inter_attn = TopkMultiheadAttention(
            self.embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.self_doc_attn = TopkMultiheadAttention(
            self.embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.doc_block_attn = MultiheadGraphAttention(
            self.embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.block_doc_attn = MultiheadGraphAttention(
            self.embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout,
        )

        self.word_fusion = nn.Sequential(nn.Linear(2 * self.embed_dim, 1), nn.Sigmoid())
        self.sentence_fusion = nn.Sequential(nn.Linear(2 * self.embed_dim, 1), nn.Sigmoid())
        self.sentence_fusion2 = nn.Sequential(nn.Linear(2 * self.embed_dim, 1), nn.Sigmoid())
        self.doc_fusion = nn.Sequential(nn.Linear(2 * self.embed_dim, 1), nn.Sigmoid())

        self.self_attn_word_layer_norm = LayerNorm(self.embed_dim)
        self.self_attn_block_layer_norm = LayerNorm(self.embed_dim)
        self.self_attn_doc_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu')
        )
        self.activation_dropout = getattr(args, 'activation_dropout', 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, 'relu_dropout', 0)
        self.normalize_before = args.encoder_normalize_before
        self.word_fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.word_fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.final_word_layer_norm = LayerNorm(self.embed_dim)

        self.block_fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.block_fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.final_block_layer_norm = LayerNorm(self.embed_dim)

        self.doc_fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.doc_fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.final_doc_layer_norm = LayerNorm(self.embed_dim)

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {
            '0': 'self_attn_word_layer_norm',
            '1': 'self_attn_block_layer_norm',
            '2': 'self_attn_doc_layer_norm',
            '3': 'final_word_layer_norm',
            '4': 'final_block_layer_norm',
            '5': 'final_doc_layer_norm'
        }
        for old, new in layer_norm_map.items():
            for m in ('weight', 'bias'):
                k = f'{name}.layer_norms.{old}.{m}'
                if k in state_dict:
                    state_dict[
                        f'{name}.{new}.{m}'
                    ] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, block_vec, doc_vec, local_padding_mask, block_padding_mask, doc_padding_mask, block_mask, doc_block_mask, batch_size, n_blocks):
        """
        Args:
            x (Tensor): input to the layer of shape `(n_tokens, batch * n_blocks, embed_dim)`
            block_vec (Tensor): input to the layer of shape `(n_blocks, batch, embed_dim)`
            doc_vec (Tensor): input to the layer of shape `(n_docs, batch, embed_dim)`
            local_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch* n_blocks, n_tokens)` where padding elements are indicated by ``1``.
            block_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, n_blocks)` where padding elements are indicated by ``1``.
            block_mask (ByteTensor): binary ByteTensor of shape
                `(batch, n_blocks, n_blocks)` where padding elements are indicated by ``1``.
            doc_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, n_docs)` where one document sentence elements are indicated by ``1``
            doc_block_mask (ByteTensor): binary ByteTensor of shape
                `(batch, n_docs, n_blocks)` where one document sentence elements are indicated by ``1``
        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """

        residual_x = x  # (n_tokens, batch * n_blocks, embed_dim)
        residual_block = block_vec  # (n_blocks, batch, embed_dim)
        residual_doc = doc_vec  # (n_docs, batch, embed_dim)

        x = self.maybe_layer_norm(self.self_attn_word_layer_norm, x, before=True)
        block_vec = self.maybe_layer_norm(self.self_attn_block_layer_norm, block_vec, before=True)
        doc_vec = self.maybe_layer_norm(self.self_attn_doc_layer_norm, doc_vec, before=True)


        # word level
        x, _ = self.self_word_attn(query=x, key=x, value=x, key_padding_mask=local_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # block_word level
        inter_block = block_vec.transpose(0, 1).contiguous().view(1, -1, self.embed_dim)  # (1, batch * n_blocks, embed_dim)
        inter_x = torch.cat([inter_block, x], 0)  # (n_tokens + 1, batch * n_blocks, embed_dim)
        inter_padding_mask = torch.cat([torch.zeros(local_padding_mask.size(0), 1).cuda().byte(), local_padding_mask], -1)
        attn_mask = torch.zeros(inter_x.size(0), inter_x.size(0)).fill_(-2 ** 32 + 1).cuda()
        attn_mask[0, :] = 0
        attn_mask[:, 0] = 0
        attn_mask[0, 0] = -2 ** 32 + 1
        inter_x, _ = self.self_inter_attn(query=inter_x, key=inter_x, value=inter_x, key_padding_mask=inter_padding_mask, attn_mask=attn_mask)
        inter_x = F.dropout(inter_x, p=self.dropout, training=self.training)  # (n_tokens + 1, batch * n_blocks, embed_dim)

        word_fusion = self.word_fusion(torch.cat([x, inter_x[1:, :, :]], 2))
        x = word_fusion * x + (1 - word_fusion) * inter_x[1:, :, :]

        sentence_fusion = self.sentence_fusion(
            torch.cat([block_vec, inter_x[0, :, :].contiguous().view(batch_size, n_blocks, -1).transpose(0, 1)], -1))
        block_vec = sentence_fusion * block_vec + (1 - sentence_fusion) * inter_x[0, :, :].contiguous().view(batch_size, n_blocks, -1)\
            .transpose(0, 1)

        # block level
        block_vec, _ = self.self_block_attn(query=block_vec, key=block_vec, value=block_vec,
                                            key_padding_mask=block_padding_mask, doc_mask=block_mask)
        block_vec = F.dropout(block_vec, p=self.dropout, training=self.training)   # (n_blocks, batch, embed_dim)

        # block2doc level
        inter_doc_vec, _ = self.doc_block_attn(query=doc_vec, key=block_vec, value=block_vec, graph_mask=doc_block_mask)
        inter_doc_vec = F.dropout(doc_vec, p=self.dropout, training=self.training)
        doc_fusion = self.word_fusion(torch.cat([doc_vec, inter_doc_vec], 2))
        doc_vec = doc_fusion * doc_vec + (1 - doc_fusion) * inter_doc_vec

        # doc level
        doc_vec, _ = self.self_doc_attn(query=doc_vec, key=doc_vec, value=doc_vec, key_padding_mask=doc_padding_mask)
        doc_vec = F.dropout(doc_vec, p=self.dropout, training=self.training)  # (n_blocks, batch, embed_dim)

        # doc2block level
        inter_block_vec, _ = self.block_doc_attn(query=block_vec, key=doc_vec, value=doc_vec, graph_mask=doc_block_mask.transpose(1, 2))
        inter_block_vec = F.dropout(inter_block_vec, p=self.dropout, training=self.training)
        sentence_fusion2 = self.sentence_fusion2(torch.cat([block_vec, inter_block_vec], 2))
        block_vec = sentence_fusion2 * block_vec + (1 - sentence_fusion2) * inter_block_vec

        x = residual_x + x
        x = self.maybe_layer_norm(self.self_attn_word_layer_norm, x, after=True)

        block_vec = residual_block + block_vec
        block_vec = self.maybe_layer_norm(self.self_attn_block_layer_norm, block_vec, after=True)

        doc_vec = residual_doc + doc_vec
        doc_vec = self.maybe_layer_norm(self.self_attn_doc_layer_norm, doc_vec, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_word_layer_norm, x, before=True)
        x = self.activation_fn(self.word_fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.word_fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_word_layer_norm, x, after=True)

        residual = block_vec
        block_vec = self.maybe_layer_norm(self.final_block_layer_norm, block_vec, before=True)
        block_vec = self.activation_fn(self.block_fc1(block_vec))
        block_vec = F.dropout(block_vec, p=self.activation_dropout, training=self.training)
        block_vec = self.block_fc2(block_vec)
        block_vec = F.dropout(block_vec, p=self.dropout, training=self.training)
        block_vec = residual + block_vec
        block_vec = self.maybe_layer_norm(self.final_block_layer_norm, block_vec, after=True)

        residual = doc_vec
        doc_vec = self.maybe_layer_norm(self.final_doc_layer_norm, doc_vec, before=True)
        doc_vec = self.activation_fn(self.doc_fc1(doc_vec))
        doc_vec = F.dropout(doc_vec, p=self.activation_dropout, training=self.training)
        doc_vec = self.doc_fc2(doc_vec)
        doc_vec = F.dropout(doc_vec, p=self.dropout, training=self.training)
        doc_vec = residual + doc_vec
        doc_vec = self.maybe_layer_norm(self.final_doc_layer_norm, doc_vec, after=True)

        return x, block_vec, doc_vec

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x


class HierarchicalTransformer_with_copyDecoder(transformer_with_copyDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, final_norm=True):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=False, final_norm=True)
        self.layers = nn.ModuleList([])
        self.layers.extend([
            transformer_with_copy_topkDecoderLayer(args, no_encoder_attn)
            for _ in range(args.decoder_layers)
        ])

    def get_normalized_probs(self, net_output, log_probs, sample):
        """Get normalized probabilities (or log probs) from a net's output."""
        # print('enter normalized.')
        if 'net_input' in sample.keys():
            enc_seq_ids = sample['net_input']['src_tokens']
        else:
            enc_seq_ids = sample['src_tokens']

        batch_size, n_blocks, n_tokens = enc_seq_ids.size()
        local_padding_mask = enc_seq_ids.eq(self.embed_tokens.padding_idx).view(batch_size * n_blocks, n_tokens)
        mask_hier = 1 - local_padding_mask[:, :, None]

        enc_seq_ids = enc_seq_ids.view(batch_size, n_blocks * n_tokens)
        enc_seq_ids = enc_seq_ids.transpose(0, 1).contiguous()  # src_len, batch_size, hidden_dim
        mask_hier = mask_hier.view(batch_size, n_blocks * n_tokens)
        mask_hier = mask_hier.transpose(0, 1).contiguous()

        unpadded = [torch.masked_select(enc_seq_ids[:, i], mask_hier[:, i].byte()).view([-1])
                    for i in range(enc_seq_ids.size(1))]

        max_l = max([p.size(0) for p in unpadded])

        unpadded = torch.stack(
            [torch.cat([p, torch.zeros(max_l - p.size(0)).long().cuda()]) for p in unpadded], 1
        )

        enc_seq_ids = unpadded.transpose(0, 1)  # batch_size, n_tokens

        if hasattr(self, 'adaptive_softmax') and self.adaptive_softmax is not None:
            if sample is not None:
                assert 'target' in sample
                target = sample['target']
            else:
                target = None
            out = self.adaptive_softmax.get_log_prob(net_output[0], target=target)
            return out.exp_() if not log_probs else out

        logits = net_output[0]
        if log_probs:
            generate = utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace) * net_output[1]['copy_or_generate']
            copy = net_output[1]['attn'] * (1 - net_output[1]['copy_or_generate'])
            enc_seq_ids = enc_seq_ids.unsqueeze(1).repeat(1, net_output[1]['attn'].size(1), 1)
            final = generate.scatter_add(2, enc_seq_ids, copy)
            final = torch.log(final+1e-15)
            return final
        else:
            generate = utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)* net_output[1]['copy_or_generate']
            copy = net_output[1]['attn'] * (1 - net_output[1]['copy_or_generate'])
            enc_seq_ids = enc_seq_ids.unsqueeze(1).repeat(1, net_output[1]['attn'].size(1), 1)
            final = generate.scatter_add(2, enc_seq_ids, copy)
            return final


class transformer_with_copy_topkDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.self_attn = TopkMultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu')
        )
        self.activation_dropout = getattr(args, 'activation_dropout', 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, 'relu_dropout', 0)
        self.normalize_before = args.decoder_normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = TopkMultiheadAttention(
                self.embed_dim, args.decoder_attention_heads,
                dropout=args.attention_dropout,
            )
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)

        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.need_attn = True

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
        self,
        x,
        encoder_out=None,
        encoder_padding_mask=None,
        incremental_state=None,
        prev_self_attn_state=None,
        prev_attn_state=None,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            self.self_attn._set_input_buffer(incremental_state, saved_state)

        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        if self.encoder_attn is not None:
            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state
                saved_state = {"prev_key": prev_key, "prev_value": prev_value}
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            self_attn_state = saved_state["prev_key"], saved_state["prev_value"]
            return x, attn, self_attn_state
        return x, attn

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


class HierarchicalTransformer_with_docDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.dropout = args.dropout
        self.pooling = MultiheadPooling(
            args.encoder_embed_dim,
            args.pooling_attention_heads,
            dropout=args.dropout
        )
        self.doc_attention = MultiheadOnlyAttention(
            args.encoder_embed_dim, 1,
            dropout=0,
        )

    def forward(self, encoder_out):
        '''
        :param encoder_out:  (doc_len, batch_size, embedding_dim)
        :return:
        '''
        batch_size = encoder_out['doc_out'].size(1)
        doc_vec = self.pooling(encoder_out['doc_out'], encoder_out['doc_out'], encoder_out['doc_padding_mask'])
        doc_vec = doc_vec.view(batch_size, 1, self.embed_dim).transpose(0, 1)
        doc_vec = F.dropout(doc_vec, p=self.dropout, training=self.training)
        _, attn = self.doc_attention(query=doc_vec,
                                      key=encoder_out['doc_out'] if encoder_out is not None else None,
                                      value=encoder_out['doc_out'] if encoder_out is not None else None,
                                      key_padding_mask=encoder_out[
                                          'doc_padding_mask'] if encoder_out is not None else None,
                                      static_kv=True,
                                      need_weights=True,
                                      )  # attn: (tgt_len, bsz, doc_len)
        return attn.squeeze(0)  # (bsz, doc_len)

    def get_normalized_probs(self, net_output, log_probs, sample):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output
        return logits


class HierarchicalTransformer_with_sentenceDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu')
        )
        self.dropout = args.dropout
        self.activation_dropout = getattr(args, 'activation_dropout', 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, 'relu_dropout', 0)
        self.fc1 = Linear(args.encoder_embed_dim, args.encoder_embed_dim)
        self.fc2 = Linear(args.encoder_embed_dim, 2)
        self.final_layer_norm = LayerNorm(args.encoder_embed_dim)

    def forward(self, encoder_out):
        x = self.activation_fn(self.fc1(encoder_out['sentence_out']))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def get_normalized_probs(self, net_output, log_probs, sample):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output
        if log_probs:
            return utils.log_softmax(logits, dim=-1)
        else:
            return utils.softmax(logits, dim=-1)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


@register_model_architecture('hierarchical_transformer', 'hierarchical_transformer')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)


@register_model_architecture('hierarchical_transformer', 'hierarchical_transformer_test')
def hierarchical_transformer_iwslt_de_en_test(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 64)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 64)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.pooling_attention_heads = getattr(args, 'pooling_attention_heads', 2)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 64)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 64)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    base_architecture(args)


@register_model_architecture('hierarchical_transformer', 'hierarchical_transformer_small')
def hierarchical_transformer_iwslt_de_en_test(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.pooling_attention_heads = getattr(args, 'pooling_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 4)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 512)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 4)
    base_architecture(args)


@register_model_architecture('hierarchical_transformer', 'hierarchical_transformer_medium')
def hierarchical_transformer_iwslt_de_en_test(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.pooling_attention_heads = getattr(args, 'pooling_attention_heads', 4)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 2048)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.1)
    args.dropout = getattr(args, 'dropout', 0.1)
    base_architecture(args)
