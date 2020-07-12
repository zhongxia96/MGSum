# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import contextlib
import os
import numpy as np
from collections import Iterable


def infer_language_pair(path):
    """Infer language pair from filename: <split>.<lang1>-<lang2>.(...).idx"""
    src, dst = None, None
    for filename in os.listdir(path):
        parts = filename.split('.')
        if len(parts) >= 3 and len(parts[1].split('-')) == 2:
            return parts[1].split('-')
    return src, dst


def collate_tokens(values, pad_idx, eos_idx, left_pad, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res

from itertools import groupby
import torch
import numpy


def acl19_collate_tokens_3d(values, sentence_separator_idx, story_separator_idx, pad_idx, eos_idx, left_pad, move_eos_to_beginning=False):
    """Convert a list of tensors into a padded 3d tensor."""
    def split_list(input_list):
        tmp = [list(g) for k, g in groupby(input_list, lambda x: x == story_separator_idx) if not k]
        return tmp
    values = [v[v!=sentence_separator_idx] for v in values]
    result = [split_list(v) for v in values]
    # result = [s.split() for s in re.split('story_separator_special_tag|sentence_separator_special_tag', ' '.join(words)) if s]
    # nwords = max(len(words) for words in result)
    # ids = torch.IntTensor(len(result), nwords + 1).fill_(self.pad_index) if append_eos else torch.IntTensor(len(result), nwords).fill_(self.pad_index)
    max_sentence_count = max(len(src) for src in result)
    if max_sentence_count > 10:
        max_sentence_count = 10
    max_wordcount_per_sentence = max(max(len(src[i]) for i in range(len(src))) for src in result)
    if max_wordcount_per_sentence > 500:
        max_wordcount_per_sentence = 500
    res = values[0][0].new(len(result), max_sentence_count, max_wordcount_per_sentence).fill_(pad_idx)
    # size = max(v.size(0) for v in values)
    # res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(result):
        for j, s in enumerate(v):
            if j == max_sentence_count:
                break
            if len(s) > max_wordcount_per_sentence:
                s = s[:max_wordcount_per_sentence]
            copy_tensor(torch.IntTensor(s), res[i, j][max_wordcount_per_sentence - len(s):] if left_pad else res[i, j][:len(s)])
    # for i, v in enumerate(values):
    #     copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


def acl19_collate_tokens_3d_for_infer(values, sentence_separator_idx, story_separator_idx, pad_idx, eos_idx, left_pad, move_eos_to_beginning=False):
    """Convert a list of tensors into a padded 3d tensor."""
    def split_list(input_list):
        tmp = [list(g) for k, g in groupby(input_list, lambda x: x == story_separator_idx) if not k]
        return tmp
    values = [v[v!=sentence_separator_idx] for v in values]
    result = [split_list(v) for v in values]
    # result = [s.split() for s in re.split('story_separator_special_tag|sentence_separator_special_tag', ' '.join(words)) if s]
    # nwords = max(len(words) for words in result)
    # ids = torch.IntTensor(len(result), nwords + 1).fill_(self.pad_index) if append_eos else torch.IntTensor(len(result), nwords).fill_(self.pad_index)
    max_sentence_count = max(len(src) for src in result)
    # if max_sentence_count > 30:
    #     max_sentence_count = 30
    max_wordcount_per_sentence = max(max(len(src[i]) for i in range(len(src))) for src in result)
    # if max_wordcount_per_sentence > 60:
    #     max_wordcount_per_sentence = 60
    res = values[0][0].new(len(result), max_sentence_count, max_wordcount_per_sentence).fill_(pad_idx)
    # size = max(v.size(0) for v in values)
    # res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(result):
        for j, s in enumerate(v):
            if j == max_sentence_count:
                break
            if len(s) > max_wordcount_per_sentence:
                s = s[:max_wordcount_per_sentence]
            copy_tensor(torch.IntTensor(s), res[i, j][max_wordcount_per_sentence - len(s):] if left_pad else res[i, j][:len(s)])
    # for i, v in enumerate(values):
    #     copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res



def collate_tokens_3d(values, sentence_separator_idx, story_separator_idx, pad_idx, eos_idx, left_pad, move_eos_to_beginning=False):
    """Convert a list of tensors into a padded 3d tensor."""
    def split_list(input_list):
        tmp = [list(g) for k, g in groupby(input_list, lambda x: x == sentence_separator_idx or x == story_separator_idx) if not k]
        return tmp
    result = [split_list(v) for v in values]
    # result = [s.split() for s in re.split('story_separator_special_tag|sentence_separator_special_tag', ' '.join(words)) if s]
    # nwords = max(len(words) for words in result)
    # ids = torch.IntTensor(len(result), nwords + 1).fill_(self.pad_index) if append_eos else torch.IntTensor(len(result), nwords).fill_(self.pad_index)
    max_sentence_count = max(len(src) for src in result)
    if max_sentence_count > 30:
        max_sentence_count = 30
    max_wordcount_per_sentence = max(max(len(src[i]) for i in range(len(src))) for src in result)
    if max_wordcount_per_sentence > 80:
        max_wordcount_per_sentence = 80
    res = values[0][0].new(len(result), max_sentence_count, max_wordcount_per_sentence).fill_(pad_idx)
    # size = max(v.size(0) for v in values)
    # res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(result):
        for j, s in enumerate(v):
            if j == max_sentence_count:
                break
            if len(s) > max_wordcount_per_sentence:
                s = s[:max_wordcount_per_sentence]
            copy_tensor(torch.IntTensor(s), res[i, j][max_wordcount_per_sentence - len(s):] if left_pad else res[i, j][:len(s)])
    # for i, v in enumerate(values):
    #     copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res

def collate_tokens_3d_and_blockmask_docBlockMask(values, sentence_separator_idx, story_separator_idx, pad_idx, eos_idx, left_pad, move_eos_to_beginning=False):
    """Convert a list of tensors into a padded 3d tensor."""
    def split_list(input_list):
        tmp = [list(g) for k, g in groupby(input_list, lambda x: x == sentence_separator_idx or x == story_separator_idx) if not k]
        return tmp
    result = [split_list(v) for v in values]

    # generate mask matrix
    separator = [[sep for sep in val.tolist() if sep == sentence_separator_idx or sep == story_separator_idx] for val in values]
    separator = [val+[story_separator_idx] for val in separator]
    separator_ids = [list(range(1, len(s)+1)) for s in separator]

    doc_mask_len = torch.zeros(len(separator_ids), max(len(val) for val in separator_ids))
    doc_mask_len_shift = torch.zeros(len(separator_ids), max(len(val) for val in separator_ids))
    doc_lengths = []
    for batch, ids in enumerate(separator_ids):
        tmp = 0
        for i in range(len(ids)-1, -1, -1):
            if separator[batch][i] == story_separator_idx:
                doc_mask_len[batch][i] = ids[i]
                tmp = ids[i]
            else:
                doc_mask_len[batch][i] = tmp

    for batch, ids in enumerate(separator_ids):
        tmp = 0
        count = 0
        for i in range(len(ids)):
            if separator[batch][i] == story_separator_idx:
                doc_mask_len_shift[batch][i] = tmp
                tmp = ids[i]
                count += 1
            else:
                doc_mask_len_shift[batch][i] = tmp
        doc_lengths.append(count)

    doc_mask = torch.arange(0, torch.max(doc_mask_len)).type_as(doc_mask_len)\
        .repeat(doc_mask_len.size(0), doc_mask_len.size(1), 1).lt(doc_mask_len.unsqueeze(-1))
    doc_mask -= torch.arange(0, torch.max(doc_mask_len)).type_as(doc_mask_len_shift)\
        .repeat(doc_mask_len_shift.size(0), doc_mask_len_shift.size(1), 1).lt(doc_mask_len_shift.unsqueeze(-1))

    doc_block_mask = torch.zeros(len(separator_ids), max(doc_lengths), doc_mask.size(1))
    for batch, ids in enumerate(separator_ids):
        count_doc = 0
        count_sent = 0
        for i in range(len(ids)):
            if separator[batch][i] == story_separator_idx:
                doc_block_mask[batch][count_doc][count_sent] = 1
                count_doc += 1
                count_sent += 1
            else:
                doc_block_mask[batch][count_doc][count_sent] = 1
                count_sent += 1
    max_doc_count = max(doc_lengths)
    # if max_doc_count > 6:
    #     max_doc_count = 6
    # doc_lengths = [length if length <= 6 else 6 for length in doc_lengths]

    # make 3d input tensor
    max_sentence_count = max(len(src) for src in result)
    if max_sentence_count > 65:
        max_sentence_count = 65
    max_wordcount_per_sentence = max(max(len(src[i]) for i in range(len(src))) for src in result)
    if max_wordcount_per_sentence > 70:
        max_wordcount_per_sentence = 70
    res = values[0][0].new(len(result), max_sentence_count, max_wordcount_per_sentence).fill_(pad_idx)


    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(result):
        for j, s in enumerate(v):
            if j == max_sentence_count:
                break
            if len(s) > max_wordcount_per_sentence:
                s = s[:max_wordcount_per_sentence]
            copy_tensor(torch.IntTensor(s), res[i, j][max_wordcount_per_sentence - len(s):] if left_pad else res[i, j][:len(s)])


    return res, doc_mask[:, :max_sentence_count, :max_sentence_count], doc_block_mask[:, :max_doc_count, :max_sentence_count], torch.LongTensor(doc_lengths)


def collate_tokens_3d_and_blockmask_docBlockMask_for_infer(values, sentence_separator_idx, story_separator_idx, pad_idx, eos_idx, left_pad, move_eos_to_beginning=False):
    """Convert a list of tensors into a padded 3d tensor."""
    def split_list(input_list):
        tmp = [list(g) for k, g in groupby(input_list, lambda x: x == sentence_separator_idx or x == story_separator_idx) if not k]
        return tmp
    result = [split_list(v) for v in values]

    # generate mask matrix
    separator = [[sep for sep in val.tolist() if sep == sentence_separator_idx or sep == story_separator_idx] for val in values]
    separator = [val+[story_separator_idx] for i, val in enumerate(separator)]
    separator_ids = [list(range(1, len(s)+1)) for s in separator]

    doc_mask_len = torch.zeros(len(separator_ids), max(len(val) for val in separator_ids))
    doc_mask_len_shift = torch.zeros(len(separator_ids), max(len(val) for val in separator_ids))
    doc_lengths = []
    for batch, ids in enumerate(separator_ids):
        tmp = 0
        for i in range(len(ids)-1, -1, -1):
            if separator[batch][i] == story_separator_idx:
                doc_mask_len[batch][i] = ids[i]
                tmp = ids[i]
            else:
                doc_mask_len[batch][i] = tmp

    for batch, ids in enumerate(separator_ids):
        tmp = 0
        count = 0
        for i in range(len(ids)):
            if separator[batch][i] == story_separator_idx:
                doc_mask_len_shift[batch][i] = tmp
                tmp = ids[i]
                count += 1
            else:
                doc_mask_len_shift[batch][i] = tmp
        doc_lengths.append(count)

    doc_mask = torch.arange(0, torch.max(doc_mask_len)).type_as(doc_mask_len)\
        .repeat(doc_mask_len.size(0), doc_mask_len.size(1), 1).lt(doc_mask_len.unsqueeze(-1))
    doc_mask -= torch.arange(0, torch.max(doc_mask_len)).type_as(doc_mask_len_shift)\
        .repeat(doc_mask_len_shift.size(0), doc_mask_len_shift.size(1), 1).lt(doc_mask_len_shift.unsqueeze(-1))

    doc_block_mask = torch.zeros(len(separator_ids), max(doc_lengths), doc_mask.size(1))
    for batch, ids in enumerate(separator_ids):
        count_doc = 0
        count_sent = 0
        for i in range(len(ids)):
            if separator[batch][i] == story_separator_idx:
                doc_block_mask[batch][count_doc][count_sent] = 1
                count_doc += 1
                count_sent += 1
            else:
                doc_block_mask[batch][count_doc][count_sent] = 1
                count_sent += 1
    max_doc_count = max(doc_lengths)
    # if max_doc_count > 6:
    #     max_doc_count = 6
    # doc_lengths = [length if length <= 6 else 6 for length in doc_lengths]

    # make 3d input tensor
    max_sentence_count = max(len(src) for src in result)
    # if max_sentence_count > 65:
    #     max_sentence_count = 65
    max_wordcount_per_sentence = max(max(len(src[i]) for i in range(len(src))) for src in result)
    # if max_wordcount_per_sentence > 70:
    #     max_wordcount_per_sentence = 70
    res = values[0][0].new(len(result), max_sentence_count, max_wordcount_per_sentence).fill_(pad_idx)


    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(result):
        for j, s in enumerate(v):
            if j == max_sentence_count:
                break
            if len(s) > max_wordcount_per_sentence:
                s = s[:max_wordcount_per_sentence]
            copy_tensor(torch.IntTensor(s), res[i, j][max_wordcount_per_sentence - len(s):] if left_pad else res[i, j][:len(s)])


    return res, doc_mask[:, :max_sentence_count, :max_sentence_count], doc_block_mask[:, :max_doc_count, :max_sentence_count], torch.LongTensor(doc_lengths)


def collate_tokens_3d_and_docmask(values, sentence_separator_idx, story_separator_idx, pad_idx, eos_idx, left_pad, move_eos_to_beginning=False):
    """Convert a list of tensors into a padded 3d tensor."""
    def split_list(input_list):
        tmp = [list(g) for k, g in groupby(input_list, lambda x: x == sentence_separator_idx or x == story_separator_idx) if not k]
        return tmp
    result = [split_list(v) for v in values]

    # generate mask matrix
    separator = [[sep for sep in val.tolist() if sep == sentence_separator_idx or sep == story_separator_idx] for val in values]
    separator = [val+[story_separator_idx] for val in separator]
    separator_ids = [list(range(1, len(s)+1)) for s in separator]

    doc_mask_len = torch.zeros(len(separator_ids), max(len(val) for val in separator_ids))
    doc_mask_len_shift = torch.zeros(len(separator_ids), max(len(val) for val in separator_ids))
    for batch, ids in enumerate(separator_ids):
        tmp = 0
        for i in range(len(ids)-1, -1, -1):
            if separator[batch][i] == story_separator_idx:
                doc_mask_len[batch][i] = ids[i]
                tmp = ids[i]
            else:
                doc_mask_len[batch][i] = tmp
    for batch, ids in enumerate(separator_ids):
        tmp = 0
        for i in range(len(ids)):
            if separator[batch][i] == story_separator_idx:
                doc_mask_len_shift[batch][i] = tmp
                tmp = ids[i]
            else:
                doc_mask_len_shift[batch][i] = tmp

    doc_mask = torch.arange(0, torch.max(doc_mask_len)).type_as(doc_mask_len)\
        .repeat(doc_mask_len.size(0), doc_mask_len.size(1), 1).lt(doc_mask_len.unsqueeze(-1))
    doc_mask -= torch.arange(0, torch.max(doc_mask_len)).type_as(doc_mask_len_shift)\
        .repeat(doc_mask_len_shift.size(0), doc_mask_len_shift.size(1), 1).lt(doc_mask_len_shift.unsqueeze(-1))

    # make 3d input tensor
    max_sentence_count = max(len(src) for src in result)
    if max_sentence_count > 30:
        max_sentence_count = 30
    max_wordcount_per_sentence = max(max(len(src[i]) for i in range(len(src))) for src in result)
    if max_wordcount_per_sentence > 80:
        max_wordcount_per_sentence = 80
    res = values[0][0].new(len(result), max_sentence_count, max_wordcount_per_sentence).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(result):
        for j, s in enumerate(v):
            if j == max_sentence_count:
                break
            if len(s) > max_wordcount_per_sentence:
                s = s[:max_wordcount_per_sentence]
            copy_tensor(torch.IntTensor(s), res[i, j][max_wordcount_per_sentence - len(s):] if left_pad else res[i, j][:len(s)])

    return res, doc_mask[:, :max_sentence_count, :max_sentence_count]


def collate_tokens_3d_for_infer(values, sentence_separator_idx, story_separator_idx, pad_idx, eos_idx, left_pad, move_eos_to_beginning=False):
    """Convert a list of tensors into a padded 3d tensor."""
    def split_list(input_list):
        tmp = [list(g) for k, g in groupby(input_list, lambda x: x == sentence_separator_idx or x == story_separator_idx) if not k]
        return tmp
    result = [split_list(v) for v in values]
    # result = [s.split() for s in re.split('story_separator_special_tag|sentence_separator_special_tag', ' '.join(words)) if s]
    # nwords = max(len(words) for words in result)
    # ids = torch.IntTensor(len(result), nwords + 1).fill_(self.pad_index) if append_eos else torch.IntTensor(len(result), nwords).fill_(self.pad_index)
    max_sentence_count = max(len(src) for src in result)
    # if max_sentence_count > 30:
    #     max_sentence_count = 30
    max_wordcount_per_sentence = max(max(len(src[i]) for i in range(len(src))) for src in result)
    # if max_wordcount_per_sentence > 60:
    #     max_wordcount_per_sentence = 60
    res = values[0][0].new(len(result), max_sentence_count, max_wordcount_per_sentence).fill_(pad_idx)
    # size = max(v.size(0) for v in values)
    # res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(result):
        for j, s in enumerate(v):
            if j == max_sentence_count:
                break
            if len(s) > max_wordcount_per_sentence:
                s = s[:max_wordcount_per_sentence]
            copy_tensor(torch.IntTensor(s), res[i, j][max_wordcount_per_sentence - len(s):] if left_pad else res[i, j][:len(s)])
    # for i, v in enumerate(values):
    #     copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


def collate_tokens_3d_and_docmask_for_infer(values, sentence_separator_idx, story_separator_idx, pad_idx, eos_idx, left_pad, move_eos_to_beginning=False):
    """Convert a list of tensors into a padded 3d tensor."""
    def split_list(input_list):
        tmp = [list(g) for k, g in groupby(input_list, lambda x: x == sentence_separator_idx or x == story_separator_idx) if not k]
        return tmp
    result = [split_list(v) for v in values]

    # generate mask matrix
    separator = [[sep for sep in val.tolist() if sep == sentence_separator_idx or sep == story_separator_idx] for val in values]
    separator = [val+[story_separator_idx] for val in separator]
    separator_ids = [list(range(1, len(s)+1)) for s in separator]

    doc_mask_len = torch.zeros(len(separator_ids), max(len(val) for val in separator_ids))
    doc_mask_len_shift = torch.zeros(len(separator_ids), max(len(val) for val in separator_ids))
    for batch, ids in enumerate(separator_ids):
        tmp = 0
        for i in range(len(ids)-1, -1, -1):
            if separator[batch][i] == story_separator_idx:
                doc_mask_len[batch][i] = ids[i]
                tmp = ids[i]
            else:
                doc_mask_len[batch][i] = tmp
    for batch, ids in enumerate(separator_ids):
        tmp = 0
        for i in range(len(ids)):
            if separator[batch][i] == story_separator_idx:
                doc_mask_len_shift[batch][i] = tmp
                tmp = ids[i]
            else:
                doc_mask_len_shift[batch][i] = tmp

    doc_mask = torch.arange(0, torch.max(doc_mask_len)).type_as(doc_mask_len)\
        .repeat(doc_mask_len.size(0), doc_mask_len.size(1), 1).lt(doc_mask_len.unsqueeze(-1))
    doc_mask -= torch.arange(0, torch.max(doc_mask_len)).type_as(doc_mask_len_shift)\
        .repeat(doc_mask_len_shift.size(0), doc_mask_len_shift.size(1), 1).lt(doc_mask_len_shift.unsqueeze(-1))

    # make 3d input tensor
    max_sentence_count = max(len(src) for src in result)
    # if max_sentence_count > 30:
    #     max_sentence_count = 30
    max_wordcount_per_sentence = max(max(len(src[i]) for i in range(len(src))) for src in result)
    # if max_wordcount_per_sentence > 60:
    #     max_wordcount_per_sentence = 60
    res = values[0][0].new(len(result), max_sentence_count, max_wordcount_per_sentence).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(result):
        for j, s in enumerate(v):
            if j == max_sentence_count:
                break
            if len(s) > max_wordcount_per_sentence:
                s = s[:max_wordcount_per_sentence]
            copy_tensor(torch.IntTensor(s), res[i, j][max_wordcount_per_sentence - len(s):] if left_pad else res[i, j][:len(s)])

    return res, doc_mask[:, :max_sentence_count, :max_sentence_count]


from itertools import chain
import itertools
def collate_tokens_2d_for_infer(values, sentence_separator_idx, story_separator_idx, pad_idx, eos_idx, left_pad, move_eos_to_beginning=False):
    """Convert a list of tensors into a padded 3d tensor."""
    def split_list(input_list):
        tmp = [list(g) for k, g in groupby(input_list, lambda x: x == sentence_separator_idx or x == story_separator_idx) if not k]
        return tmp
    result = [split_list(v) for v in values]
    values = [list(itertools.chain.from_iterable(r)) for r in result]
    size = max(len(v) for v in values)
    values = [torch.LongTensor(v) for v in values]
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res




def collate_masks(values, pad_idx, left_pad):
    """Convert a list of 2d tensors into a padded 3d tensor."""
    size1 = max(v.size(0) for v in values)
    size2 = max(v.size(1) for v in values)
    res = values[0].new(len(values), size1, size2).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i, size1 - v.size(0):, size2 - v.size(1):] if left_pad else res[i, :v.size(0), :v.size(1)])
    # tmp = res[-1]
    # tmp2 = res[-1, :values[-1].size(0), :values[-1].size(1)]
    # a = 1 + 1
    return res


@contextlib.contextmanager
def numpy_seed(seed):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def collect_filtered(function, iterable, filtered):
    """
    Similar to :func:`filter` but collects filtered elements in ``filtered``.

    Args:
        function (callable): function that returns ``False`` for elements that
            should be filtered
        iterable (iterable): iterable to filter
        filtered (list): list to store filtered elements
    """
    for el in iterable:
        if function(el):
            yield el
        else:
            filtered.append(el)


def filter_by_size(indices, size_fn, max_positions, raise_exception=False):
    """
    Filter indices based on their size.

    Args:
        indices (List[int]): ordered list of dataset indices
        size_fn (callable): function that returns the size of a given index
        max_positions (tuple): filter elements larger than this size.
            Comparisons are done component-wise.
        raise_exception (bool, optional): if ``True``, raise an exception if
            any elements are filtered (default: False).
    """
    def check_size(idx):
        if isinstance(max_positions, float) or isinstance(max_positions, int):
            return size_fn(idx) <= max_positions
        elif isinstance(max_positions, dict):
            idx_size = size_fn(idx)
            assert isinstance(idx_size, dict)
            intersect_keys = set(max_positions.keys()) & set(idx_size.keys())
            return all(
                all(a is None or b is None or a <= b
                    for a, b in zip(idx_size[key], max_positions[key]))
                for key in intersect_keys
            )
        else:
            # For MultiCorpusSampledDataset, will generalize it later
            if not isinstance(size_fn(idx), Iterable):
                return all(size_fn(idx) <= b for b in max_positions)
            return all(a is None or b is None or a <= b
                       for a, b in zip(size_fn(idx), max_positions))

    ignored = []
    itr = collect_filtered(check_size, indices, ignored)

    for idx in itr:
        if len(ignored) > 0 and raise_exception:
            raise Exception((
                'Size of sample #{} is invalid (={}) since max_positions={}, '
                'skip this example with --skip-invalid-size-inputs-valid-test'
            ).format(ignored[0], size_fn(ignored[0]), max_positions))
        yield idx

    if len(ignored) > 0:
        print((
            '| WARNING: {} samples have invalid sizes and will be skipped, '
            'max_positions={}, first few sample ids={}'
        ).format(len(ignored), max_positions, ignored[:10]))


def batch_by_size(
    indices, num_tokens_fn, max_tokens=None, max_sentences=None,
    required_batch_size_multiple=1,
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        max_tokens (int, optional): max number of tokens in each batch
            (default: None).
        max_sentences (int, optional): max number of sentences in each
            batch (default: None).
        required_batch_size_multiple (int, optional): require batch size to
            be a multiple of N (default: 1).
    """
    max_tokens = max_tokens if max_tokens is not None else float('Inf')
    max_sentences = max_sentences if max_sentences is not None else float('Inf')
    bsz_mult = required_batch_size_multiple

    batch = []

    def is_batch_full(num_tokens):
        if len(batch) == 0:
            return False
        if len(batch) == max_sentences:
            return True
        if num_tokens > max_tokens:
            return True
        return False

    sample_len = 0
    sample_lens = []
    for idx in indices:
        sample_lens.append(num_tokens_fn(idx))
        sample_len = max(sample_len, sample_lens[-1])
        assert sample_len <= max_tokens, (
            f"sentence at index {idx} of size {sample_len} exceeds max_tokens "
            f"limit of {max_tokens}!"
        )
        num_tokens = (len(batch) + 1) * sample_len
        if is_batch_full(num_tokens):
            mod_len = max(
                bsz_mult * (len(batch) // bsz_mult),
                len(batch) % bsz_mult,
            )
            yield batch[:mod_len]
            batch = batch[mod_len:]
            sample_lens = sample_lens[mod_len:]
            sample_len = max(sample_lens) if len(sample_lens) > 0 else 0

        batch.append(idx)

    if len(batch) > 0:
        yield batch


def process_bpe_symbol(sentence: str, bpe_symbol: str):
    if bpe_symbol == 'sentencepiece':
        sentence = sentence.replace(' ', '').replace('\u2581', ' ').strip()
    elif bpe_symbol is not None:
        sentence = (sentence + ' ').replace(bpe_symbol, '').rstrip()
    return sentence
