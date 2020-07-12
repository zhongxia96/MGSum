# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import Counter
import os
import json
import nltk
from itertools import chain
import torch
import numpy as np

from fairseq.tokenizer import tokenize_line


def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


class Binarizer:

    @staticmethod
    def binarize(filename, dict, consumer, tokenize=tokenize_line, append_eos=True, reverse_order=False,
                 offset=0, end=-1):
        nseq, ntok = 0, 0
        replaced = Counter()

        def replaced_consumer(word, idx):
            if idx == dict.unk_index and word != dict.unk_word:
                replaced.update([word])

        with open(filename, 'r', encoding='utf-8') as f:
            f.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)
            while line:
                if end > 0 and f.tell() > end:
                    break
                ids = dict.encode_line(
                        line=line,
                        line_tokenizer=tokenize,
                        add_if_not_exist=False,
                        consumer=replaced_consumer,
                        append_eos=append_eos,
                        reverse_order=reverse_order,
                )
                nseq += 1
                ntok += len(ids)
                consumer(ids)
                line = f.readline()
        return {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': replaced}

    @staticmethod
    def binarize_sent_doc(filename, consumer, append_eos=True, reverse_order=False,
                 offset=0, end=-1):
        nseq, ntok = 0, 0
        replaced = Counter()
        with open(filename, 'r', encoding='utf-8') as f:
            f.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)
            while line:
                if end > 0 and f.tell() > end:
                    break
                ids = json.loads(line)
                ids = torch.IntTensor(ids)
                nseq += 1
                ntok += len(ids)
                consumer(ids)
                line = f.readline()
        return {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': replaced}

    @staticmethod
    def binarize_position(filename, consumer, tokenize=tokenize_line, append_eos=True, reverse_order=False,
                              offset=0, end=-1):
        nseq, ntok = 0, 0
        replaced = Counter()

        def encode_positions(line, sentence_tokenizer, line_tokenizer=tokenize_line, add_if_not_exist=True,
                             append_eos=True, reverse_order=False):

            def split_sentence(paragraph, tokenizer):
                # tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
                sentences = tokenizer.tokenize(paragraph)
                return sentences

            words = list(split_sentence(line, sentence_tokenizer))
            words = [line_tokenizer(str(w)) for w in words]
            ids = torch.IntTensor(len(words))
            for i, word in enumerate(words):
                ids[i] = len(word)
            return ids

        tokenize_sentence = nltk.data.load('tokenizers/punkt/english.pickle')

        with open(filename, 'r', encoding='utf-8') as f:
            f.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)
            while line:
                if end > 0 and f.tell() > end:
                    break
                ids = encode_positions(
                    line=line,
                    sentence_tokenizer=tokenize_sentence,
                    line_tokenizer=tokenize_line,
                    add_if_not_exist=False,
                    append_eos=append_eos,
                    reverse_order=reverse_order,
                )
                nseq += 1
                ntok += len(ids)
                consumer(ids)
                # ntok += ids.ne(dict.pad_index).sum().item()
                line = f.readline()
        return {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': replaced}

    @staticmethod
    def binarize_hierarchical(filename, dict, consumer, tokenize=tokenize_line, append_eos=True, reverse_order=False,
                 offset=0, end=-1):
        nseq, ntok = 0, 0
        replaced = Counter()

        def replaced_consumer(word, idx):
            if idx == dict.unk_index and word != dict.unk_word:
                replaced.update([word])

        tokenize_sentence = nltk.data.load('tokenizers/punkt/english.pickle')

        with open(filename, 'r', encoding='utf-8') as f:
            f.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)
            while line:
                if end > 0 and f.tell() > end:
                    break
                ids = dict.encode_hierarchical_line(
                    line=line,
                    sentence_tokenizer=tokenize_sentence,
                    line_tokenizer=tokenize,
                    add_if_not_exist=False,
                    consumer=replaced_consumer,
                    append_eos=append_eos,
                    reverse_order=reverse_order,
                )
                nseq += 1
                ntok += len(ids)
                # ntok += ids.ne(dict.pad_index).sum().item()
                consumer(ids)
                line = f.readline()
        return {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': replaced}

    @staticmethod
    def binarize_graph(filename, dict, consumer, append_eos=True, reverse_order=False,
                 offset=0, end=-1):
        nseq, ntok = 0, 0
        replaced = Counter()

        with open(filename, 'r', encoding='utf-8') as f:
            f.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)
            while line:
                line = json.loads(line)
                if end > 0 and f.tell() > end:
                    break
                ids = dict.encode_graph(
                        line=line,
                )
                nseq += 1
                ntok += len(ids)
                consumer(ids)
                line = f.readline()
        return {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': replaced}


    def binarize_mask(filename, consumer, append_eos=True, reverse_order=False,
                 offset=0, end=-1):
        nseq, ntok = 0, 0
        replaced = Counter()

        def encode_mask(mask_item):
            keys = list(mask_item.keys())
            keys = [[int(k)] for k in keys]
            values = list(mask_item.values())
            lists = list(chain.from_iterable(zip(keys, values)))
            lists = list(chain.from_iterable(zip(lists, [[-1] for _ in range(len(lists))])))
            lists = list(chain(*lists))
            return torch.IntTensor(np.array(lists))

        with open(filename, 'r', encoding='utf-8') as f:
            f.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)
            while line:
                line = json.loads(line)
                if end > 0 and f.tell() > end:
                    break
                ids = encode_mask(mask_item=line)
                nseq += 1
                ntok += len(ids)
                consumer(ids)
                line = f.readline()
        return {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': replaced}


    def binarize_graph2(filename, dict, consumer, append_eos=True, reverse_order=False,
                 offset=0, end=-1):
        nseq, ntok = 0, 0
        replaced = Counter()

        with open(filename, 'r', encoding='utf-8') as f:
            f.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)
            while line:
                line = json.loads(line)
                if end > 0 and f.tell() > end:
                    break
                ids = dict.encode_graph2(
                        line=line,
                )
                nseq += 1
                ntok += len(ids)
                consumer(ids)
                line = f.readline()
        return {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': replaced}

    @staticmethod
    def find_offsets(filename, num_chunks):
        with open(filename, 'r', encoding='utf-8') as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_chunks
            offsets = [0 for _ in range(num_chunks + 1)]
            for i in range(1, num_chunks):
                f.seek(chunk_size * i)
                safe_readline(f)
                offsets[i] = f.tell()
            return offsets
