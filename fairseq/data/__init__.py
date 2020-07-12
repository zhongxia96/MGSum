# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from .dictionary import Dictionary, TruncatedDictionary
from .fairseq_dataset import FairseqDataset
from .backtranslation_dataset import BacktranslationDataset
from .concat_dataset import ConcatDataset
from .indexed_dataset import IndexedCachedDataset, IndexedDataset, IndexedRawTextDataset, MMapIndexedDataset
from .language_pair_dataset import LanguagePairDataset
from .multi_news_pair_dataset import LanguagePairMultiNewsDataset
from .multi_news_pair_dataset_for_infer import LanguagePairMultiNewsDataset_for_infer

from .lm_context_window_dataset import LMContextWindowDataset
from .monolingual_dataset import MonolingualDataset
from .noising import NoisingDataset
from .round_robin_zip_datasets import RoundRobinZipDatasets
from .token_block_dataset import TokenBlockDataset
from .transform_eos_dataset import TransformEosDataset
from .transform_eos_lang_pair_dataset import TransformEosLangPairDataset
from .language_pair_hierarchical_dataset import LanguagePairHierarchicalDataset
from .language_pair_hierarchical_dataset_for_infer import LanguagePairHierarchicalDataset_for_infer

from .iterators import (
    CountingIterator,
    EpochBatchIterator,
    GroupedIterator,
    ShardedIterator,
)

__all__ = [
    'BacktranslationDataset',
    'ConcatDataset',
    'CountingIterator',
    'Dictionary',
    'EpochBatchIterator',
    'FairseqDataset',
    'GroupedIterator',
    'IndexedCachedDataset',
    'IndexedDataset',
    'IndexedRawTextDataset',
    'LanguagePairHierarchicalDataset_for_infer',
    'LanguagePairHierarchicalDataset',
    'LanguagePairDataset',
    'LanguagePairMultiNewsDataset',
    'LanguagePairMultiNewsDataset_for_infer',
    'LMContextWindowDataset',
    'MMapIndexedDataset',
    'MonolingualDataset',
    'NoisingDataset',
    'RoundRobinZipDatasets',
    'ShardedIterator',
    'TokenBlockDataset',
    'TransformEosDataset',
    'TransformEosLangPairDataset',
    'TruncatedDictionary',
]
