# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import contextlib
from dataclasses import dataclass, field
from typing import Optional
from omegaconf import MISSING, II, open_dict, OmegaConf

import numpy as np
from fairseq.data import (
    ConcatSentencesDataset,
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    OffsetTokensDataset,
    PrependTokenDataset,
    RawLabelDataset,
    RightPadDataset,
    RollDataset,
    SortDataset,
    StripTokenDataset,
    data_utils,
)
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.data.lev_mask_tokens_dataset import LevenshteinMaskTokensDataset
from fairseq.tasks import FairseqDataclass, FairseqTask, register_task
from fairseq.dataclass import ChoiceEnum


logger = logging.getLogger(__name__)
SHORTEN_METHOD_CHOICES = ChoiceEnum(["none", "truncate", "random_crop"])


@dataclass
class LevenshteinPredictionConfig(FairseqDataclass):
    data: str = field(
        default=MISSING, metadata={"help": "path to data directory"}
    )
    init_token: Optional[int] = field(
        default=None,
        metadata={"help": "add token at the beginning of each batch item"},
    )
    separator_token: Optional[int] = field(
        default=None, metadata={"help": "add separator token between inputs"},
    )
    no_shuffle: bool = field(default=False,)
    shorten_method: SHORTEN_METHOD_CHOICES = field(
        default="none",
        metadata={
            "help": "if not none, shorten sequences that exceed tokens_per_sample"
        },
    )
    shorten_data_split_list: str = field(
        default="",
        metadata={
            "help": "comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)'
        },
    )
    add_prev_output_tokens: bool = field(
        default=False,
        metadata={
            "help": "add prev_output_tokens to sample, used for encoder-decoder arch"
        },
    )
    max_positions: int = field(
        default=512, metadata={"help": "max tokens per example"},
    )

    delta_x_loss: bool = field(default=False)

    levenshtein_head_name: str = II(
        "criterion.levenshtein_prediction_head_name"
    )
    delta_x_head_name: str = II("criterion.delta_x_head_name")

    mask_prob: float = field(
        default=0.15,
        metadata={"help": "probability of replacing a token with mask"},
    )
    leave_unmasked_prob: float = field(
        default=0.1,
        metadata={"help": "probability that a masked token is unmasked"},
    )
    random_token_prob: float = field(
        default=0.1,
        metadata={
            "help": "probability of replacing a token with a random token"
        },
    )
    freq_weighted_replacement: bool = field(
        default=False,
        metadata={
            "help": "sample random replacement words based on word frequencies"
        },
    )
    mask_whole_words: bool = field(
        default=False,
        metadata={"help": "mask whole words; you may also want to set --bpe"},
    )
    mask_multiple_length: int = field(
        default=1, metadata={"help": "repeat the mask indices multiple times"},
    )
    mask_stdev: float = field(
        default=0.0, metadata={"help": "stdev of the mask length"},
    )

    seed: int = II("common.seed")


@register_task("levenshtein_prediction", dataclass=LevenshteinPredictionConfig)
class LevenshteinPredictionTask(FairseqTask):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    def __init__(self, cfg, data_dictionary, label_dictionary):
        super().__init__(cfg)
        self.dictionary = data_dictionary
        self._label_dictionary = label_dictionary

        # add mask token
        self.mask_idx = self.dictionary.add_symbol("<mask>")

    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        dictionary.add_symbol("<mask>")
        return dictionary

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        # load data dictionary
        data_dict = cls.load_dictionary(
            os.path.join(cfg.data, "input0", "dict.txt"),
        )
        logger.info("[input] dictionary: {} types".format(len(data_dict)))

        label_dict = cls.load_dictionary(
            os.path.join(cfg.data, "label0", "dict.txt"),
        )
        logger.info("[label] dictionary: {} types".format(len(label_dict)))

        return cls(cfg, data_dict, label_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        def get_path(key, split):
            return os.path.join(self.cfg.data, key, split)

        def make_dataset(key, dictionary):
            split_path = get_path(key, split)

            try:
                dataset = data_utils.load_indexed_dataset(
                    split_path, dictionary, combine=combine,
                )
            except Exception as e:
                if "StorageException: [404] Path not found" in str(e):
                    logger.warning(f"dataset {e} not found")
                    dataset = None
                else:
                    raise e
            return dataset

        input0 = make_dataset("input0", self.source_dictionary)
        assert input0 is not None, "could not find dataset: {}".format(
            get_path("input0", split)
        )
        input1 = make_dataset("input1", self.source_dictionary)

        if self.cfg.init_token is not None:
            input0 = PrependTokenDataset(input0, self.cfg.init_token)

        numel_input0 = NumelDataset(input0, reduce=True)

        if input1 is None:
            src_tokens = input0
        else:
            if self.cfg.separator_token is not None:
                input1 = PrependTokenDataset(input1, self.cfg.separator_token)

            src_tokens = ConcatSentencesDataset(input0, input1)

        with data_utils.numpy_seed(self.cfg.seed):
            shuffle = np.random.permutation(len(src_tokens))

        src_tokens = maybe_shorten_dataset(
            src_tokens,
            split,
            self.cfg.shorten_data_split_list,
            self.cfg.shorten_method,
            self.max_positions(),
            self.cfg.seed,
        )

        # edit ops for x_minus
        label0 = make_dataset("label0", self.label_dictionary)
        label0 = PrependTokenDataset(label0, self.cfg.init_token)

        # edit ops for x_plus
        label1 = make_dataset("label1", self.label_dictionary)
        label1 = PrependTokenDataset(label1, self.cfg.separator_token)

        # add CLS, concat with SEP in the middle
        label_tokens = ConcatSentencesDataset(label0, label1)
        label_tokens = maybe_shorten_dataset(
            label_tokens,
            split,
            self.cfg.shorten_data_split_list,
            self.cfg.shorten_method,
            self.max_positions(),
            self.cfg.seed,
        )

        # create masked input and targets
        mask_whole_words = (
            get_whole_word_mask(self.cfg, self.source_dictionary)
            if self.cfg.mask_whole_words
            else None
        )

        src_tokens, mask_tgt_tokens = LevenshteinMaskTokensDataset.apply_mask(
            src_tokens,
            numel_input0,
            label_tokens,
            self.source_dictionary,
            pad_idx=self.source_dictionary.pad(),
            mask_idx=self.mask_idx,
            seed=self.cfg.seed,
            mask_prob=self.cfg.mask_prob,
            leave_unmasked_prob=self.cfg.leave_unmasked_prob,
            random_token_prob=self.cfg.random_token_prob,
            freq_weighted_replacement=self.cfg.freq_weighted_replacement,
            mask_whole_words=mask_whole_words,
            mask_multiple_length=self.cfg.mask_multiple_length,
            mask_stdev=self.cfg.mask_stdev,
        )

        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": RightPadDataset(
                    src_tokens, pad_idx=self.source_dictionary.pad(),
                ),
                "src_lengths": NumelDataset(src_tokens, reduce=False),
            },
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens, reduce=True),
            "mask_target": RightPadDataset(
                mask_tgt_tokens, pad_idx=self.source_dictionary.pad(),
            ),
            "levenshtein_target": RightPadDataset(
                label_tokens, pad_idx=self.label_dictionary.pad(),
            ),
        }

        if self.cfg.add_prev_output_tokens:
            prev_tokens_dataset = RightPadDataset(
                RollDataset(src_tokens, 1), pad_idx=self.dictionary.pad(),
            )
            dataset["net_input"].update(prev_output_tokens=prev_tokens_dataset,)

        # delta_x tokens
        if self.cfg.delta_x_loss:
            label2 = make_dataset("label2", self.source_dictionary)
            delta_x_tokens = maybe_shorten_dataset(
                label2,
                split,
                self.cfg.shorten_data_split_list,
                self.cfg.shorten_method,
                self.max_positions(),
                self.cfg.seed,
            )

            dataset.update(
                {
                    "delta_x_target": RightPadDataset(
                        delta_x_tokens, pad_idx=self.source_dictionary.pad(),
                    )
                }
            )

        nested_dataset = NestedDictionaryDataset(
            dataset, sizes=[src_tokens.sizes],
        )

        if self.cfg.no_shuffle:
            dataset = nested_dataset
        else:
            dataset = SortDataset(
                nested_dataset,
                # shuffle
                sort_order=[shuffle],
            )

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, cfg):
        from fairseq import models

        with open_dict(cfg) if OmegaConf.is_config(
            cfg
        ) else contextlib.ExitStack():
            cfg.max_positions = self.cfg.max_positions

        model = models.build_model(cfg, self)

        # hardcode number of classes
        model.register_classification_head(
            self.cfg.levenshtein_head_name, num_classes=5, pooling=False
        )

        if self.cfg.delta_x_loss is True:
            model.register_classification_head(
                self.cfg.delta_x_head_name,
                num_classes=len(self.source_dictionary),
                pooling=True,
            )

        return model

    def max_positions(self):
        return self.cfg.max_positions

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    @property
    def label_dictionary(self):
        return self._label_dictionary
