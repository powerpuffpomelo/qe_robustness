# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """

from __future__ import absolute_import, division, print_function

import linecache
from io import open
from multiprocessing import Pool, cpu_count

import torch
from torch.functional import split
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels, x0=None, y0=None, x1=None, y1=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
            bbox: (Optional) list. The bounding boxes for each word of the sequence.
        """
        self.guid = guid
        self.words = words
        self.labels = labels
        if x0 is None:
            self.bboxes = None
        else:
            self.bboxes = [[a, b, c, d] for a, b, c, d in zip(x0, y0, x1, y1)]


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, bboxes=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        if bboxes:
            self.bboxes = bboxes


def read_examples_from_file(data_file, mode, bbox=False):
    file_path = data_file
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if bbox:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        examples.append(
                            InputExample(
                                guid="{}-{}".format(mode, guid_index),
                                words=words,
                                labels=labels,
                                x0=x0,
                                y0=y0,
                                x1=x1,
                                y1=y1,
                            )
                        )
                        guid_index += 1
                        words = []
                        labels = []
                        x0 = []
                        y0 = []
                        x1 = []
                        y1 = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[1].replace("\n", ""))
                        x0.append(split[2].replace("\n", ""))
                        y0.append(split[3].replace("\n", ""))
                        x1.append(split[4].replace("\n", ""))
                        y1.append(split[5].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            else:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        examples.append(
                            InputExample(guid="{}-{}".format(mode, guid_index), words=words, labels=labels)
                        )
                        guid_index += 1
                        words = []
                        labels = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
        if words:
            if bbox:
                examples.append(
                    InputExample(
                        guid="%s-%d".format(mode, guid_index), words=words, labels=labels, x0=x0, y0=y0, x1=x1, y1=y1
                    )
                )
            else:
                examples.append(InputExample(guid="%s-%d".format(mode, guid_index), words=words, labels=labels))
    return examples


def sweep_config_to_sweep_values(sweep_config):
    """
    Converts an instance of wandb.Config to plain values map.

    wandb.Config varies across versions quite significantly,
    so we use the `keys` method that works consistently.
    """

    return {key: sweep_config[key] for key in sweep_config.keys()}


def get_examples_from_df(data, bbox=False):
    if bbox:
        return [
            InputExample(
                guid=sentence_id,
                words=sentence_df["words"].tolist(),
                labels=sentence_df["labels"].tolist(),
                x0=sentence_df["x0"].tolist(),
                y0=sentence_df["y0"].tolist(),
                x1=sentence_df["x1"].tolist(),
                y1=sentence_df["y1"].tolist(),
            )
            for sentence_id, sentence_df in data.groupby(["sentence_id"])
        ]
    else:
        return [
            InputExample(guid=sentence_id, words=sentence_df["words"].tolist(), labels=sentence_df["labels"].tolist(), )
            for sentence_id, sentence_df in data.groupby(["sentence_id"])
        ]


def convert_example_to_feature(example_row):
    (
        example,
        label_map,
        max_seq_length,
        tokenizer,
        cls_token_at_end,
        cls_token,
        cls_token_segment_id,
        sep_token,
        sep_token_extra,
        pad_on_left,
        pad_token,
        pad_token_segment_id,
        pad_token_label_id,
        sequence_a_segment_id,
        mask_padding_with_zero,
    ) = example_row

    tokens = []
    label_ids = []
    bboxes = []
    if example.bboxes:
        for word, label, bbox in zip(example.words, example.labels, example.bboxes):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
            bboxes.extend([bbox] * len(word_tokens))

        cls_token_box = [0, 0, 0, 0]
        sep_token_box = [1000, 1000, 1000, 1000]
        pad_token_box = [0, 0, 0, 0]

    else:
        """
        example.words ['rowboats', 'and', 'pedalboats', 'can', 'be', 'rented', 'at', 'the', 'boathouse', '.', '[SEP]', '_', '游艇', '_', '和', '_', '脚踏船', '_', '可以', '_', '在', '_', '船坞', '_', '租用', '_', '.', '_']
        example.labels ['OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK']
        """
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)  # bpe ['▁', 'row', 'bo', 'ats']
            
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            if word_tokens:  # avoid non printable character like '\u200e' which are tokenized as a void token ''
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))  # [0, -100, -100, -100]  pad_token_label_id 是 -100
    
    """
    print(sep_token) # </s>
    print(cls_token) # <s>

    print(tokens)
    print(label_ids)
    ['▁', 'row', 'bo', 'ats', '▁and', '▁pedal', 'bo', 'ats', '▁can', '▁be', '▁rent', 'ed', '▁at', '▁the', '▁boat', 'house', '▁', '.', '▁[', 'S', 'EP', ']', '▁_', '▁', '游', '艇', '▁_', '▁', '和', '▁_', '▁', '脚', '踏', '船', '▁_', '▁', '可以', '▁_', '▁在', '▁_', '▁', '船', '坞', '▁_', '▁', '租', '用', '▁_', '▁', '.', '▁_']
    [0, -100, -100,     -100,    0,          0,   -100,  -100,     0,       0,       0,    -100,    0,      0,       0,     -100,     0, -100,   0,  -100, -100, -100, 0,     0,  -100, -100,   0,     0,  -100,   0,    0,  -100, -100, -100,   0,     0,   -100,    0,     0,     0,     0,  -100, -100,  0,     0,  -100, -100,    0,    0, -100,   0]
    这个bpe感觉非常离谱
    """

    # 长度裁剪
    # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    special_tokens_count = 3 if sep_token_extra else 2
    if len(tokens) > max_seq_length - special_tokens_count:
        tokens = tokens[: (max_seq_length - special_tokens_count)]
        label_ids = label_ids[: (max_seq_length - special_tokens_count)]
        if bboxes:
            bboxes = bboxes[: (max_seq_length - special_tokens_count)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens += [sep_token]
    label_ids += [pad_token_label_id]
    if bboxes:
        bboxes += [sep_token_box]
    if sep_token_extra:
        # roberta uses an extra separator b/w pairs of sentences
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        if bboxes:
            bboxes += [sep_token_box]
    segment_ids = [sequence_a_segment_id] * len(tokens)  # sequence_a_segment_id = 0

    if cls_token_at_end:
        tokens += [cls_token]
        label_ids += [pad_token_label_id]
        segment_ids += [cls_token_segment_id]
    else:
        tokens = [cls_token] + tokens
        label_ids = [pad_token_label_id] + label_ids
        segment_ids = [cls_token_segment_id] + segment_ids
        if bboxes:
            bboxes = [cls_token_box] + bboxes

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    """
    tokens 开头补<s>，结尾补</s>；开头cls，结尾sep
    ['<s>', '▁', 'row', 'bo', 'ats', '▁and', '▁pedal', 'bo', 'ats', '▁can', '▁be', '▁rent', 'ed', '▁at', '▁the', '▁boat', 'house', '▁', '.', '▁[', 'S', 'EP', ']', '▁_', '▁', '游', '艇', '▁_', '▁', '和', '▁_', '▁', '脚', '踏', '船', '▁_', '▁', '可以', '▁_', '▁在', '▁_', '▁', '船', '坞', '▁_', '▁', '租', '用', '▁_', '▁', '.', '▁_', '</s>']
    input_ids 开头补0，结尾补2；pad的时候都用1
    [0, 6, 15555, 837, 9821, 136, 95212, 837, 9821, 831, 186, 14457, 297, 99, 70, 139098, 23923, 6, 5, 378, 294, 21290, 268, 101, 6, 13621, 125047, 101, 6, 264, 101, 6, 24986, 49841, 15065, 101, 6, 1441, 101, 6376, 101, 6, 15065, 246215, 101, 6, 32981, 1173, 101, 6, 5, 101, 2]
    """

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)   # input_mask标识输入长度，有token的地方是1，pad是0

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        label_ids = ([pad_token_label_id] * padding_length) + label_ids
    else:
        input_ids += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
        label_ids += [pad_token_label_id] * padding_length
        if bboxes:
            bboxes += [pad_token_box] * padding_length

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    """
    print('here==============')
    print(input_ids)
    print(sum(torch.tensor(input_ids) != 1))
    print(len(input_ids))
    print(input_mask)   # 有token就是1，pad就是0，先1后0
    print(segment_ids)  # 全是0
    print(label_ids)    # label_ids由0和-100组成，不大搞得懂
    assert 1==2
    """

    if bboxes:
        assert len(bboxes) == max_seq_length

    if bboxes:
        return InputFeatures(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_ids=label_ids, bboxes=bboxes
        )
    else:
        return InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_ids=label_ids, )


def convert_examples_to_features(
        examples,
        label_list,
        max_seq_length,
        tokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-1,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
        process_count=cpu_count() - 2,
        chunksize=500,
        silent=False,
        use_multiprocessing=True,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}  # {'OK': 0, 'BAD': 1}

    examples = [
        (
            example,
            label_map,
            max_seq_length,
            tokenizer,
            cls_token_at_end,
            cls_token,
            cls_token_segment_id,
            sep_token,
            sep_token_extra,
            pad_on_left,
            pad_token,
            pad_token_segment_id,
            pad_token_label_id,
            sequence_a_segment_id,
            mask_padding_with_zero,
        )
        for example in examples
    ]
    """
    [(<transquest.algo.word_level.microtransquest.utils.InputExample object at 0x7f7ee71f9910>, {'OK': 0, 'BAD': 1}, 200, PreTrainedTokenizer(name_or_path='/home/yanym/save/transquest/outputs/best_model', vocab_size=250002, model_max_len=512, is_fast=False, padding_side='right', special_tokens={'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'sep_token': '</s>', 'pad_token': '<pad>', 'cls_token': '<s>', 'mask_token': AddedToken("<mask>", rstrip=False, lstrip=True, single_word=False, normalized=True)}), False, '<s>', 0, '</s>', False, False, 1, 0, -100, 0, True)
    """

    if use_multiprocessing:
        with Pool(process_count) as p:
            features = list(
                tqdm(
                    p.imap(convert_example_to_feature, examples, chunksize=chunksize),
                    total=len(examples),
                    disable=silent,
                )
            )
    else:
        features = []
        for example in tqdm(examples):
            features.append(convert_example_to_feature(example))
    return features


def get_labels(path):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "SEP" not in labels:
            labels = ["SEP"] + labels
        return labels
    else:
        return [
            "SEP",
            "OK",
            "BAD",
        ]


class LazyQEDataset(Dataset):
    def __init__(self, data_file, tokenizer, args):
        self.data_file = data_file
        self.lazy_loading_start_line = args.lazy_loading_start_line if args.lazy_loading_start_line else 0
        self.example_lines, self.num_entries = self._get_examples(self.data_file, self.lazy_loading_start_line)
        self.tokenizer = tokenizer
        self.args = args
        self.pad_token_label_id = CrossEntropyLoss().ignore_index

    @staticmethod
    def _get_examples(data_file, lazy_loading_start_line):
        example_lines = {}
        start = lazy_loading_start_line
        entry_num = 0
        with open(data_file, encoding="utf-8") as f:
            for line_idx, _ in enumerate(f, 1):
                if _ == "\n" and line_idx > lazy_loading_start_line:
                    example_lines[entry_num] = (start, line_idx)
                    start = line_idx + 1
                    entry_num += 1

        return example_lines, entry_num

    def __getitem__(self, idx):
        start, end = self.example_lines[idx]
        words, labels = [], []
        for idx in range(start, end):
            line = linecache.getline(self.data_file, idx).rstrip("\n")
            splits = line.split(" ")
            words.append(splits[0])
            if len(splits) > 1:
                labels.append(splits[-1].replace("\n", ""))
            else:
                # Examples could have no label for mode = "test"
                labels.append("O")
        if words:
            example = InputExample(guid="%s-%d".format("train", idx), words=words, labels=labels)

        label_map = {label: i for i, label in enumerate(self.args.labels_list)}

        example_row = (
            example,
            label_map,
            self.args.max_seq_length,
            self.tokenizer,
            bool(self.args.model_type in ["xlnet"]),
            self.tokenizer.cls_token,
            2 if self.args.model_type in ["xlnet"] else 0,
            self.tokenizer.sep_token,
            bool(self.args.model_type in ["roberta"]),
            bool(self.args.model_type in ["xlnet"]),
            self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
            4 if self.args.model_type in ["xlnet"] else 0,
            self.pad_token_label_id,
            0,
            True,
        )

        features = convert_example_to_feature(example_row)
        all_input_ids = torch.tensor(features.input_ids, dtype=torch.long)
        all_input_mask = torch.tensor(features.input_mask, dtype=torch.long)
        all_segment_ids = torch.tensor(features.segment_ids, dtype=torch.long)
        all_label_ids = torch.tensor(features.label_ids, dtype=torch.long)
        return (all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    def __len__(self):
        return self.num_entries
