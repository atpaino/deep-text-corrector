from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import random
from collections import Counter

import numpy as np
import tensorflow as tf

from tensorflow.models.rnn.translate import data_utils

from model_trainer import DataReader


class PTBDataReader(DataReader):
    UNKNOWN_TOKEN = "<unk>" # already defined in the source data
    PAD_TOKEN = "PAD"
    EOS_TOKEN = "EOS"
    GO_TOKEN = "GO"

    EOS_ID = data_utils.EOS_ID
    GO_ID = data_utils.GO_ID
    PAD_ID = data_utils.PAD_ID

    DROPOUT_WORDS = {"a", "an", "the"}
    DROPOUT_PROB = 0.25

    REPLACEMENTS = {"there": "their", "their": "there"}
    REPLACEMENT_PROB = 0.25

    def __init__(self, config, train_path):
        super(PTBDataReader, self).__init__(config)

        max_vocabulary_size = self.config.max_vocabulary_size

        word_counts = Counter()

        for line in self._read_raw(train_path):
            word_counts.update(line.split())

        self.word_counts = word_counts

        # Get to max_vocab_size words
        count_pairs = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))
        vocabulary, _ = list(zip(*count_pairs))
        vocabulary = list(vocabulary)[:max_vocabulary_size-3]
        # Replace the first 3 tokens with [PAD, GO, EOS]
        vocabulary.insert(0, data_utils._EOS)
        vocabulary.insert(0, data_utils._GO)
        vocabulary.insert(0, data_utils._PAD)
        self.word_to_id = dict(zip(vocabulary, range(len(vocabulary))))

        self.UNKNOWN_ID = self.word_to_id[PTBDataReader.UNKNOWN_TOKEN]
        # self.PAD_ID = self.word_to_id[PTBDataReader.PAD_TOKEN]
        # self.EOS_ID = self.word_to_id[PTBDataReader.EOS_TOKEN]
        # self.GO_ID = self.word_to_id[PTBDataReader.GO_TOKEN]

    def _read(self, path, downsample=True):

        for line in self._read_raw(path):
            source = []
            target = []

            for word in line.split():
                word_id = self.convert_word_to_id(word)

                target.append(word_id)

                # Randomly dropout some words from the input.
                dropout_word = (word in PTBDataReader.DROPOUT_WORDS and
                                random.random() < PTBDataReader.DROPOUT_PROB)
                replace_word = (word in PTBDataReader.REPLACEMENTS and
                                random.random() <
                                PTBDataReader.REPLACEMENT_PROB)

                if replace_word:
                    source.append(self.convert_word_to_id(
                        PTBDataReader.REPLACEMENTS[word]))
                elif not dropout_word:
                    source.append(word_id)

            target.append(data_utils.EOS_ID)

            yield source, target

    def convert_word_to_id(self, word):
        return self.word_to_id[
            word] if word in self.word_to_id else self.UNKNOWN_ID

    def sentence_to_word_ids(self, sentence):
        word_to_id = self.word_to_id
        return [word_to_id[word] if word in word_to_id else self.UNKNOWN_ID for
                word in sentence.split()]

    def eos_id(self):
        return PTBDataReader.EOS_ID

    def unknown_token(self):
        return PTBDataReader.UNKNOWN_TOKEN

    def build_dataset(self, path):
        dataset = [[] for _ in self.config.buckets]

        for source, target in self._read(path):
            for bucket_id, (source_size, target_size) in enumerate(
                    self.config.buckets):
                if len(source) < source_size and len(
                        target) < target_size:
                    dataset[bucket_id].append([source, target])
                    break

        return dataset

    def batchify(self):
        pass
        # # Pad input
        # encoder_pad = [self.PAD_ID] * (
        #     self.config.max_sequence_length - len(source))
        # encoder_input = list(reversed(source + encoder_pad))
        #
        # # Decoder inputs get an extra "GO" symbol, and are padded then.
        # decoder_pad_size = self.config.max_sequence_length - len(target) - 1
        # decoder_input = (
        #     [self.GO_ID] + target + [self.PAD_ID] * decoder_pad_size)
        #
        # target_weights = np.concatenate(
        #     (np.ones(len(target), dtype=np.float32) + np.zeros(
        #         self.config.max_sequence_length - len(target),
        #         dtype=np.float32)))


    def _read_raw(self, path):
        with open(path, "r") as f:
            for line in f:
                yield line.rstrip().lstrip()


class MovieDialogReader(DataReader):
    UNKNOWN_TOKEN = "UNK"

    EOS_ID = data_utils.EOS_ID
    GO_ID = data_utils.GO_ID
    PAD_ID = data_utils.PAD_ID

    DROPOUT_WORDS = {"a", "an", "the", "'ll", "'s", "'m", "'ve"}  # Add "to"

    REPLACEMENTS = {"there": "their", "their": "there", "then": "than",
                    "than": "then"}
    # Add: "be":"to"

    def __init__(self, config, train_path, dropout_prob=0.25,
                 replacement_prob=0.25,
                 dataset_copies=2):
        super(MovieDialogReader, self).__init__(config)

        self.dropout_prob = dropout_prob
        self.replacement_prob = replacement_prob
        self.dataset_copies = dataset_copies

        max_vocabulary_size = self.config.max_vocabulary_size

        word_counts = Counter()

        for line in self._read_raw(train_path):
            word_counts.update(line.split())

        self.word_counts = word_counts

        # Get to max_vocab_size words
        count_pairs = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))
        vocabulary, _ = list(zip(*count_pairs))
        vocabulary = list(vocabulary)
        # Insert the tokens [PAD, GO, EOS, UNK]
        vocabulary[0:0] = [data_utils._PAD, data_utils._GO, data_utils._EOS,
                           MovieDialogReader.UNKNOWN_TOKEN]
        self.word_to_id = dict(
            zip(vocabulary[:max_vocabulary_size], range(max_vocabulary_size)))

        self.UNKNOWN_ID = self.word_to_id[MovieDialogReader.UNKNOWN_TOKEN]

    def read_words(self, path):
        for line in self._read_raw(path):
            source = []
            target = []

            for word in line.split():

                target.append(word)

                # Randomly dropout some words from the input.
                dropout_word = (word in MovieDialogReader.DROPOUT_WORDS and
                                random.random() < self.dropout_prob)
                replace_word = (word in MovieDialogReader.REPLACEMENTS and
                                random.random() < self.replacement_prob)

                if replace_word:
                    source.append(MovieDialogReader.REPLACEMENTS[word])
                elif not dropout_word:
                    source.append(word)

            yield source, target

    def _read(self, path, downsample=True):

        for source_words, target_words in self.read_words(path):
            source = [self.convert_word_to_id(word) for word in source_words]
            target = [self.convert_word_to_id(word) for word in target_words]
            target.append(data_utils.EOS_ID)

            yield source, target

    def convert_word_to_id(self, word):
        return self.word_to_id[
            word] if word in self.word_to_id else self.UNKNOWN_ID

    def sentence_to_word_ids(self, sentence):
        return [self.convert_word_to_id(word) for word in sentence.split()]

    def eos_id(self):
        return MovieDialogReader.EOS_ID

    def unknown_token(self):
        return MovieDialogReader.UNKNOWN_TOKEN

    def build_dataset(self, path):
        dataset = [[] for _ in self.config.buckets]

        # Make multiple copies of the dataset so that we synthesize different
        # dropouts.
        for _ in range(self.dataset_copies):
            for source, target in self._read(path):
                for bucket_id, (source_size, target_size) in enumerate(
                        self.config.buckets):
                    if len(source) < source_size and len(
                            target) < target_size:
                        dataset[bucket_id].append([source, target])
                        break

        return dataset

    def batchify(self):
        pass

    def _read_raw(self, path):
        with open(path, "r") as f:
            for line in f:
                yield line.lower().rstrip().lstrip()
