from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter

# Define constants associated with the usual special-case tokens.
PAD_ID = 0
GO_ID = 1
EOS_ID = 2

PAD_TOKEN = "PAD"
EOS_TOKEN = "EOS"
GO_TOKEN = "GO"


class DataReader(object):

    def __init__(self, config, train_path=None, token_to_id=None,
                 special_tokens=(), dataset_copies=1):
        self.config = config
        self.dataset_copies = dataset_copies

        # Construct vocabulary.
        max_vocabulary_size = self.config.max_vocabulary_size

        if train_path is None:
            self.token_to_id = token_to_id
        else:
            token_counts = Counter()

            for tokens in self.read_tokens(train_path):
                token_counts.update(tokens)

            self.token_counts = token_counts

            # Get to max_vocab_size words
            count_pairs = sorted(token_counts.items(),
                                 key=lambda x: (-x[1], x[0]))
            vocabulary, _ = list(zip(*count_pairs))
            vocabulary = list(vocabulary)
            # Insert the special tokens at the beginning.
            vocabulary[0:0] = special_tokens
            full_token_and_id = zip(vocabulary, range(len(vocabulary)))
            self.full_token_to_id = dict(full_token_and_id)
            self.token_to_id = dict(full_token_and_id[:max_vocabulary_size])

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def read_tokens(self, path):
        """
        Reads the given file line by line and yields the list of tokens present
        in each line.

        :param path:
        :return:
        """
        raise NotImplementedError("Must implement read_tokens")

    def read_samples_by_string(self, path):
        """
        Reads the given file line by line and yields the word-form of each
        derived sample.

        :param path:
        :return:
        """
        raise NotImplementedError("Must implement read_word_samples")

    def unknown_token(self):
        raise NotImplementedError("Must implement read_word_samples")

    def convert_token_to_id(self, token):
        """

        :param token:
        :return:
        """
        token_with_id = token if token in self.token_to_id else \
            self.unknown_token()
        return self.token_to_id[token_with_id]

    def convert_id_to_token(self, token_id):
        return self.id_to_token[token_id]

    def is_unknown_token(self, token):
        """
        True if the given token is out of the vocabulary used or if it is the
        actual unknown token.

        :param token:
        :return:
        """
        return token not in self.token_to_id or token == self.unknown_token()

    def sentence_to_token_ids(self, sentence):
        """
        Converts a whitespace-delimited sentence into a list of word ids.
        """
        return [self.convert_token_to_id(word) for word in sentence.split()]

    def token_ids_to_tokens(self, word_ids):
        """
        Converts a list of word ids to a list of their corresponding words.
        """
        return [self.convert_id_to_token(word) for word in word_ids]

    def read_samples(self, path):
        """

        :param path:
        :return:
        """
        for source_words, target_words in self.read_samples_by_string(path):
            source = [self.convert_token_to_id(word) for word in source_words]
            target = [self.convert_token_to_id(word) for word in target_words]
            target.append(EOS_ID)

            yield source, target

    def build_dataset(self, path):
        dataset = [[] for _ in self.config.buckets]

        # Make multiple copies of the dataset so that we synthesize different
        # dropouts.
        for _ in range(self.dataset_copies):
            for source, target in self.read_samples(path):
                for bucket_id, (source_size, target_size) in enumerate(
                        self.config.buckets):
                    if len(source) < source_size and len(
                            target) < target_size:
                        dataset[bucket_id].append([source, target])
                        break

        return dataset

