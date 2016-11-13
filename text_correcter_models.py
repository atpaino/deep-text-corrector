from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from collections import defaultdict

import numpy as np
import tensorflow as tf

from data_reader import PAD_ID, GO_ID


class TextCorrecterModel(object):
    """Sequence-to-sequence model used to correct grammatical errors in text.

    NOTE: mostly copied from TensorFlow's seq2seq_model.py; only modifications
    are:
     - the introduction of RMSProp as an optional optimization algorithm
     - the introduction of a "projection bias" that biases decoding towards
       selecting tokens that appeared in the input
    """

    def __init__(self, source_vocab_size, target_vocab_size, buckets, size,
                 num_layers, max_gradient_norm, batch_size, learning_rate,
                 learning_rate_decay_factor, use_lstm=False,
                 num_samples=512, forward_only=False, config=None):
        """Create the model.

        Args:
          source_vocab_size: size of the source vocabulary.
          target_vocab_size: size of the target vocabulary.
          buckets: a list of pairs (I, O), where I specifies maximum input
            length that will be processed in that bucket, and O specifies
            maximum output length. Training instances that have longer than I
            or outputs longer than O will be pushed to the next bucket and
            padded accordingly. We assume that the list is sorted, e.g., [(2,
            4), (8, 16)].
          size: number of units in each layer of the model.
          num_layers: number of layers in the model.
          max_gradient_norm: gradients will be clipped to maximally this norm.
          batch_size: the size of the batches used during training;
            the model construction is independent of batch_size, so it can be
            changed after initialization if this is convenient, e.g.,
            for decoding.
          learning_rate: learning rate to start with.
          learning_rate_decay_factor: decay learning rate by this much when
            needed.
          use_lstm: if true, we use LSTM cells instead of GRU cells.
          num_samples: number of samples for sampled softmax.
          forward_only: if set, we do not construct the backward pass in the
            model.
        """
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        self.config = config

        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in range(buckets[-1][0]):  # Last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="encoder{0}".format(
                                                          i)))
        for i in range(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="decoder{0}".format(
                                                          i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                      name="weight{0}".format(
                                                          i)))

        # Our targets are decoder inputs shifted by one.
        targets = [self.decoder_inputs[i + 1]
                   for i in range(len(self.decoder_inputs) - 1)]
        # If we use sampled softmax, we need an output projection.
        output_projection = None
        softmax_loss_function = None
        # Sampled softmax only makes sense if we sample less than vocabulary
        # size.
        if num_samples > 0 and num_samples < self.target_vocab_size:
            w = tf.get_variable("proj_w", [size, self.target_vocab_size])
            w_t = tf.transpose(w)
            b = tf.get_variable("proj_b", [self.target_vocab_size])

            output_projection = (w, b)

            def sampled_loss(inputs, labels):
                labels = tf.reshape(labels, [-1, 1])
                return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels,
                                                  num_samples,
                                                  self.target_vocab_size)
            softmax_loss_function = sampled_loss

        # Create the internal multi-layer cell for our RNN.
        single_cell = tf.nn.rnn_cell.GRUCell(size)
        if use_lstm:
            single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
        cell = single_cell
        if num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            w, b = output_projection

            # TODO: modify bias here to bias the model towards selecting words
            # present in the input sentence.
            # The alternative is to use the raw attention_decoder in
            # seq2seq_model and specify an alternate loop function.
            input_bias = tf.reduce_sum(
                tf.one_hot(indices=tf.concat(0, encoder_inputs),
                           depth=self.target_vocab_size,
                           on_value=self.config.projection_bias),
                reduction_indices=0)

            projection_bias = b + input_bias

            # Redefined seq2seq to allow for the injection of a special decoding
            # function that leverages an n-gram model over the inputs.
            # TODO: construct function that encloses the mini n-gram model
            # derived from encoder inputs.
            return tf.nn.seq2seq.embedding_attention_seq2seq(
                encoder_inputs, decoder_inputs, cell,
                num_encoder_symbols=source_vocab_size,
                num_decoder_symbols=target_vocab_size,
                embedding_size=size,
                output_projection=(w, projection_bias),
                feed_previous=do_decode)
                # loop_fn_factory=build_decoder_fn_factory(encoder_inputs))

        # Training outputs and losses.
        if forward_only:
            self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets,
                lambda x, y: seq2seq_f(x, y, True),
                softmax_loss_function=softmax_loss_function)
            # If we use output projection, we need to project outputs for
            # decoding.
            if output_projection is not None:
                for b in range(len(buckets)):
                    self.outputs[b] = [
                        tf.matmul(output, output_projection[0]) +
                        output_projection[1]
                        for output in self.outputs[b]]
        else:
            self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets,
                lambda x, y: seq2seq_f(x, y, False),
                softmax_loss_function=softmax_loss_function)

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        if not forward_only:
            self.gradient_norms = []
            self.updates = []
            opt = tf.train.RMSPropOptimizer(0.001) if self.config.use_rms_prop \
                else tf.train.GradientDescentOptimizer(self.learning_rate)
            # opt = tf.train.AdamOptimizer()

            for b in range(len(buckets)):
                gradients = tf.gradients(self.losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(
                    gradients, max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(
                    zip(clipped_gradients, params),
                    global_step=self.global_step))

        self.saver = tf.train.Saver(tf.all_variables())

    def step(self, session, encoder_inputs, decoder_inputs, target_weights,
             bucket_id, forward_only):
        """Run a step of the model feeding the given inputs.

        Args:
          session: tensorflow session to use.
          encoder_inputs: list of numpy int vectors to feed as encoder inputs.
          decoder_inputs: list of numpy int vectors to feed as decoder inputs.
          target_weights: list of numpy float vectors to feed as target weights.
          bucket_id: which bucket of the model to use.
          forward_only: whether to do the backward step or only forward.

        Returns:
          A triple consisting of gradient norm (or None if we did not do
          backward), average perplexity, and the outputs.

        Raises:
          ValueError: if length of encoder_inputs, decoder_inputs, or
            target_weights disagrees with bucket size for the specified
            bucket_id.
        """
        # Check if the sizes match.
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError(
                "Encoder length must be equal to the one in bucket,"
                " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError(
                "Decoder length must be equal to the one in bucket,"
                " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError(
                "Weights length must be equal to the one in bucket,"
                " %d != %d." % (len(target_weights), decoder_size))

        # Input feed: encoder inputs, decoder inputs, target_weights,
        # as provided.
        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                           self.gradient_norms[bucket_id],  # Gradient norm.
                           self.losses[bucket_id]]  # Loss for this batch.
        else:
            output_feed = [self.losses[bucket_id]]  # Loss for this batch.
            for l in range(decoder_size):  # Output logits.
                output_feed.append(self.outputs[bucket_id][l])

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            # Gradient norm, loss, no outputs.
            return outputs[1], outputs[2], None
        else:
            # No gradient norm, loss, outputs.
            return None, outputs[0], outputs[1:]

    def get_batch(self, data, bucket_id):
        """Get a random batch of data from the specified bucket, prepare for
        step.

        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for
        feeding.

        Args:
          data: a tuple of size len(self.buckets) in which each element contains
            lists of pairs of input and output data that we use to create a
            batch.
          bucket_id: integer, which bucket to get the batch for.

        Returns:
          The triple (encoder_inputs, decoder_inputs, target_weights) for
          the constructed batch that has the proper format to call step(...)
          later.
        """
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []

        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for _ in range(self.batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id])

            # Encoder inputs are padded and then reversed.
            encoder_pad = [PAD_ID] * (
                encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([GO_ID] + decoder_input +
                                  [PAD_ID] * decoder_pad_size)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in range(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                          for batch_idx in range(self.batch_size)],
                         dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in range(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in range(self.batch_size)],
                         dtype=np.int32))

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in range(self.batch_size):
                # We set weight to 0 if the corresponding target is a PAD
                # symbol. The corresponding target is decoder_input shifted by 1
                # forward.
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights


class BiasedNGramModel(object):
    """Combines n-gram models over the training corpus and the encoder input."""

    class NGramModel(object):

        def __init__(self, data, fake_backoff_discount=0.0):
            self.fake_backoff_discount = fake_backoff_discount

            unigram_model_counts = defaultdict(int)
            unigram_model_partition = 0
            bigram_model_counts = defaultdict(int)
            bigram_model_partitions = defaultdict(int)

            # Count all the things.
            for tokens in data:
                prev_token = BiasedNGramModel.LEFT_PAD
                for token in tokens:
                    unigram_model_counts[token] += 1
                    unigram_model_partition += 1
                    bigram_model_counts[(prev_token, token)] += 1
                    bigram_model_partitions[prev_token] += 1

            self.unigram_model_counts = unigram_model_counts
            self.unigram_model_partition = unigram_model_partition
            self.bigram_model_counts = bigram_model_counts
            self.bigram_model_partition = bigram_model_partitions

        def prob(self, word, context, k=0):
            # We only go up to bigram models at the moment.
            if len(context) == 0:
                prev_word = BiasedNGramModel.LEFT_PAD
            else:
                prev_word = context[-1]

            # TODO: Katz-backoff and Good-Turing discounting
            if self.bigram_model_counts[(prev_word, word)] > k:
                # Use the bigram model.
                prob = (1.0 * self.bigram_model_counts[(prev_word,
                                                        word)] /
                        self.bigram_model_partition[prev_word])
            elif self.unigram_model_counts[word] > k:
                prob = (
                    self.fake_backoff_discount * self.unigram_model_counts[
                        word] /
                    self.unigram_model_partition)
            else:
                # Fake prob of unseen.
                prob = 0.0001

            return prob

    LEFT_PAD = "LPAD"

    def __init__(self, data_reader, train_path):
        self.corpus_model = BiasedNGramModel.NGramModel(
            data_reader.read_tokens(train_path))

    def prob(self, word, context, original_input):
        """

        :param word:
        :param context:
        :param original_input: list of words
        :return:
        """
        # The idea here is that the unigram prob dist from the input model is
        # still highly relevant, especially in the context of the model having
        # "corrected away" the relevant bigram from the input.
        input_model = BiasedNGramModel.NGramModel([original_input],
                                                  fake_backoff_discount=0.8)

        p_input_model = input_model.prob(word, context)
        p_corpus = self.corpus_model.prob(word, context)

        # Totally made up mixture.
        return 0.8 * p_input_model + 0.2 * p_corpus


#
# def build_decoder_fn_factory(input_tokens):
#     # input_tokens is a tensor of blah blah. whole batch?
#     # Build n-gram model
#
#     def decoder_fn_factory(embedding, output_projection=None,
#                            update_embedding=True):
#         # Similar to _extract_argmax_and_embed, but here we bias things by
#         # the n-gram model.
#         def decoder_fn(prev_output, i, decoder_outputs):
#             # decoder outputs thus far.
#
#             # note: we're still operating on a batch of tensors. every real
#             # operation needs to be a tf op.
#
#
#             return embed_prev, embed_symbol
#
#     return decoder_fn_factory
