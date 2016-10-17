from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf

from tensorflow.models.rnn.translate import seq2seq_model


tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("en_vocab_size", 40000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("fr_vocab_size", 40000, "French vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS


class TestConfig():
    # We use a number of buckets and pad to the closest one for efficiency.
    buckets = [(10, 10), (15, 15), (20, 20), (40, 40)]

    steps_per_checkpoint = 20
    max_steps = 100

    max_vocabulary_size = 10000

    size = 128
    num_layers = 1
    max_gradient_norm = 5.0
    batch_size = 64
    learning_rate = 0.5
    learning_rate_decay_factor = 0.99


class DefaultConfig():
    buckets = [(10, 10), (15, 15), (20, 20), (40, 40)]

    steps_per_checkpoint = 100
    max_steps = 20000

    max_vocabulary_size = 10000

    size = 512
    num_layers = 2
    max_gradient_norm = 5.0
    batch_size = 64
    learning_rate = 0.5
    learning_rate_decay_factor = 0.99


class DefaultMovieDialogConfig():
    buckets = [(10, 10), (15, 15), (20, 20), (40, 40)]

    steps_per_checkpoint = 100
    max_steps = 20000

    max_vocabulary_size = 40000

    size = 512
    num_layers = 2
    max_gradient_norm = 5.0
    batch_size = 64
    learning_rate = 0.5
    learning_rate_decay_factor = 0.99


def create_model(session, forward_only, model_path, config=TestConfig()):
    """Create translation model and initialize or load parameters in session."""
    model = seq2seq_model.Seq2SeqModel(
        config.max_vocabulary_size,
        config.max_vocabulary_size,
        config.buckets,
        config.size,
        config.num_layers,
        config.max_gradient_norm,
        config.batch_size,
        config.learning_rate,
        config.learning_rate_decay_factor,
        forward_only=forward_only)
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    return model


def train(data_reader, train_path, test_path, model_path):
    """"""
    print(
        "Reading data; train = {}, test = {}".format(train_path, test_path))
    config = data_reader.config
    train_data = data_reader.build_dataset(train_path)
    test_data = data_reader.build_dataset(test_path)

    with tf.Session() as sess:
        # Create model.
        print(
            "Creating %d layers of %d units." % (
                config.num_layers, config.size))
        model = create_model(sess, False, model_path, config=config)

        # Read data into buckets and compute their sizes.
        train_bucket_sizes = [len(train_data[b]) for b in
                              range(len(config.buckets))]
        print("Training bucket sizes: {}".format(train_bucket_sizes))
        train_total_size = float(sum(train_bucket_sizes))
        print("Total train size: {}".format(train_total_size))

        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        train_buckets_scale = [
            sum(train_bucket_sizes[:i + 1]) / train_total_size
            for i in range(len(train_bucket_sizes))]

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while current_step < config.max_steps:
            # Choose a bucket according to data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                train_data, bucket_id)
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, False)
            step_time += (time.time() - start_time) / config \
                .steps_per_checkpoint
            loss += step_loss / config.steps_per_checkpoint
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % config.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(float(loss)) if loss < 300 else float(
                    "inf")
                print("global step %d learning rate %.4f step-time %.2f "
                      "perplexity %.2f" % (
                          model.global_step.eval(), model.learning_rate.eval(),
                          step_time, perplexity))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(
                        previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(model_path,
                                               "translate.ckpt")
                model.saver.save(sess, checkpoint_path,
                                 global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                # Run evals on development set and print their perplexity.
                for bucket_id in range(len(config.buckets)):
                    if len(test_data[bucket_id]) == 0:
                        print("  eval: empty bucket %d" % (bucket_id))
                        continue
                    encoder_inputs, decoder_inputs, target_weights = \
                        model.get_batch(test_data, bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs,
                                                 decoder_inputs,
                                                 target_weights, bucket_id,
                                                 True)
                    eval_ppx = math.exp(
                        float(eval_loss)) if eval_loss < 300 else float(
                        "inf")
                    print("  eval: bucket %d perplexity %.2f" % (
                        bucket_id, eval_ppx))
                sys.stdout.flush()


def decode(sess, model, data_reader, sentences, verbose=True):
    model.batch_size = 1
    id_to_word = {id: word for word, id in data_reader.word_to_id.items()}

    decoded_sentences = []

    for sentence in sentences:
        word_ids = data_reader.sentence_to_word_ids(sentence)

        # Which bucket does it belong to?
        matching_buckets = [b for b in range(len(model.buckets))
                            if model.buckets[b][0] > len(word_ids)]
        if not matching_buckets:
            continue
        bucket_id = min(matching_buckets)

        # Get a 1-element batch to feed the sentence to the model.
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
            {bucket_id: [(word_ids, [])]}, bucket_id)

        # Get output logits for the sentence.
        _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, True)

        # This is a greedy decoder - outputs are just argmaxes of output_logits.
        # TODO(atpaino) use beam search?
        outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

        # If there is an EOS symbol in outputs, cut them at that point.
        if data_reader.eos_id() in outputs:
            outputs = outputs[:outputs.index(data_reader.eos_id())]

        decoding = [id_to_word[word_id] if word_id in id_to_word
                    else data_reader.unknown_token()
                    for word_id in outputs]

        if verbose:

            decoded_sentence = " ".join(decoding)

            print("Input: {}".format(sentence))
            print("Output: {}\n".format(decoded_sentence))

        decoded_sentences.append(decoding)

    return decoded_sentences


def main(_):
    pass
    # if FLAGS.self_test:
    #     self_test()
    # elif FLAGS.decode:
    #     decode()
    # else:
    #     train()


if __name__ == "__main__":
    tf.app.run()
