"""Program used to create, train, and evaluate "text correcting" models.

Defines utilities that allow for:
1. Creating a TextCorrectorModel
2. Training a TextCorrectorModel using a given DataReader (i.e. a data source)
3. Decoding predictions from a trained TextCorrectorModel

The program is best run from the command line using the flags defined below or
through an IPython notebook.

Note: this has been mostly copied from Tensorflow's translate.py demo
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import tensorflow as tf

from data_reader import EOS_ID
from text_corrector_data_readers import MovieDialogReader, PTBDataReader

from text_corrector_models import TextCorrectorModel

tf.app.flags.DEFINE_string("config", "TestConfig", "Name of config to use.")
tf.app.flags.DEFINE_string("data_reader_type", "MovieDialogReader",
                           "Type of data reader to use.")
tf.app.flags.DEFINE_string("train_path", "train", "Training data path.")
tf.app.flags.DEFINE_string("val_path", "val", "Validation data path.")
tf.app.flags.DEFINE_string("test_path", "test", "Testing data path.")
tf.app.flags.DEFINE_string("model_path", "model", "Path where the model is "
                                                  "saved.")
tf.app.flags.DEFINE_boolean("decode", False, "Whether we should decode data "
                                             "at test_path. The default is to "
                                             "train a model and save it at "
                                             "model_path.")

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

    use_lstm = False
    use_rms_prop = False


class DefaultPTBConfig():
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

    use_lstm = False
    use_rms_prop = False


class DefaultMovieDialogConfig():
    buckets = [(10, 10), (15, 15), (20, 20), (40, 40)]

    steps_per_checkpoint = 100
    max_steps = 20000

    # The OOV resolution scheme used in decode() allows us to use a much smaller
    # vocabulary.
    max_vocabulary_size = 2000

    size = 512
    num_layers = 4
    max_gradient_norm = 5.0
    batch_size = 64
    learning_rate = 0.5
    learning_rate_decay_factor = 0.99

    use_lstm = True
    use_rms_prop = False

    projection_bias = 0.0


def create_model(session, forward_only, model_path, config=TestConfig()):
    """Create translation model and initialize or load parameters in session."""
    model = TextCorrectorModel(
        config.max_vocabulary_size,
        config.max_vocabulary_size,
        config.buckets,
        config.size,
        config.num_layers,
        config.max_gradient_norm,
        config.batch_size,
        config.learning_rate,
        config.learning_rate_decay_factor,
        use_lstm=config.use_lstm,
        forward_only=forward_only,
        config=config)
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

        # A bucket scale is a list of increasing numbers from 0 to 1 that
        # we'll use to select a bucket. Length of [scale[i], scale[i+1]] is
        # proportional to the size if i-th training bucket, as used later.
        train_buckets_scale = [
            sum(train_bucket_sizes[:i + 1]) / train_total_size
            for i in range(len(train_bucket_sizes))]

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while current_step < config.max_steps:
            # Choose a bucket according to data distribution. We pick a random
            # number in [0, 1] and use the corresponding interval in
            # train_buckets_scale.
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

            # Once in a while, we save checkpoint, print statistics, and run
            # evals.
            if current_step % config.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(float(loss)) if loss < 300 else float(
                    "inf")
                print("global step %d learning rate %.4f step-time %.2f "
                      "perplexity %.2f" % (
                          model.global_step.eval(), model.learning_rate.eval(),
                          step_time, perplexity))
                # Decrease learning rate if no improvement was seen over last
                #  3 times.
                if len(previous_losses) > 2 and loss > max(
                        previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(model_path, "translate.ckpt")
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


def get_corrective_tokens(data_reader, train_path):
    # TODO: this should be part of the model, learned during training
    corrective_tokens = set()
    for source_tokens, target_tokens in data_reader.read_samples_by_string(
            train_path):
        corrective_tokens.update(set(target_tokens) - set(source_tokens))
    return corrective_tokens


def decode(sess, model, data_reader, data_to_decode, corrective_tokens=set(),
           verbose=True):
    """

    :param sess:
    :param model:
    :param data_reader:
    :param data_to_decode: an iterable of token lists representing the input
        data we want to decode
    :param corrective_tokens
    :param verbose:
    :return:
    """
    model.batch_size = 1

    corrective_tokens_mask = np.zeros(model.target_vocab_size)
    corrective_tokens_mask[EOS_ID] = 1.0
    for token in corrective_tokens:
        corrective_tokens_mask[data_reader.convert_token_to_id(token)] = 1.0

    for tokens in data_to_decode:
        token_ids = [data_reader.convert_token_to_id(token) for token in tokens]

        # Which bucket does it belong to?
        matching_buckets = [b for b in range(len(model.buckets))
                            if model.buckets[b][0] > len(token_ids)]
        if not matching_buckets:
            # The input string has more tokens than the largest bucket, so we
            # have to skip it.
            continue

        bucket_id = min(matching_buckets)

        # Get a 1-element batch to feed the sentence to the model.
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
            {bucket_id: [(token_ids, [])]}, bucket_id)

        # Get output logits for the sentence.
        _, _, output_logits = model.step(
            sess, encoder_inputs, decoder_inputs, target_weights, bucket_id,
            True, corrective_tokens=corrective_tokens_mask)

        oov_input_tokens = [token for token in tokens if
                            data_reader.is_unknown_token(token)]

        outputs = []
        next_oov_token_idx = 0

        for logit in output_logits:

            max_likelihood_token_id = int(np.argmax(logit, axis=1))
            # First check to see if this logit most likely points to the EOS
            # identifier.
            if max_likelihood_token_id == EOS_ID:
                break

            token = data_reader.convert_id_to_token(max_likelihood_token_id)
            if data_reader.is_unknown_token(token):
                # Replace the "unknown" token with the most probable OOV
                # token from the input.
                if next_oov_token_idx < len(oov_input_tokens):
                    # If we still have OOV input tokens available,
                    # pick the next available one.
                    token = oov_input_tokens[next_oov_token_idx]
                    # Advance to the next OOV input token.
                    next_oov_token_idx += 1
                else:
                    # If we've already used all OOV input tokens,
                    # then we just leave the token as "UNK"
                    pass

            outputs.append(token)

        if verbose:
            decoded_sentence = " ".join(outputs)

            print("Input: {}".format(" ".join(tokens)))
            print("Output: {}\n".format(decoded_sentence))

        yield outputs


def decode_sentence(sess, model, data_reader, sentence, corrective_tokens=set(),
                    verbose=True):
    """Used with InteractiveSession in an IPython notebook."""
    return next(decode(sess, model, data_reader, [sentence.split()],
                       corrective_tokens=corrective_tokens, verbose=verbose))


def evaluate_accuracy(sess, model, data_reader, corrective_tokens, test_path,
                      max_samples=None):
    """Evaluates the accuracy and BLEU score of the given model."""

    import nltk  # Loading here to avoid having to bundle it in lambda.

    # Build a collection of "baseline" and model-based hypotheses, where the
    # baseline is just the (potentially errant) source sequence.
    baseline_hypotheses = defaultdict(list)  # The model's input
    model_hypotheses = defaultdict(list)  # The actual model's predictions
    targets = defaultdict(list)  # Groundtruth

    errors = []

    n_samples_by_bucket = defaultdict(int)
    n_correct_model_by_bucket = defaultdict(int)
    n_correct_baseline_by_bucket = defaultdict(int)
    n_samples = 0

    # Evaluate the model against all samples in the test data set.
    for source, target in data_reader.read_samples_by_string(test_path):

        matching_buckets = [i for i, bucket in enumerate(model.buckets) if
                            len(source) < bucket[0]]
        if not matching_buckets:
            continue

        bucket_id = matching_buckets[0]

        decoding = next(
            decode(sess, model, data_reader, [source],
                   corrective_tokens=corrective_tokens, verbose=False))
        model_hypotheses[bucket_id].append(decoding)
        if decoding == target:
            n_correct_model_by_bucket[bucket_id] += 1
        else:
            errors.append((decoding, target))

        baseline_hypotheses[bucket_id].append(source)
        if source == target:
            n_correct_baseline_by_bucket[bucket_id] += 1

        # nltk.corpus_bleu expects a list of one or more reference
        # tranlsations per sample, so we wrap the target list in another list
        # here.
        targets[bucket_id].append([target])

        n_samples_by_bucket[bucket_id] += 1
        n_samples += 1

        if max_samples is not None and n_samples > max_samples:
            break

    # Measure the corpus BLEU score and accuracy for the model and baseline
    # across all buckets.
    for bucket_id in targets.keys():
        baseline_bleu_score = nltk.translate.bleu_score.corpus_bleu(
            targets[bucket_id], baseline_hypotheses[bucket_id])
        model_bleu_score = nltk.translate.bleu_score.corpus_bleu(
            targets[bucket_id], model_hypotheses[bucket_id])
        print("Bucket {}: {}".format(bucket_id, model.buckets[bucket_id]))
        print("\tBaseline BLEU = {:.4f}\n\tModel BLEU = {:.4f}".format(
            baseline_bleu_score, model_bleu_score))
        print("\tBaseline Accuracy: {:.4f}".format(
            1.0 * n_correct_baseline_by_bucket[bucket_id] /
            n_samples_by_bucket[bucket_id]))
        print("\tModel Accuracy: {:.4f}".format(
            1.0 * n_correct_model_by_bucket[bucket_id] /
            n_samples_by_bucket[bucket_id]))

    return errors


def main(_):
    # Determine which config we should use.
    if FLAGS.config == "TestConfig":
        config = TestConfig()
    elif FLAGS.config == "DefaultMovieDialogConfig":
        config = DefaultMovieDialogConfig()
    elif FLAGS.config == "DefaultPTBConfig":
        config = DefaultPTBConfig()
    else:
        raise ValueError("config argument not recognized; must be one of: "
                         "TestConfig, DefaultPTBConfig, "
                         "DefaultMovieDialogConfig")

    # Determine which kind of DataReader we want to use.
    if FLAGS.data_reader_type == "MovieDialogReader":
        data_reader = MovieDialogReader(config, FLAGS.train_path)
    elif FLAGS.data_reader_type == "PTBDataReader":
        data_reader = PTBDataReader(config, FLAGS.train_path)
    else:
        raise ValueError("data_reader_type argument not recognized; must be "
                         "one of: MovieDialogReader, PTBDataReader")

    if FLAGS.decode:
        # Decode test sentences.
        with tf.Session() as session:
            model = create_model(session, True, FLAGS.model_path, config=config)
            print("Loaded model. Beginning decoding.")
            decodings = decode(session, model=model, data_reader=data_reader,
                               data_to_decode=data_reader.read_tokens(
                                   FLAGS.test_path), verbose=False)
            # Write the decoded tokens to stdout.
            for tokens in decodings:
                print(" ".join(tokens))
                sys.stdout.flush()
    else:
        print("Training model.")
        train(data_reader, FLAGS.train_path, FLAGS.val_path, FLAGS.model_path)


if __name__ == "__main__":
    tf.app.run()
