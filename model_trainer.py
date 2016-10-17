from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf


class Model(object):

    def __init__(self, is_training, config):
        self._lr_update = None
        self._new_lr = None
        self._input_data = None
        self._targets = None
        self._cost = None
        self._lr = None
        self._train_op = None
        self._predictions = None

    def data_to_feed_dict(self, data):
        raise NotImplementedError("Must implement data_to_feed_dict")

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def cost(self):
        return self._cost

    @property
    def predictions(self):
        return self._predictions

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


class DataReader(object):

    def __init__(self, config):
        self.config = config

    def _read(self, path, downsample=True):
        raise NotImplementedError("Must implement _read")

    def iterator(self, path, batch_size, downsample=True):
        batch = None
        for entry in self._read(path, downsample=downsample):
            if batch is None:
                batch = [[] for _ in entry]

            for i, e in enumerate(entry):
                batch[i].append(e)

            if len(batch[0]) == batch_size:
                yield tuple(np.asarray(b) for b in batch)

                batch = [[] for _ in entry]


def run_epoch(session, model, data_reader, data_path, op, downsample=True,
              verbose=False):
    start_time = time.time()
    costs = 0.0
    n_batches = 0
    outputs = []

    for i_batch, data in enumerate(
            data_reader.iterator(data_path, model.batch_size,
                                 downsample=downsample)):
        cost, op_output = session.run(
            [model.cost, op],
            model.data_to_feed_dict(data))

        costs += cost
        if op_output is not None:
            outputs.extend(op_output)
        n_batches += 1

        if verbose and i_batch % (10000 / model.batch_size) == 100:
            print("Batch number: {} Cost: {:.3f} Speed: {:.3f} records per "
                  "second".format(
                i_batch, costs / n_batches, i_batch * model.batch_size / (
                    time.time() - start_time)))

    return costs / n_batches, outputs


def train_model(session, model_constructor, model_path, config, data_reader,
                train_path, test_path, graph=tf.get_default_graph()):

    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with graph.as_default():
        # Declare models
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = model_constructor(is_training=True, config=config)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            m_test = model_constructor(is_training=False, config=config)

        #
        saver = tf.train.Saver()

        tf.initialize_all_variables().run()

        train_costs = []
        for i in range(config.num_epochs):
            lr_decay = config.lr_decay ** max(
                i - config.epochs_with_max_learning_rate, 0.0)
            m.assign_lr(session, config.learning_rate * lr_decay)

            # Run an epoch of training
            print("Epoch: {} Learning rate: {:.3f}".format(i + 1,
                                                           session.run(m.lr)))
            train_cost, _ = run_epoch(session, m, data_reader, train_path,
                                      m.train_op, verbose=True)
            print("Train mean cost: {:.3f}".format(train_cost))
            train_costs.append(train_cost)

            saver.save(session, model_path, global_step=i)

        test_cost, predictions = run_epoch(session, m_test, data_reader,
                                           test_path, m_test.predictions,
                                           downsample=False,
                                           verbose=True)
        print("Test mean cost: {:.3f}".format(test_cost))

        return predictions, train_costs


def restore_model(session, model_constructor, model_path, config,
                  graph=tf.get_default_graph()):

    with graph.as_default():
        with tf.variable_scope("model"):
            m_test = model_constructor(is_training=False, config=config)

        saver = tf.train.Saver()
        saver.restore(session, model_path)

        return m_test

if __name__ == "__main__":
    tf.app.run()
