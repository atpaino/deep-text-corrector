# Deep Text Correcter

Deep Text Correcter uses [TensorFlow](https://www.tensorflow.org/) to train sequence-to-sequence models that are capable of automatically correcting small grammatical errors in conversational written English (e.g. SMS messages). It does this by taking English text samples that are known to be mostly grammatically correct and randomly introducing a handful of small grammatical errors (e.g. removing articles) to each sentence to produce input-output pairs (where the output is the original sample), which are then used to train a sequence-to-sequence model.

## Motivation
While context-sensitive spell-check systems are able to automatically correct a large number of input errors in instant messaging, email, and SMS messages, they are unable to correct even simple grammatical errors. For example, the message "I'm going to store" would be unaffected by typical autocorrection systems, when the user most likely intendend to write "I'm going to _the_ store".

The goal of this project is to train sequence-to-sequence models that are capable of automatically correcting such errors. Specifically, the models are trained to provide a function mapping a potentially errant input sequence to a sequence with all (small) grammatical errors corrected.

## Correcting Grammatical Errors with Deep Learning
The basic idea behind this project is that we can generate large training data sets for the task of grammar correction by starting with grammatically correct samples and introducing small errors to produce input-output pairs.

Instead of using the most probable decoding according to the seq2seq model, this project takes advantage of the unique structure of the problem in a post-processing step that biases the final decoding toward the original input sequence. We do this by introducing a custom language model named `InputBiasedNGramModel`. This model additionally helps us to resolve out-of-vocabulary words by injecting a strong prior that rare words in the output should have been in the input.

### InputBiasedNGramModel
TODO


## Implementation
This project reuses and slightly extends TensorFlow's [`Seq2SeqModel`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/seq2seq_model.py), which itself implements a sequence-to-sequence model with an attention mechanism as described in https://arxiv.org/pdf/1412.7449v3.pdf. The primary contributions of this project are:

- `data_reader.py`: an abstract class that defines the interface for classes which are capable of reading a source data set and producing input-output pairs, where the input is a grammatically incorrect variant of a source sentence and the output is the original sentence.
- `text_correcter_data_readers.py`: contains a few implementations of `DataReader`, one over the [Penn Treebank dataset](https://www.google.com/url?q=http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz&usg=AFQjCNG0IP5OHusdIAdJIrrem-HMck9AzA) and one over the [Cornell Movie-Dialogs Corpus](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html).
- `text_correcter_models.py`: contains a slightly modified version of `Seq2SeqModel` in addition to a custom language model, `InputBiasedNGramModel`, which is used in post-processing of the decoding derived from the sequence-to-sequence model. See [InputBiasedNGramModel](https://github.com/atpaino/deep-text-correcter#inputbiasedngrammodel) below for details on how these two models are combined to produce the final corrected sequence.
- `correct_text.py`: a collection of helper functions that together allow for the training of a model and the usage of it to decode errant input sequences (at test time). This also defines a main method, and can be invoked from the command line. It was largely derived from TensorFlow's [`translate.py`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/translate.py).
- `TextCorrecter.ipynb`: an IPython notebook which ties together all of the above pieces to allow for the training and evaluation of the model in an interactive fashion.

### Example usage

TODO

## Experimental Results

TODO
