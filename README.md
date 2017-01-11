# Deep Text Corrector

Deep Text Corrector uses [TensorFlow](https://www.tensorflow.org/) to train sequence-to-sequence models that are capable of automatically correcting small grammatical errors in conversational written English (e.g. SMS messages). 
It does this by taking English text samples that are known to be mostly grammatically correct and randomly introducing a handful of small grammatical errors (e.g. removing articles) to each sentence to produce input-output pairs (where the output is the original sample), which are then used to train a sequence-to-sequence model.

See [this blog post](http://atpaino.com/2017/01/03/deep-text-correcter.html) for a more thorough write-up of this work.

## Motivation
While context-sensitive spell-check systems are able to automatically correct a large number of input errors in instant messaging, email, and SMS messages, they are unable to correct even simple grammatical errors. 
For example, the message "I'm going to store" would be unaffected by typical autocorrection systems, when the user most likely intendend to write "I'm going to _the_ store". 
These kinds of simple grammatical mistakes are common in so-called "learner English", and constructing systems capable of detecting and correcting these mistakes has been the subect of multiple [CoNLL shared tasks](http://www.aclweb.org/anthology/W14-1701.pdf).

The goal of this project is to train sequence-to-sequence models that are capable of automatically correcting such errors. 
Specifically, the models are trained to provide a function mapping a potentially errant input sequence to a sequence with all (small) grammatical errors corrected.
Given these models, it would be possible to construct tools to help correct these simple errors in written communications, such as emails, instant messaging, etc.

## Correcting Grammatical Errors with Deep Learning
The basic idea behind this project is that we can generate large training datasets for the task of grammar correction by starting with grammatically correct samples and introducing small errors to produce input-output pairs, which can then be used to train a sequence-to-sequence models.
The details of how we construct these datasets, train models using them, and produce predictions for this task are described below.

### Datasets
To create a dataset for Deep Text Corrector models, we start with a large collection of mostly grammatically correct samples of conversational written English. 
The primary dataset considered in this project is the [Cornell Movie-Dialogs Corpus](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html), which contains over 300k lines from movie scripts.
This was the largest collection of conversational written English I could find that was mostly grammatically correct. 

Given a sample of text like this, the next step is to generate input-output pairs to be used during training. 
This is done by:
1. Drawing a sample sentence from the dataset.
2. Setting the input sequence to this sentence after randomly applying certain perturbations.
3. Setting the output sequence to the unperturbed sentence.

where the perturbations applied in step (2) are intended to introduce small grammatical errors which we would like the model to learn to correct. 
Thus far, these perturbations are limited to the:
- subtraction of articles (a, an, the)
- subtraction of the second part of a verb contraction (e.g. "'ve", "'ll", "'s", "'m")
- replacement of a few common homophones with one of their counterparts (e.g. replacing "their" with "there", "then" with "than")

The rates with which these perturbations are introduced are loosely based on figures taken from the [CoNLL 2014 Shared Task on Grammatical Error Correction](http://www.aclweb.org/anthology/W14-1701.pdf). 
In this project, each perturbation is applied in 25% of cases where it could potentially be applied.

### Training
To artificially increase the dataset when training a sequence model, we perform the sampling strategy described above multiple times to arrive at 2-3x the number of input-output pairs. 
Given this augmented dataset, training proceeds in a very similar manner to [TensorFlow's sequence-to-sequence tutorial](https://www.tensorflow.org/tutorials/seq2seq/). 
That is, we train a sequence-to-sequence model using LSTM encoders and decoders with an attention mechanism as described in [Bahdanau et al., 2014](http://arxiv.org/abs/1409.0473) using stochastic gradient descent. 

### Decoding

Instead of using the most probable decoding according to the seq2seq model, this project takes advantage of the unique structure of the problem to impose the prior that all tokens in a decoded sequence should either exist in the input sequence or belong to a set of "corrective" tokens. 
The "corrective" token set is constructed during training and contains all tokens seen in the target, but not the source, for at least one sample in the training set. 
The intuition here is that the errors seen during training involve the misuse of a relatively small vocabulary of common words (e.g. "the", "an", "their") and that the model should only be allowed to perform corrections in this domain.

This prior is carried out through a modification to the seq2seq model's decoding loop in addition to a post-processing step that resolves out-of-vocabulary (OOV) tokens:

**Biased Decoding**

To restrict the decoding such that it only ever chooses tokens from the input sequence or corrective token set, this project applies a binary mask to the model's logits prior to extracting the prediction to be fed into the next time step. 
This mask is constructed such that `mask[i] == 1.0 if (i in input or corrective_tokens) else 0.0`. 
Since this mask is applited to the result of a softmax transormation (which guarantees all outputs are non-negative), we can be sure that only input or corrective tokens are ever selected.

Note that this logic is not used during training, as this would only serve to eliminate potentially useful signal from the model.

**Handling OOV Tokens**

Since the decoding bias described above is applied within the truncated vocabulary used by the model, we will still see the unknown token in its output for any OOV tokens. 
The more generic problem of resolving these OOV tokens is non-trivial (e.g. see [Addressing the Rare Word Problem in NMT](https://arxiv.org/pdf/1410.8206v4.pdf)), but in this project we can again take advantage of its unique structure to create a fairly straightforward OOV token resolution scheme. 
That is, if we assume the sequence of OOV tokens in the input is equal to the sequence of OOV tokens in the output sequence, then we can trivially assign the appropriate token to each "unknown" token encountered int he decoding. 
Empirically, and intuitively, this appears to be an appropriate assumption, as the relatively simple class of errors these models are being trained to address should never include mistakes that warrant the insertion or removal of a rare token.

## Experiments and Results

Below are some anecdotal and aggregate results from experiments using the Deep Text Corrector model with the [Cornell Movie-Dialogs Corpus](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html). 
The dataset consists of 304,713 lines from movie scripts, of which 243,768 lines were used to train the model and 30,474 lines each were used for the validation and testing sets. 
The sets were selected such that no lines from the same movie were present in both the training and testing sets.

The model being evaluated below is a sequence-to-sequence model, with attention, where the encoder and decoder were both 2-layer, 512 hidden unit LSTMs. 
The model was trained with a vocabulary of the 2k most common words seen in the training set.

### Aggregate Performance
Below are reported the BLEU scores and accuracy numbers over the test dataset for both a trained model and a baseline, where the baseline is the identity function (which assumes no errors exist in the input).

You'll notice that the model outperforms this baseline for all bucket sizes in terms of accuracy, and outperforms all but one in terms of BLEU score. 
This tells us that applying the Deep Text Corrector model to a potentially errant writing sample would, on average, result in a more grammatically correct writing sample. 
Anyone who tends to make errors similar to those the model has been trained on could therefore benefit from passing their messages through this model.

```
Bucket 0: (10, 10)
        Baseline BLEU = 0.8341
        Model BLEU = 0.8516
        Baseline Accuracy: 0.9083
        Model Accuracy: 0.9384
Bucket 1: (15, 15)
        Baseline BLEU = 0.8850
        Model BLEU = 0.8860
        Baseline Accuracy: 0.8156
        Model Accuracy: 0.8491
Bucket 2: (20, 20)
        Baseline BLEU = 0.8876
        Model BLEU = 0.8880
        Baseline Accuracy: 0.7291
        Model Accuracy: 0.7817
Bucket 3: (40, 40)
        Baseline BLEU = 0.9099
        Model BLEU = 0.9045
        Baseline Accuracy: 0.6073
        Model Accuracy: 0.6425
```

### Examples
Decoding a sentence with a missing article:

```
In [31]: decode("Kvothe went to market")
Out[31]: 'Kvothe went to the market'
```

Decoding a sentence with then/than confusion:

```
In [30]: decode("the Cardinals did better then the Cubs in the offseason")
Out[30]: 'the Cardinals did better than the Cubs in the offseason'
```


## Implementation Details
This project reuses and slightly extends TensorFlow's [`Seq2SeqModel`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/seq2seq_model.py), which itself implements a sequence-to-sequence model with an attention mechanism as described in https://arxiv.org/pdf/1412.7449v3.pdf. 
The primary contributions of this project are:

- `data_reader.py`: an abstract class that defines the interface for classes which are capable of reading a source dataset and producing input-output pairs, where the input is a grammatically incorrect variant of a source sentence and the output is the original sentence.
- `text_corrector_data_readers.py`: contains a few implementations of `DataReader`, one over the [Penn Treebank dataset](https://www.google.com/url?q=http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz&usg=AFQjCNG0IP5OHusdIAdJIrrem-HMck9AzA) and one over the [Cornell Movie-Dialogs Corpus](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html).
- `text_corrector_models.py`: contains a version of `Seq2SeqModel` modified such that it implements the logic described in [Biased Decoding](#biased-decoding)
- `correct_text.py`: a collection of helper functions that together allow for the training of a model and the usage of it to decode errant input sequences (at test time). The `decode` method defined here implements the [OOV token resolution logic](#handling-oov-tokens). This also defines a main method, and can be invoked from the command line. It was largely derived from TensorFlow's [`translate.py`](https://www.tensorflow.org/tutorials/seq2seq/).
- `TextCorrector.ipynb`: an IPython notebook which ties together all of the above pieces to allow for the training and evaluation of the model in an interactive fashion.

### Example Usage
Note: this project requires TensorFlow version >= 0.11. See [this page](https://www.tensorflow.org/get_started/os_setup) for setup instructions.

**Preprocess Movie Dialog Data**
```
python preprocessors/preprocess_movie_dialogs.py --raw_data movie_lines.txt \
                                                 --out_file preprocessed_movie_lines.txt
```
This preprocessed file can then be split up however you like to create training, validation, and testing sets.

**Training:**
```
python correct_text.py --train_path /movie_dialog_train.txt \
                       --val_path /movie_dialog_val.txt \
                       --config DefaultMovieDialogConfig \
                       --data_reader_type MovieDialogReader \
                       --model_path /movie_dialog_model
```

**Testing:**
```
python correct_text.py --test_path /movie_dialog_test.txt \
                       --config DefaultMovieDialogConfig \
                       --data_reader_type MovieDialogReader \
                       --model_path /movie_dialog_model \
                       --decode
```

