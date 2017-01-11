"""Preprocesses Cornell Movie Dialog data."""
import nltk
import tensorflow as tf

tf.app.flags.DEFINE_string("raw_data", "", "Raw data path")
tf.app.flags.DEFINE_string("out_file", "", "File to write preprocessed data "
                                           "to.")

FLAGS = tf.app.flags.FLAGS


def main(_):
    with open(FLAGS.raw_data, "r") as raw_data, \
            open(FLAGS.out_file, "w") as out:
        for line in raw_data:
            parts = line.split(" +++$+++ ")
            dialog_line = parts[-1]
            s = dialog_line.strip().lower().decode("utf-8", "ignore")
            preprocessed_line = " ".join(nltk.word_tokenize(s))
            out.write(preprocessed_line + "\n")

if __name__ == "__main__":
    tf.app.run()
