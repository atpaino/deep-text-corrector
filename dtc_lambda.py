from __future__ import print_function

import json
import os
import pickle

import boto3
import tensorflow as tf

from correct_text import create_model, DefaultMovieDialogConfig, decode_sentence
from text_corrector_data_readers import MovieDialogReader


def safe_mkdir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass


def download(client, filename, local_path=None, s3_path=None):
    if s3_path is None:
        s3_path = MODEL_PARAMS_DIR + "/" + filename
    if local_path is None:
        local_path = os.path.join(MODEL_PATH, filename)

    print("Downloading " + filename)
    client.download_file(BUCKET_NAME, s3_path, local_path)


# Define resources on S3.
BUCKET_NAME = "deeptextcorrecter"
ROOT_DATA_PATH = "/tmp/"
MODEL_PARAMS_DIR = "model_params"
MODEL_PATH = os.path.join(ROOT_DATA_PATH, MODEL_PARAMS_DIR)

# Create tmp dirs for storing data locally.
safe_mkdir(ROOT_DATA_PATH)
safe_mkdir(MODEL_PATH)

# Download files from S3 to local disk.
s3_client = boto3.client('s3')

model_ckpt = "41900"
tf_meta_filename = "translate.ckpt-{}.meta".format(model_ckpt)
download(s3_client, tf_meta_filename)

tf_params_filename = "translate.ckpt-{}".format(model_ckpt)
download(s3_client, tf_params_filename)

tf_ckpt_filename = "checkpoint"
download(s3_client, tf_ckpt_filename)

corrective_tokens_filename = "corrective_tokens.pickle"
corrective_tokens_path = os.path.join(ROOT_DATA_PATH,
                                      corrective_tokens_filename)
download(s3_client, corrective_tokens_filename,
         local_path=corrective_tokens_path)

token_to_id_filename = "token_to_id.pickle"
token_to_id_path = os.path.join(ROOT_DATA_PATH, token_to_id_filename)
download(s3_client, token_to_id_filename, local_path=token_to_id_path)

# Load model.
config = DefaultMovieDialogConfig()
sess = tf.Session()
print("Loading model")
model = create_model(sess, True, MODEL_PATH, config=config)
print("Loaded model")

with open(corrective_tokens_path) as f:
    corrective_tokens = pickle.load(f)
with open(token_to_id_path) as f:
    token_to_id = pickle.load(f)
data_reader = MovieDialogReader(config, token_to_id=token_to_id)
print("Done initializing.")


def process_event(event, context):
    print("Received event: " + json.dumps(event, indent=2))

    outputs = decode_sentence(sess, model, data_reader, event["text"],
                              corrective_tokens=corrective_tokens,
                              verbose=False)
    return {"input": event["text"], "output": " ".join(outputs)}
