import sys
import re
import matplotlib.pyplot as plt
import argparse
import time
import pandas as pd
import numpy as np
import string
import joblib
import spacy
import tensorflow as tf

from spacy.lang.fr import French
from spacy.lang.fr.stop_words import STOP_WORDS

from tensorflow.keras import layers
from tensorflow.keras import losses

from utils import Loader

#############
# GLOBAL VARS
#############


# Create our list of punctuation marks
punctuations = string.punctuation

# Create our list of stopwords
nlp = spacy.load('fr_core_news_sm')

parser = French()

# Use GPUs
tf.config.set_soft_device_placement(True)
#tf.config.run_functions_eagerly(True) # debug for model not compiling
#tf.debugging.set_log_device_placement(True) #uncomment if need to check that it is executing off of GPU
tf.get_logger().setLevel('ERROR')

@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')

# Basic function to clean the text
def clean_text(text):
    # Removing spaces and converting text into lowercase
    return text.strip().lower()


def sentence_tokenizer(sentence):
    tokens = parser(sentence)

    tokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens] # only keep words that are unique from each other (runs == run != runner)

    #tokens = [word for word in tokens if word not in STOP_WORDS and word not in punctuations]

    return tokens

def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label

if __name__ == "__main__":
    global data_folder
    global dataset
    global docs_train, docs_test, y_train, y_test

    loader = Loader("Loading dataset...", end="datasets loaded")

    arg_parser = argparse.ArgumentParser(description="Script for classifying french poems")

    arg_parser.add_argument("dataPath", default="datasets",nargs='?', type=str, help="Path to the folder where all the text data is stored")

    args = arg_parser.parse_args()

    #loader.start()

    batch_size = 32
    seed = 42

    raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    args.dataPath,
    labels='inferred',
    label_mode='int',
    batch_size=32,
    shuffle=True,
    seed=seed,
    validation_split=0.5,
    subset="training"
    )

    raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    args.dataPath, 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='validation', 
    seed=seed)

    max_features = 10000
    sequence_length = 250   

    vectorize_layer = layers.TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length)

    # Make a text-only dataset (without labels), then call adapt
    train_text = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)

    embedding_dim = 16

    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)

    model = tf.keras.Sequential([
    layers.Embedding(max_features + 1, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(4)])

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=['accuracy'])

    epochs = 10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs)

    export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')
    ])

    export_model.compile(
        loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
    )

    export_model.save("tf_model.tf")