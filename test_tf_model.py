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
import os

from os.path import join, isfile

from spacy.lang.fr import French
from spacy.lang.fr.stop_words import STOP_WORDS

from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras import losses

from utils import Loader


loader = Loader("Loading dataset and model...")
#loader.start()

model = load_model("tf_model.tf")
texts = []
labels = []

raw_ds = tf.keras.utils.text_dataset_from_directory(
"datasets",
labels='inferred',
label_mode='int',
)

for x in os.listdir("test_files"):
    #print(args.newPath)
    if isfile(join("test_files", x)):
        with open(join("test_files", x), "r") as f:
            texts.append(f.read())
            labels.append(x.split("_")[1][:-4])

predictions = model.predict(texts)
for i,p in enumerate(predictions):
    if raw_ds.class_names[p] != labels[i]:
        print(f"text {i} was classified as {raw_ds.class_names[p]} when it is of class {labels[i]}")