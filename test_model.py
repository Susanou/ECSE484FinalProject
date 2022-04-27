import sys
import os
import matplotlib.pyplot as plt
import argparse
import time
import pandas as pd
import numpy as np
import string
import joblib
import spacy

from os.path import isfile, join

from spacy.lang.fr import French
from spacy.lang.fr.stop_words import STOP_WORDS

from sklearn.naive_bayes import MultinomialNB as naive
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics

from utils import Loader


model = None
texts = []

languages_data_folder = "datasets"
dataset = load_files(languages_data_folder, encoding='ISO-8859-1')
docs_train, docs_test, labels_train, labels_test = train_test_split(
    dataset.data, dataset.target, test_size=0.5, random_state=0, shuffle=True)

if os.path.isfile("model.pkl"):
    model = joblib.load("model.pkl")
else:
    print("No file")
    exit()

for x in os.listdir("test_files"):
    #print(args.newPath)
    if isfile(join("test_files", x)):
        with open(join("test_files", x), "r") as f:
            texts.append(f.read())

print(dataset.target_names[model.predict(texts)[0]])
