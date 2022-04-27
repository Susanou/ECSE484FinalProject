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

loader = Loader("Loading dataset and model...")
loader.start()

model = None
texts = []
labels = []

languages_data_folder = "datasets"
dataset = load_files(languages_data_folder, encoding='ISO-8859-1')
docs_train, docs_test, labels_train, labels_test = train_test_split(
    dataset.data, dataset.target, test_size=0.9, random_state=0, shuffle=True)

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
            labels.append(x.split("_")[1][:-4])

loader.stop()

predictions = model.predict(texts)
prob = model.predict_proba(texts)
probs = np.argsort(prob, axis=1)[:,-3:]

for i,p in enumerate(predictions):
    if dataset.target_names[p] != labels[i]:
        print(f"text {i} was classified as {dataset.target_names[p]} when it is of class {labels[i]}")

frame1 = pd.DataFrame(probs, index=labels)
frame1 = frame1.applymap(lambda x: dataset.target_names[x])
frame2 = pd.DataFrame(prob, index=labels, columns=dataset.target_names)

print(frame1) #column 2 is the choice made by the model
print()
print(frame2) #probability of being of a label for each text. Index is the actual value

predicted = model.predict(docs_test)
# Model Accuracy
print(metrics.classification_report(labels_test, predicted,
                                    target_names=dataset.target_names))
cm = metrics.confusion_matrix(labels_test, predicted)
print(cm)
plt.matshow(cm, cmap=plt.cm.jet)
plt.show()
