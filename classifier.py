import sys
import matplotlib.pyplot as plt
import argparse
import time
import pandas as pd
import numpy as np
import string
import joblib
import spacy

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

#############
# GLOBAL VARS
#############


# Create our list of punctuation marks
punctuations = string.punctuation

# Create our list of stopwords
nlp = spacy.load('fr_core_news_sm')

parser = French()


# Custom transformer using spaCy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# Basic function to clean the text
def clean_text(text):
    # Removing spaces and converting text into lowercase
    return text.strip().lower()


def sentence_tokenizer(sentence):
    tokens = parser(sentence)

    tokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens] # only keep words that are unique from each other (runs == run != runner)

    #tokens = [word for word in tokens if word not in STOP_WORDS and word not in punctuations]

    return tokens


if __name__ == "__main__":
    global data_folder
    global dataset
    global docs_train, docs_test, y_train, y_test

    arg_parser = argparse.ArgumentParser(description="Script for classifying french poems")

    arg_parser.add_argument("dataPath", default="../dataFitting",nargs='?', type=str, help="Path to the folder where all the text data is stored")

    args = arg_parser.parse_args()

    languages_data_folder = args.dataPath
    dataset = load_files(languages_data_folder, encoding='ISO-8859-1')
    docs_train, docs_test, labels_train, labels_test = train_test_split(
        dataset.data, dataset.target, test_size=0.5, random_state=0, shuffle=True)

    print(labels_train)
    print(labels_test)

    vector = CountVectorizer(tokenizer = sentence_tokenizer, ngram_range=(1,1))

    classifier = LogisticRegression()

    pipe = Pipeline(
        [
            #("cleaner", predictors()),
            ("vect", vector),
            ('clf', naive(alpha=1.0, fit_prior=True))
        ], verbose=True
    )

    parameters={
        'vect__ngram_range': [(1, 1), (1, 2), (1,3), (1,4), (1,5)],
        'clf__fit_prior': (True, False),
        'clf__alpha': (1.0, 0.1, 0.5, 2.0, .25, 0.75, 0.002),
    }

    gs_clf = GridSearchCV(pipe, parameters, cv=5, n_jobs=-1)
    gs_clf.fit(docs_train, labels_train)

    print(gs_clf.best_params_)

    predicted = gs_clf.predict(docs_test)
    # Model Accuracy
    print(metrics.classification_report(labels_test, predicted,
                                        target_names=dataset.target_names))
    cm = metrics.confusion_matrix(labels_test, predicted)
    print(cm)
    plt.matshow(cm, cmap=plt.cm.jet)
    plt.show()