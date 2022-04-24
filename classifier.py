import sys
import matplotlib.pyplot as plt
import argparse
import time
import pandas as pd
import numpy as np
import string

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from spacy.lang.fr import French
from spacy.lang.fr.stop_words import STOP_WORDS

#############
# GLOBAL VARS
#############


# Create our list of punctuation marks
punctuations = string.punctuation

# Create our list of stopwords
nlp = spacy.load('fr')

parser = French()

def sentence_tokenizer(sentence):
    tokens = parser(sentence)

    tokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens] # only keep words that are unique from each other (runs == run != runner)

    tokens = [word for word in tokens if word not in STOP_WORDS and word not in punctuations]

    return tokens