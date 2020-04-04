import os
from pathlib import Path
import re

import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import pandas as pd

RE_PUNCT = re.compile('[/(){}\[\]\|@,;]')
RE_ALPHA = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

DATA_DIR = '/data/'
DATA_PICKLE_PATH = DATA_DIR + 'okc_df.pkl'
DATA_URL = 'https://s3.amazonaws.com/techblog-static/interview_dataset.csv'


def read_data_into_df():
    Path(DATA_DIR).mkdir(parents=True,
                         exist_ok=True)
    if not os.path.exists(DATA_PICKLE_PATH):
        df = pd.read_csv(DATA_URL)
        pd.to_pickle(DATA_PICKLE_PATH)

    else:
        df = pd.read_pickle(DATA_PICKLE_PATH)
    return df


def clean_text(text, stopwords=True):
    text = BeautifulSoup(text, "lxml").text.lower()
    text = RE_PUNCT.sub(' ', text).RE_ALPHA.sub('', text)
    if stopwords:
        text = ' '.join(word for word in text.split() if
                        word not in STOPWORDS)
    return text


def tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    porter_stemmer = nltk.PorterStemmer()
    words = [porter_stemmer.stem(word) for word in words]
    return words


def pos_ratio(string, pos):
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    tokens = nltk.word_tokenize(string)
    tagged = nltk.pos_tag(tokens)
    matching_tagged = [word for word, p in tagged if pos == p]
    return len(matching_tagged) / len(tagged)
