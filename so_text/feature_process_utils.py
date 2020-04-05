import re

import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import pandas as pd

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

DATA_PICKLE_PATH = 'okc_df.pkl'
DATA_URL = 'https://s3.amazonaws.com/techblog-static/interview_dataset.csv'


def read_data_into_df():
    df = pd.read_csv(
        'https://s3.amazonaws.com/techblog-static/interview_dataset'
        '.csv')
    return df


def process_text(text):
    """
        text: a string

        return: modified initial string
    """
    text = BeautifulSoup(text, "lxml").text  # HTML decoding
    text = text.lower()  # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ',
                                   text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('',
                              text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if
                    word not in STOPWORDS)  # delete stopwors from text
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
