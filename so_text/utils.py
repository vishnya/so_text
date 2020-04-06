import re
import os.path
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, \
    classification_report

PUNC_TO_SPACE = re.compile('[/(){}\[\]\|@,;]')
STOPWORDS = set(stopwords.words('english'))

DATA_PICKLE_PATH = 'okc_df.pkl'
DATA_URL = 'https://s3.amazonaws.com/techblog-static/interview_dataset.csv'


def read_data():
    if os.path.exists(DATA_PICKLE_PATH):
        df = pd.read_pickle(DATA_PICKLE_PATH)
    else:
        df = pd.read_csv(DATA_URL)
        df.to_pickle(DATA_PICKLE_PATH)
    return df


#
# Text processing
#

def process_text(text):
    text = BeautifulSoup(text, "lxml").text
    text = text.lower()
    text = PUNC_TO_SPACE.sub(' ', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
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


#
# Performance metrics reports
#


def formatted_confusion_matrix(y_test, y_pred):
    confusion = confusion_matrix(y_test, y_pred)
    row_label = "Actual"
    col_label = "Predicted"
    col_space = len(row_label)
    index_middle = int(int(len(set(y_test))) / 2)
    print(" " * (col_space + 4), "  ".join([str(i) for i in set(y_test)]),
          " <-  {}".format(col_label))
    for index in range(len(set(y_test))):
        if index == index_middle:
            print(row_label, " ", index, confusion[index])
        else:
            print(" " * (col_space + 2), index, confusion[index])


def generate_performance_report(y_test, y_class, y_score):
    print('roc_auc score: %s' % roc_auc_score(y_test, y_score))
    print(classification_report(y_test, y_class))
    print('confusion matrix: \n')
    print(formatted_confusion_matrix(y_test, y_class))
