"""
To run these examples, e.g.:
 >>> import so_text.readme_analysis_runs as run
 >>> run.run_alternate_parser_example()
"""

from so_text.feature_process_utils import read_data_into_df, process_text
from so_text.performance import generate_performance_report

from scipy.sparse import hstack
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def run_title_and_body_example_sep():
    df = read_data_into_df()
    df.dropna(inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(df.drop('label',
                                                                axis=1),
                                                        df['label'],
                                                        random_state=0)
    vect = TfidfVectorizer(min_df=5).fit(X_train['Body_processed'])

    X_train_vectorized_body = vect.transform(X_train['Body_processed'])
    X_train_vectorized_title = vect.transform(X_train['Title_processed'])
    X_train_vectorized = hstack(
        [X_train_vectorized_body, X_train_vectorized_title],
        'csr')

    clf = MultinomialNB(alpha=0.1).fit(X_train_vectorized, y_train)

    X_test_vectorized_body = vect.transform(X_test['Body_processed'])
    X_test_vectorized_title = vect.transform(X_test['Title_processed'])
    X_test_vectorized = hstack(
        [X_test_vectorized_body, X_test_vectorized_title],
        'csr')

    y_score = clf.predict_proba(X_test_vectorized)[:, 1]
    y_class = clf.predict(X_test_vectorized)

    generate_performance_report(y_test, y_score=y_score, y_class=y_class)


def run_title_and_body_example_concat():
    df = read_data_into_df()
    df.dropna(inplace=True)
    df['Text'] = df['Body_processed'] + df['Title_processed']
    X_train, X_test, y_train, y_test = train_test_split(df.drop('label',
                                                                axis=1),
                                                        df['label'],
                                                        random_state=0)

    vect = TfidfVectorizer(min_df=5).fit(X_train['Text'])
    X_train_vectorized = vect.transform(X_train['Text'])

    clf = MultinomialNB(alpha=0.1).fit(X_train_vectorized, y_train)

    X_test_vectorized = vect.transform(X_test['Text'])
    y_score = clf.predict_proba(X_test_vectorized)[:, 1]
    y_class = clf.predict(X_test_vectorized)

    generate_performance_report(y_test, y_score=y_score, y_class=y_class)


def run_no_preprocess_example():
    df = read_data_into_df()
    df.dropna(inplace=True)
    df['Text'] = df['Body'] + df['Title']
    X_train, X_test, y_train, y_test = train_test_split(df.drop('label',
                                                                axis=1),
                                                        df['label'],
                                                        random_state=0)

    vect = TfidfVectorizer(min_df=5).fit(X_train['Text'])
    X_train_vectorized = vect.transform(X_train['Text'])

    clf = MultinomialNB(alpha=0.1).fit(X_train_vectorized, y_train)

    X_test_vectorized = vect.transform(X_test['Text'])
    y_score = clf.predict_proba(X_test_vectorized)[:, 1]
    y_class = clf.predict(X_test_vectorized)

    generate_performance_report(y_test, y_score=y_score, y_class=y_class)


def run_alternate_parser_example():
    df = read_data_into_df()
    df.dropna(inplace=True)
    df['Text'] = df['Body'] + df['Title']
    df['Text'] = df['Text'].apply(process_text)
    X_train, X_test, y_train, y_test = train_test_split(df.drop('label',
                                                                axis=1),
                                                        df['label'],
                                                        random_state=0)

    vect = TfidfVectorizer(min_df=5).fit(X_train['Text'])
    X_train_vectorized = vect.transform(X_train['Text'])

    clf = MultinomialNB(alpha=0.1).fit(X_train_vectorized, y_train)

    X_test_vectorized = vect.transform(X_test['Text'])
    y_score = clf.predict_proba(X_test_vectorized)[:, 1]
    y_class = clf.predict(X_test_vectorized)

    generate_performance_report(y_test, y_score=y_score, y_class=y_class)


def get_largest_smallest_tfidf(vect, X_train_vectorized, bound=5):
    feature_names = np.array(vect.get_feature_names()).reshape(-1, 1)
    tfidf_values = X_train_vectorized.max(0).toarray()[0].reshape(-1, 1)
    tfidf_df = pd.DataFrame(data=np.hstack((feature_names, tfidf_values)),
                            columns=['features', 'tfidf'])
    smallest_tfidf = tfidf_df.sort_values(by=['tfidf', 'features']).set_index(
        'features')[:bound]
    largest_tfidf = tfidf_df.sort_values(by=['tfidf', 'features'],
                                         ascending=[False, True]).set_index(
        'features')[:bound]
    return smallest_tfidf['tfidf'].apply(float), largest_tfidf['tfidf'].apply(
        float)
