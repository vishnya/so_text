from feature_process_utils import read_data_into_df
from performance import generate_performance_report
from modeling_utils import tts

from scipy.sparse import hstack
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


def run_title_and_body_example_sep():
    df = read_data_into_df()
    X_train, X_test, y_train, y_test = tts(df.drop('label',
                                                   axis=1),
                                           df['label'])
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
    X_train, X_test, y_train, y_test = tts(df.drop('label',
                                                   axis=1),
                                           df['label'])
    vect = TfidfVectorizer(min_df=5).fit(X_train['Body_processed'])

    X_train['Text'] = X_train['Body_processed'] + X_train['Title_processed']
    X_train_vectorized = vect.transform(X_train['Text'])

    clf = MultinomialNB(alpha=0.1).fit(X_train_vectorized, y_train)

    X_test_vectorized = vect.transform(X_test['Text'])
    y_score = clf.predict_proba(X_test_vectorized)[:, 1]
    y_class = clf.predict(X_test_vectorized)

    generate_performance_report(y_test, y_score=y_score, y_class=y_class)
