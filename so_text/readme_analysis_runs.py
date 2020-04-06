"""
To run these examples, e.g.:
 >>> import so_text.readme_analysis_runs as run
 >>> run.run_alternate_parser_example()
"""

from so_text.utils import read_data, process_text, \
    generate_performance_report

from scipy.sparse import hstack
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def run_title_and_body_example_sep():
    df = read_data()
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


def run_title_and_body_example_concat(ngram_range=(1, 1)):
    df = read_data()
    df.dropna(inplace=True)
    df['Text'] = df['Body_processed'] + df['Title_processed']
    X_train, X_test, y_train, y_test = train_test_split(df.drop('label',
                                                                axis=1),
                                                        df['label'],
                                                        random_state=0)

    vect = TfidfVectorizer(min_df=5,
                           ngram_range=ngram_range).fit(X_train['Text'])
    X_train_vectorized = vect.transform(X_train['Text'])

    clf = MultinomialNB(alpha=0.1).fit(X_train_vectorized, y_train)

    X_test_vectorized = vect.transform(X_test['Text'])
    y_score = clf.predict_proba(X_test_vectorized)[:, 1]
    y_class = clf.predict(X_test_vectorized)

    generate_performance_report(y_test, y_score=y_score, y_class=y_class)


def run_bigrams_comparison():
    return run_title_and_body_example_concat(ngram_range=(1, 2))


def run_long_short_analysis(ngram_range=(1, 1), upper=1700, lower=20):
    df = read_data()
    df.dropna(inplace=True)
    df['Text'] = df['Body_processed'] + df['Title_processed']
    df['Length'] = df.Text.apply(lambda y: len(y.split()))
    X_train, X_test, y_train, y_test = train_test_split(df.drop('label',
                                                                axis=1),
                                                        df['label'],
                                                        random_state=0)

    vect = TfidfVectorizer(min_df=5,
                           ngram_range=ngram_range).fit(X_train['Text'])
    X_train_vectorized = vect.transform(X_train['Text'])

    clf = MultinomialNB(alpha=0.1).fit(X_train_vectorized, y_train)

    X_test_vectorized = vect.transform(X_test['Text'])
    y_score = clf.predict_proba(X_test_vectorized)[:, 1]
    y_class = clf.predict(X_test_vectorized)

    y_actual_df = pd.DataFrame(y_test)
    y_actual_df.reset_index(inplace=True, drop=True)
    y_actual_df.rename(columns={'label': 'actual'}, inplace=True)

    y_score_df = pd.DataFrame(y_score)
    y_score_df.reset_index(inplace=True, drop=True)
    y_score_df.rename(columns={0: 'score'}, inplace=True)

    y_class_df = pd.DataFrame(y_class)
    y_class_df.reset_index(inplace=True, drop=True)
    y_class_df.rename(columns={0: 'class'}, inplace=True)

    X_test.reset_index(inplace=True, drop=True)

    df_concat = pd.concat([X_test, y_actual_df, y_class_df, y_score_df], axis=1)

    X_test_ri = X_test.reset_index(inplace=True, drop=True)
    pd.concat([X_test_ri, y_actual_df, y_class_df, y_score_df], axis=1)
    df_upper = df_concat[df_concat.Length > upper]
    upper_size = df_upper.shape[0]
    df_lower = df_concat[df_concat.Length < lower]
    lower_size = df_lower.shape[0]

    print(f"Performance on docs of word length > {upper}, comprising "
          f"{upper_size} documents")
    generate_performance_report(y_test=df_upper.actual,
                                y_class=df_upper['class'],
                                y_score=df_upper.score)

    print(f"Performance on docs of word length < {lower}, comprising "
          f"{lower_size} documents")
    generate_performance_report(y_test=df_lower.actual,
                                y_class=df_lower['class'],
                                y_score=df_lower.score)


def run_no_preprocess_example():
    df = read_data()
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
    df = read_data()
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


def run_nn_word_similarity_example(glove_6B_100d_txt_path):
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.models import Sequential
    from keras.layers import Dense, Flatten
    from keras.layers.embeddings import Embedding

    df = read_data()
    df["Text"] = df['Title_processed'] + df['Body_processed']
    df.Text = df.Text.astype(str)
    docs = df["Text"].values
    labels = df["label"].values

    t = Tokenizer()
    t.fit_on_texts(docs)
    vocab_size = len(t.word_index) + 1
    encoded_docs = t.texts_to_sequences(docs)
    max_length = 20
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

    embeddings_index = dict()
    with open(glove_6B_100d_txt_path, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((vocab_size, 100))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    model = Sequential()
    model.add(
        Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=20,
                  trainable=False))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                  metrics=['acc'])
    print(model.summary())

    X_train, X_test, y_train, y_test = train_test_split(padded_docs, labels,
                                                        test_size=0.2,
                                                        random_state=42)
    model.fit(X_train, y_train, epochs=10, verbose=0)
