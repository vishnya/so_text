from so_text.feature_process_utils import read_data_into_df, tokenizer
from so_text.performance import generate_performance_report

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

TFIDF_MIN_DF = .0025
TFIDF_MAX_DF = 0.25
TF_IDF_NGRAM_RANGE = (1, 3)

XGB_MAX_DEPTH = 3
XGB_N_ESTIMATORS = 100
XGB_LEARNING_RATE = .1
XGB_SVD_N_COMPONENTS = 100


def feature_process():
    df = read_data_into_df()
    df.dropna(inplace=True)
    df['Text'] = df['Body_processed'] + df['Title_processed']
    df['Length'] = df.Text.apply(lambda x: len(x.split()))
    return df


class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.field]


class NumberSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.field]]


def main():
    df = feature_process()

    X_train, X_test, y_train, y_test = train_test_split(
        df[['Text', 'Length']],
        df['label'],
        random_state=0)

    classifier = Pipeline([
        ('features', FeatureUnion([
            ('text', Pipeline([
                ('colext', TextSelector('Text')),
                ('tfidf',
                 TfidfVectorizer(tokenizer=tokenizer,
                                 # stop_words=stop_words,
                                 min_df=TFIDF_MIN_DF,
                                 max_df=TFIDF_MAX_DF,
                                 ngram_range=TF_IDF_NGRAM_RANGE)),
                ('svd', TruncatedSVD(algorithm='randomized',
                                     n_components=XGB_SVD_N_COMPONENTS)),
            ])),
            ('words', Pipeline([
                ('wordext', NumberSelector('Body_length')),
                ('wscaler', StandardScaler()),
            ])),
        ])),
        (
            'clf',
            XGBClassifier(max_depth=XGB_MAX_DEPTH,
                          n_estimators=XGB_N_ESTIMATORS,
                          learning_rate=XGB_LEARNING_RATE)),
    ])

    classifier.fit(X_train, y_train)
    y_class = classifier.predict(X_test)
    y_score = classifier.predict_proba(X_test)[:, 1]

    generate_performance_report(y_test, y_class, y_score)


if __name__ == "__main__":
    main()
