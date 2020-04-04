from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, \
    classification_report


def generate_performance_report(y_test, y_class, y_score):
    print('accuracy: %s' % accuracy_score(y_class, y_test))
    print('roc_auc score: %s' % roc_auc_score(y_test, y_score))
    print(classification_report(y_test, y_class))
