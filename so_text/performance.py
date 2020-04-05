from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, \
    classification_report
from sklearn.metrics import confusion_matrix


def generate_performance_report(y_test, y_class, y_score):
    print('roc_auc score: %s' % roc_auc_score(y_test, y_score))
    print(classification_report(y_test, y_class))
    print(formatted_confusion_matrix(y_test, y_class))


def formatted_confusion_matrix(y_test, y_pred):
    confusion = confusion_matrix(y_test, y_pred)
    row_label = "True"
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
