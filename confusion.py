import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold

# import IRIS dataset to play with
iris = datasets.load_iris()

data = iris.data
target = iris.target

data.shape
# Data description
print(iris.DESCR)
class_names = iris.target_names
class_names
labels, counts = np.unique(target, return_counts=True)


def evaluate_model(data_x, data_y):
    k_fold = KFold(12, shuffle=True, random_state=1)

    predicted_targets = np.array([])
    actual_targets = np.array([])
    accuracy_list = list()

    for train_ix, test_ix in k_fold.split(data_x):
        train_x, train_y, test_x, test_y = data_x[train_ix], data_y[train_ix], data_x[test_ix], data_y[test_ix]

        # Fit the classifier
        classifier = svm.SVC().fit(train_x, train_y)

        # Predict the labels of the test set samples
        predicted_labels = classifier.predict(test_x)
        accuracy = accuracy_score(test_y, predicted_labels)

        predicted_targets = np.append(predicted_targets, predicted_labels)
        actual_targets = np.append(actual_targets, test_y)
        accuracy_list.append(accuracy)

    return predicted_targets, actual_targets, accuracy_list


def plot_confusion_matrix(predicted_labels_list, y_test_list):
    cnf_matrix = confusion_matrix(y_test_list, predicted_labels_list)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    generate_confusion_matrix(
        cnf_matrix, classes=class_names, title='Confusion matrix but not normalized')
    plt.show()

    # Plot normalized confusion matrix
    plt.figure()
    generate_confusion_matrix(cnf_matrix, classes=class_names,
                              normalize=True, title='Normalized confusion matrix')
    plt.show()


def generate_confusion_matrix(cnf_matrix, classes, normalize=False, title='Confusion matrix'):
    if normalize:
        cnf_matrix = cnf_matrix.astype(
            'float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix but not normalized')

    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.get_cmap('Reds'))
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cnf_matrix.max() / 2.

    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt), horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return cnf_matrix


predicted_target, actual_target, accuracy_list = evaluate_model(data, target)
plot_confusion_matrix(predicted_target, actual_target)
print("Mean accuracy:", np.mean(accuracy_list))
print("Standard deviation of accuracy:", np.std(accuracy_list))
