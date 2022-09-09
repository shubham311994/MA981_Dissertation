import numpy as np
import glob
from pathlib import Path
from typing import Tuple, Any

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold


def check_file_exists(file_path: str) -> bool:
    """Check if the file exists in a particular path.
    :param:: file_path: file path in string format.
    :return: boolean
    """
    file_path = Path(file_path)
    check_file_exist = file_path.is_file()

    return check_file_exist


def get_file_list_from_path(file_path: Path) -> Tuple[list, list]:
    """
    Get the list of files in the directory with particular format
    :param:: Path object of the file directory path.
    :return list of set files and list of fdt files.
    """
    list_of_fdt = glob.glob(f'{file_path}/*.fdt')
    list_of_set = glob.glob(f'{file_path}/*.set')

    return list_of_fdt, list_of_set


def compute_accuracy(y_actual: np.ndarray, y_predicted: np.ndarray) -> float:
    """
    Compute the accuracy for the given true and predicted values.
    :param: y_actual: The actual value of y variable or target.
    :param: y_predicted: The predicted value of y variable or target.
    :return: accuracy percentage score of the values.
    """
    accuracy = accuracy_score(y_actual, y_predicted)

    return accuracy


def compute_classification_report(y_actual: np.ndarray, y_predicted: np.ndarray):
    """
    Compute the classification report for the given true and predicted values.
    :param: y_actual: The actual value of y variable or target.
    :param: y_predicted: The predicted value of y variable or target.
    :return: the classification report with precision and recall.
    """

    cf_report = classification_report(y_actual, y_predicted)

    return cf_report


def compute_sensitivity_specificity(y_actual: np.ndarray, y_predicted: np.ndarray) -> Tuple[int, int]:
    """
    Compute the sensitivity and specificity using actual and predicted target values
    :param: y_actual: The actual value of y variable or target.
    :param: y_predicted: The predicted value of y variable or target.
    :return: sensitivity and specificity
    """
    pass


def apply_stratified_cv(model_object, x: np.ndarray, y: np.ndarray):
    """
    Divide the data into pre-defined splits and used the passed model_object to train and compute accuracy and
    classification_report.
    :param: model_object: model object of particular model applied on the dataset.
    :param: X: value or the features
    :param: y: value or the targets
    :return: list of accuracy value in decimals for all the splits.
    """
    accuracy_value = []
    f1_score_list = []
    sensitivity_list = []
    specificity_list = []
    sfk = StratifiedKFold(n_splits=10)
    for train_index, test_index in sfk.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model_object.fit(x_train, y_train)
        predictions = model_object.predict(x_test)
        accuracy_value.append(compute_accuracy(y_test, predictions))
        f1_score, sensitivity, specificity = accuracy_metrics(y_test, predictions)
        f1_score_list.append(f1_score)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)

    return accuracy_value, f1_score_list, sensitivity_list, specificity_list


def accuracy_metrics(y_test, predictions):
    conf_matrix = confusion_matrix(y_test, predictions)

    TP = conf_matrix[1][1]
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]

    # calculate accuracy
    conf_accuracy = (float(TP + TN) / float(TP + TN + FP + FN))

    # calculate mis-classification
    conf_misclassification = 1 - conf_accuracy

    # calculate the sensitivity
    conf_sensitivity = (TP / float(TP + FN))
    # calculate the specificity
    conf_specificity = (TN / float(TN + FP))

    # calculate precision
    conf_precision = (TN / float(TN + FP))
    # calculate f_1 score
    conf_f1 = 2 * ((conf_precision * conf_sensitivity) / (conf_precision + conf_sensitivity))

    return conf_f1, conf_sensitivity, conf_specificity
