from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay
)
from sklearn.calibration import (
    calibration_curve,
    CalibrationDisplay,
)
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


def calculate_metrics(predictions, probabilities, true_labels):
    f1 = f1_score(true_labels, predictions)
    mcc = matthews_corrcoef(true_labels, predictions)
    accuracy = accuracy_score(true_labels, predictions)
    cm = confusion_matrix(true_labels, predictions)
    auc = roc_auc_score(true_labels, probabilities)
    return f1, mcc, accuracy, cm, auc


def merge_np_arrays(*args):
    return np.concatenate(args, axis=0)


def load_data():
    predictions = sorted(glob('predictions_rank_*.npy'))
    probabilities = sorted(glob('proba_rank_*.npy'))
    labels = sorted(glob('true_labels_rank_*.npy'))

    predictions = [np.load(p) for p in predictions]
    probabilities = [np.load(p) for p in probabilities]
    labels = [np.load(l) for l in labels]

    predictions = merge_np_arrays(*predictions)
    probabilities = merge_np_arrays(*probabilities)
    labels = merge_np_arrays(*labels)
    return predictions, probabilities, labels


def plot_roc_curve(probabilities, true_labels):
    RocCurveDisplay.from_predictions(true_labels, probabilities)
    plt.savefig('roc_curve.png', dpi=300)
    plt.clf()


def plot_precision_recall_curve(probabilities, true_labels):
    PrecisionRecallDisplay.from_predictions(true_labels, probabilities)
    plt.savefig('precision_recall_curve.png', dpi=300)
    plt.clf()


def plot_calibration_curve(probabilities, true_labels):
    CalibrationDisplay.from_predictions(true_labels, probabilities)
    plt.savefig('calibration_curve.png', dpi=300)
    plt.clf()


def plot_confusion_matrix(predictions, true_labels):
    ConfusionMatrixDisplay.from_predictions(true_labels, predictions)
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.clf()


def main():
    predictions, probabilities, true_labels = load_data()

    f1, mcc, accuracy, cm, auc = calculate_metrics(
        predictions, probabilities, true_labels)
    print(f'F1: {f1}, MCC: {mcc}, Accuracy: {accuracy}, AUC: {auc}')

    plot_roc_curve(probabilities, true_labels)
    plot_precision_recall_curve(probabilities, true_labels)
    plot_calibration_curve(probabilities, true_labels)
    plot_confusion_matrix(predictions, true_labels)


if __name__ == '__main__':
    main()
