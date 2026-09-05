"""Accuracy and confusion-matrix helpers shared across experiment days.

This module holds the general-purpose metric functions needed starting Day 2
(overall accuracy, per-class accuracy, confusion matrix) so they are pure,
testable, and reusable rather than duplicated inline wherever predictions get
scored. Day 6-specific metrics (recovery %, confidence-based measures) are not
implemented here yet — adding them now would be building ahead of the day that
needs them.
"""

from sklearn.metrics import confusion_matrix as sk_confusion_matrix


def top1_accuracy(rows):
    """Return overall top-1 accuracy over a list of prediction rows.

    Each row is a dict-like object with "true_label" and "predicted_label"
    keys. Returns a float in [0, 1].
    """
    if not rows:
        return 0.0
    correct = sum(1 for row in rows if row["predicted_label"] == row["true_label"])
    return correct / len(rows)


def per_class_accuracy(rows, class_names):
    """Return {class_name: accuracy} for each class in class_names.

    Accuracy for a class is computed only over rows whose true_label equals
    that class (i.e. per-class recall). A class with zero rows in `rows`
    reports accuracy None rather than dividing by zero.
    """
    result = {}
    for class_name in class_names:
        class_rows = [row for row in rows if row["true_label"] == class_name]
        if not class_rows:
            result[class_name] = None
            continue
        correct = sum(
            1 for row in class_rows if row["predicted_label"] == row["true_label"]
        )
        result[class_name] = correct / len(class_rows)
    return result


def confusion_matrix_counts(rows, class_names):
    """Return a len(class_names) x len(class_names) confusion matrix.

    Rows are true labels, columns are predicted labels, both ordered per
    class_names. Cell [i][j] is the count of samples with true label
    class_names[i] predicted as class_names[j].
    """
    true_labels = [row["true_label"] for row in rows]
    predicted_labels = [row["predicted_label"] for row in rows]
    matrix = sk_confusion_matrix(true_labels, predicted_labels, labels=class_names)
    return matrix.tolist()
