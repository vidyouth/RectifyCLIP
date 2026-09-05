"""Accuracy and confusion-matrix helpers shared across experiment days.

This module holds the general-purpose metric functions needed starting Day 2
(overall accuracy, per-class accuracy, confusion matrix) so they are pure,
testable, and reusable rather than duplicated inline wherever predictions get
scored. Day 6-specific metrics (recovery %, confidence-based measures) are not
implemented here yet — adding them now would be building ahead of the day that
needs them.
"""

from scipy.stats import binomtest
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


def paired_mcnemar(rows_a, rows_b, key_field="image_id"):
    """McNemar's exact test for two conditions evaluated on the same images.

    Added on Day 5 to properly characterize the headline table's
    "clean >= rectified >= distorted" sanity check: a small numeric gap
    between two conditions' accuracy (e.g. rectified slightly below
    distorted) can be ordinary sampling noise rather than a real effect, and
    a raw accuracy comparison can't tell the two apart on its own. This test
    can, because it uses the fact that both conditions were evaluated on the
    *same* images (a paired design), by looking only at the discordant pairs
    — images where the two conditions disagree on correctness.

    rows_a, rows_b: lists of prediction-row dicts (each with `key_field` and
    a boolean-ish "correct" field — "True"/"False" strings from a CSV are
    accepted) for the two conditions being compared, covering the same set of
    images.

    Returns a dict: b (count correct in A, wrong in B), c (count wrong in A,
    correct in B), n_discordant (b + c), and p_value (two-sided exact
    binomial test of b against Binomial(b + c, 0.5) — the standard McNemar
    exact test for small/moderate discordant counts).
    """

    def _is_correct(value):
        """Normalize a row's "correct" field (bool, or "True"/"False" from a CSV) to bool."""
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() == "true"

    correct_a = {row[key_field]: _is_correct(row["correct"]) for row in rows_a}
    correct_b = {row[key_field]: _is_correct(row["correct"]) for row in rows_b}
    shared_keys = set(correct_a) & set(correct_b)

    b = sum(1 for k in shared_keys if correct_a[k] and not correct_b[k])
    c = sum(1 for k in shared_keys if not correct_a[k] and correct_b[k])
    n_discordant = b + c

    if n_discordant == 0:
        p_value = 1.0
    else:
        p_value = binomtest(min(b, c), n_discordant, 0.5, alternative="two-sided").pvalue

    return {"b": b, "c": c, "n_discordant": n_discordant, "p_value": p_value}
