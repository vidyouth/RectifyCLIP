"""Accuracy and confusion-matrix helpers shared across experiment days.

This module holds the general-purpose metric functions needed starting Day 2
(overall accuracy, per-class accuracy, confusion matrix) so they are pure,
testable, and reusable rather than duplicated inline wherever predictions get
scored. Day 6-specific metrics (recovery %, confidence-based measures) are not
implemented here yet — adding them now would be building ahead of the day that
needs them.

Day 8 v2 adds bootstrap-CI, balanced-accuracy, macro-F1, and prediction-
agreement helpers, needed because the v2 grid has 5 independent random-tilt
replicates per image per severity (not a single draw) — a plain accuracy
number no longer tells you how much of the spread is sampling noise over
images and over tilt directions.
"""

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score, f1_score
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

    def _key(row):
        """key_field may be a single field name (Day 5 usage) or a tuple/list
        of field names (Day 8 v2: pooling 5 replicates per image needs a
        composite (image_id, replicate) key — a plain image_id key would
        collide across replicates and silently drop data)."""
        if isinstance(key_field, (tuple, list)):
            return tuple(row[f] for f in key_field)
        return row[key_field]

    correct_a = {_key(row): _is_correct(row["correct"]) for row in rows_a}
    correct_b = {_key(row): _is_correct(row["correct"]) for row in rows_b}
    shared_keys = set(correct_a) & set(correct_b)

    b = sum(1 for k in shared_keys if correct_a[k] and not correct_b[k])
    c = sum(1 for k in shared_keys if not correct_a[k] and correct_b[k])
    n_discordant = b + c

    if n_discordant == 0:
        p_value = 1.0
    else:
        p_value = binomtest(min(b, c), n_discordant, 0.5, alternative="two-sided").pvalue

    return {"b": b, "c": c, "n_discordant": n_discordant, "p_value": p_value}


# --- Day 8 v2: bootstrap CI + replicate-aware metrics ------------------------------------


def bootstrap_ci(rows, statistic_fn, n_resamples=1000, seed=0, confidence=0.95, image_id_field="image_id"):
    """Bootstrap CI for a statistic, resampling at the IMAGE level (not the row level).

    The v2 grid has 5 random-tilt replicates per image per severity, so rows
    are not independent samples — two rows for the same image (different
    replicates) are more alike than two rows for different images. Resampling
    individual rows would understate the true uncertainty. Instead, this
    resamples the set of unique image_ids with replacement `n_resamples`
    times, and for each resampled image_id pulls in ALL of its rows (every
    replicate), so each bootstrap draw respects the same clustered structure
    as the real data.

    `statistic_fn(rows) -> float` computes whatever point statistic is of
    interest (accuracy, balanced accuracy, recovery %, ...) over a list of
    row dicts; it is called once on the real data for the point estimate and
    once per resample. Returns (point_estimate, ci_low, ci_high) via the
    percentile method. Rows where statistic_fn returns None (e.g. recovery %
    with a zero denominator) are dropped before taking percentiles; if ALL
    resamples return None, returns (point_estimate, None, None).
    """
    rng = np.random.RandomState(seed)
    image_ids = sorted(set(r[image_id_field] for r in rows))
    by_image = {}
    for r in rows:
        by_image.setdefault(r[image_id_field], []).append(r)

    point = statistic_fn(rows)

    boot_stats = []
    n = len(image_ids)
    for _ in range(n_resamples):
        sampled_ids = rng.choice(image_ids, size=n, replace=True)
        resampled_rows = []
        for img_id in sampled_ids:
            resampled_rows.extend(by_image[img_id])
        stat = statistic_fn(resampled_rows)
        if stat is not None:
            boot_stats.append(stat)

    if not boot_stats:
        return point, None, None

    alpha = 1 - confidence
    lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return point, lo, hi


def balanced_accuracy(rows, class_names):
    """Balanced accuracy (mean per-class recall) — robust to the 34%-Baroque
    class imbalance that plain top1_accuracy is not."""
    if not rows:
        return None
    true_labels = [row["true_label"] for row in rows]
    predicted_labels = [row["predicted_label"] for row in rows]
    return float(balanced_accuracy_score(true_labels, predicted_labels))


def macro_f1(rows, class_names):
    """Macro-averaged F1 (unweighted mean of each class's F1), over class_names."""
    if not rows:
        return None
    true_labels = [row["true_label"] for row in rows]
    predicted_labels = [row["predicted_label"] for row in rows]
    return float(f1_score(true_labels, predicted_labels, labels=class_names, average="macro", zero_division=0))


def accuracy_per_replicate(rows, n_replicates):
    """Return a list of per-replicate top1 accuracy, one value per replicate 0..n_replicates-1.

    Used to report "accuracy mean +/- std across replicates" as a direct
    descriptive statistic over the 5 replicate draws themselves — distinct
    from (and a complement to) the image-level bootstrap CI, which instead
    quantifies sampling uncertainty over which images happened to be in the
    dataset.
    """
    return [
        top1_accuracy([r for r in rows if int(r["replicate"]) == rep])
        for rep in range(n_replicates)
    ]


def prediction_agreement(rows, clean_by_image_id):
    """Fraction of rows whose predicted_label matches that image's clean prediction.

    `clean_by_image_id` is {image_id: clean_predicted_label}. The clean
    condition carries no randomness, so this is matched by image_id alone,
    regardless of which replicate/severity the row in `rows` is.
    """
    if not rows:
        return None
    agree = sum(1 for r in rows if r["predicted_label"] == clean_by_image_id[r["image_id"]])
    return agree / len(rows)


def recovery_percentage(rows):
    """recovery = (acc_rectified - acc_distorted) / (acc_clean - acc_distorted) * 100.

    `rows` must contain rows from all 3 conditions ("clean", "distorted",
    "rectified") for the severity/subset being measured. Returns None if the
    denominator (clean - distorted accuracy) is exactly 0, matching Day 6's
    existing recovery-% convention (src/day6_analysis.py:compute_recovery).
    """
    clean_acc = top1_accuracy([r for r in rows if r["condition"] == "clean"])
    distorted_acc = top1_accuracy([r for r in rows if r["condition"] == "distorted"])
    rectified_acc = top1_accuracy([r for r in rows if r["condition"] == "rectified"])
    denom = clean_acc - distorted_acc
    if denom == 0:
        return None
    return (rectified_acc - distorted_acc) / denom * 100
