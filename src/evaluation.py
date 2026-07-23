import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def evaluate_binary_classifier(
    model_name,
    y_true,
    y_pred,
    y_score=None,
    y_proba=None,
):
    """
    Evaluate a binary classification model.

    Parameters
    ----------
    model_name : str
        Name of the model.

    y_true : array-like
        True target values.

    y_pred : array-like
        Predicted class labels: 0 or 1.

    y_score : array-like, optional
        Continuous model scores used for ROC-AUC.
        These do not have to be probabilities.

    y_proba : array-like, optional
        Predicted probabilities for the positive class.
        Used for log loss and, if y_score is not provided,
        for ROC-AUC.

    Returns
    -------
    dict
        Dictionary containing calculated metrics.
    """

    if y_score is None and y_proba is not None:
        y_score = y_proba

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "roc_auc": (
            roc_auc_score(y_true, y_score)
            if y_score is not None
            else np.nan
        ),
        "log_loss": (
            log_loss(y_true, y_proba)
            if y_proba is not None
            else np.nan
        ),
    }

    print(f"Model: {model_name}\n")

    print(
        classification_report(
            y_true,
            y_pred,
            digits=4,
            zero_division=0,
        )
    )

    for metric_name, metric_value in metrics.items():
        if metric_name != "model" and not np.isnan(metric_value):
            print(f"{metric_name}: {metric_value:.4f}")

    cm = confusion_matrix(y_true, y_pred)

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
    ).plot(values_format="d")

    plt.title(f"Confusion Matrix — {model_name}")
    plt.show()

    return metrics

def save_experiment_result(result_row, results_file):
    """
    Save one model experiment result to a CSV file.

    If a row with the same experiment_id already exists, it is removed
    and replaced with the new result. If the CSV file does not exist,
    a new one is created.

    Parameters
    ----------
    result_row : dict
        Dictionary containing the experiment name, feature description,
        evaluation split, threshold, model name, and calculated metrics.
        It must contain the ``experiment_id`` key.

    results_file : str
        Path to the CSV file where experiment results are stored.

    Returns
    -------
    pandas.DataFrame
        Updated table containing all saved experiment results.
    """
    current_result = pd.DataFrame([result_row])

    os.makedirs(
        os.path.dirname(results_file),
        exist_ok=True,
    )

    if os.path.exists(results_file):
        model_results = pd.read_csv(
            results_file
        )

        model_results = model_results[
            model_results["experiment_id"]
            != result_row["experiment_id"]
        ]

        model_results = pd.concat(
            [
                model_results,
                current_result,
            ],
            ignore_index=True,
        )

    else:
        model_results = current_result.copy()

    model_results.to_csv(
        results_file,
        index=False,
    )

    print(
        f"Results saved to: {results_file}"
    )

    return model_results