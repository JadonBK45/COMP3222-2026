import csv
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score

from part3_setup import (CV, DATASETS, RANDOM_STATE, load_dataset, make_pool, make_stacking)

from .ensemble_selection import GreedyEnsembleSelector
from .pool import generate_oof_probas

def best_single_classifier(X_train, y_train, X_test, y_test):
    pool = make_pool()
    _, scores = generate_oof_probas(pool, X_train, y_train, cv=CV, random_state=RANDOM_STATE)
    best_index = int(np.argmax(scores))
    fitted = clone(pool[best_index][1]).fit(X_train, y_train).predict(X_test)
    return accuracy_score(y_test, fitted)

def full_pool_ensemble(X_train, y_train, X_test, y_test):
    pool = make_pool()
    classes = np.unique(y_train)
    probas = np.zeros((len(pool), X_test.shape[0], len(classes)))
    for i, (name, estimator) in enumerate(pool):
        fitted = clone(estimator).fit(X_train, y_train)
        fitted_probas = fitted.predict_proba(X_test)
        for column, class_label in enumerate(fitted.classes_):
            class_index = np.where(classes == class_label)[0][0]
            probas[i, :, class_index] = fitted_probas[:, column]
    predictions = classes[np.argmax(probas.mean(axis=0), axis=1)]
    return accuracy_score(y_test, predictions)

def stacking_ensemble(X_train, y_train, X_test, y_test):
    stacking = make_stacking().fit(X_train, y_train)
    return accuracy_score(y_test, stacking.predict(X_test))


def greedy_accuracy_ensemble(X_train, y_train, X_test, y_test, max_ensemble_size=None):
    greedy_accuracy = GreedyEnsembleSelector(make_pool(), cv=CV, max_ensemble_size=max_ensemble_size, metric="accuracy", random_state=RANDOM_STATE).fit(X_train, y_train)
    return (
        accuracy_score(y_test, greedy_accuracy.predict(X_test)),
        len(greedy_accuracy.selected_indices_),
        greedy_accuracy.selection_scores_[-1],
    )

def greedy_log_loss_ensemble(X_train, y_train, X_test, y_test, max_ensemble_size=None):
    greedy_log_loss = GreedyEnsembleSelector(make_pool(),cv=CV,max_ensemble_size=max_ensemble_size,metric="log loss",random_state=RANDOM_STATE).fit(X_train, y_train)
    return accuracy_score(y_test, greedy_log_loss.predict(X_test)), len(greedy_log_loss.selected_indices_)


def main():
    fieldnames = [
        "dataset",
        "best_single",
        "full_pool",
        "stacking",
        "greedy_accuracy",
        "greedy_log_loss",
        "n_selected_accuracy",
        "n_selected_log_loss",
        "oof_accuracy",
    ]
    output_path = Path("results.csv")
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for dataset in DATASETS:
            X_train, y_train, X_test, y_test = load_dataset(dataset)
            greedy_accuracy = greedy_accuracy_ensemble(X_train, y_train, X_test, y_test)
            greedy_log_loss = greedy_log_loss_ensemble(X_train, y_train, X_test, y_test)
            row = {
            "dataset": dataset,
            "best_single": f"{best_single_classifier(X_train, y_train, X_test, y_test):.4f}",
            "full_pool": f"{full_pool_ensemble(X_train, y_train, X_test, y_test):.4f}",
            "stacking": f"{stacking_ensemble(X_train, y_train, X_test, y_test):.4f}",
            "greedy_accuracy": f"{greedy_accuracy[0]:.4f}",
            "greedy_log_loss": f"{greedy_log_loss[0]:.4f}",
            "n_selected_accuracy": greedy_accuracy[1],
            "n_selected_log_loss": greedy_log_loss[1],
            "oof_accuracy": f"{greedy_accuracy[2]:.4f}",
        }
            writer.writerow(row)

if __name__ == "__main__":
    main()