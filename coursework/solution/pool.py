"""Helpers for generating out-of-fold probability estimates for a model pool."""

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
import numpy as np


def generate_oof_probas(estimators, X, y, cv=5, random_state=None):
    """Return out-of-fold class probabilities and per-estimator validation scores.

    Parameters
    ----------
    estimators : list[tuple[str, estimator]]
        A list of ``(name, estimator)`` pairs. Each estimator must implement
        ``predict_proba`` and be compatible with scikit-learn's ``clone``.
    X : np.ndarray
        Two-dimensional feature matrix with shape ``(n_samples, n_features)``.
    y : np.ndarray
        One-dimensional target array of length ``n_samples``.
    cv : int, default=5
        Number of stratified folds to use for cross-validation.
    random_state : int or None, default=None
        Random seed used to make the fold splits reproducible.

    Returns
    -------
    probabilities : np.ndarray
        Array of shape ``(n_estimators, n_samples, n_classes)`` containing the
        out-of-fold probability estimates for each estimator.
    scores : np.ndarray
        One-dimensional score array of length ``n_estimators`` containing the
        accuracy of each estimator on its out-of-fold predictions.

    Notes
    -----
    StratifiedKFold is used here instead of a plain K-fold split because the
    coursework datasets are class imbalanced. Preserving class proportions in
    every fold gives a fairer estimate of each estimator's predictive quality.
    """

    # Input validation
    if not isinstance(estimators, list):
        raise TypeError("estimators must be a list.")
    if len(estimators) == 0:
        raise ValueError("estimators must not be empty.")
    if not all(isinstance(item, tuple) and len(item) == 2 for item in estimators):
        raise ValueError("Each estimator must be a (name, estimator) tuple.")
    names = [name for name, _ in estimators]
    if not all(isinstance(name, str) and name for name in names):
        raise ValueError("Each estimator name must be a non-empty string.")
    if len(names) != len(set(names)):
        raise ValueError("Estimator names must be unique.")
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a NumPy array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of cases.")
    if len(y) == 0:
        raise ValueError("X and y must not be empty.")
    if isinstance(cv, (bool, np.bool_)) or not isinstance(cv, (int, np.integer)):
        raise TypeError("cv must be an integer.")
    if cv < 2:
        raise ValueError("cv must be at least 2.")
    class_order = np.unique(y)

    if len(class_order) < 2:
        raise ValueError("y must contain at least two classes.")

    if cv > len(y):
        raise ValueError("cv cannot exceed the number of cases.")

    _, class_counts = np.unique(y, return_counts=True)

    if np.any(class_counts < cv):
        raise ValueError("Each class must have at least cv examples.")
    if isinstance(random_state, (bool, np.bool_)) or not (random_state is None or isinstance(random_state, (int, np.integer))):
        raise TypeError("random_state must be an integer or None.")
    for name, estimator in estimators:
        if not callable(getattr(estimator, "predict_proba", None)):
            raise ValueError(f"Estimator '{name}' must implement predict_proba.")

    #Initialise the probabilities array with zeros
    probabilities = np.zeros((len(estimators), X.shape[0], len(class_order)))
    scores = np.zeros(len(estimators))
    # Create the folds in data using StratifiedKFold to maintain class distribution
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    folds = list(skf.split(X, y))

    for i, (name, estimator) in enumerate(estimators):

        for train, held_out in folds:
            # Clone the estimator to ensure a fresh model for each fold
            model = clone(estimator)
            model.fit(X[train], y[train])

            fold_probas = model.predict_proba(X[held_out])

            for j, class_label in enumerate(model.classes_):
                class_index = np.where(class_order == class_label)[0][0]
                probabilities[i, held_out, class_index] = fold_probas[:, j]

        # Calc accuracy for all OOF predictions for this estimator
        predicted = class_order[np.argmax(probabilities[i], axis=1)]
        scores[i] = np.mean(predicted == y)
    
    return probabilities, scores