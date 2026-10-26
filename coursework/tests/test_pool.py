import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from solution.pool import generate_oof_probas


class TrainingMembershipClassifier(BaseEstimator, ClassifierMixin):
    """Return class 1 for cases seen during fit and class 0 otherwise."""

    def fit(self, X, y):
        self.seen_values_ = set(X[:, 0])
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return np.asarray([
            [0.0, 1.0] if row[0] in self.seen_values_ else [1.0, 0.0]
            for row in X
        ])


def test_output_shapes():
    X, y = load_iris(return_X_y=True)
    estimators = [
        ("first", LogisticRegression(max_iter=1000)),
        ("second", LogisticRegression(max_iter=1000)),
    ]

    oof_probas, scores = generate_oof_probas(
        estimators, X, y, cv=5, random_state=42
    )

    assert oof_probas.shape == (2, 150, 3)
    assert scores.shape == (2,)


def test_probabilities_are_valid():
    X, y = load_iris(return_X_y=True)

    oof_probas, _ = generate_oof_probas(
        [("logistic", LogisticRegression(max_iter=1000))],
        X,
        y,
        cv=5,
        random_state=42,
    )

    assert np.all(oof_probas >= 0)
    assert np.all(oof_probas <= 1)
    assert np.allclose(oof_probas.sum(axis=2), 1.0)


def test_scores_match_oof_predictions():
    X, y = load_iris(return_X_y=True)

    oof_probas, scores = generate_oof_probas(
        [("logistic", LogisticRegression(max_iter=1000))],
        X,
        y,
        cv=5,
        random_state=42,
    )

    classes = np.unique(y)
    predicted = classes[np.argmax(oof_probas[0], axis=1)]
    expected_score = np.mean(predicted == y)

    assert scores[0] == expected_score


def test_predictions_are_genuinely_out_of_fold():
    X = np.arange(12).reshape(-1, 1)
    y = np.array([0, 1] * 6)

    oof_probas, _ = generate_oof_probas(
        [("membership", TrainingMembershipClassifier())],
        X,
        y,
        cv=3,
        random_state=42,
    )

    assert np.all(oof_probas[0, :, 0] == 1.0)
    assert np.all(oof_probas[0, :, 1] == 0.0)


def test_random_state_is_reproducible():
    X, y = load_iris(return_X_y=True)
    estimators = [("logistic", LogisticRegression(max_iter=1000))]

    first = generate_oof_probas(
        estimators, X, y, cv=5, random_state=123
    )
    second = generate_oof_probas(
        estimators, X, y, cv=5, random_state=123
    )

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])