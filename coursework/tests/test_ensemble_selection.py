import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from solution.ensemble_selection import GreedyEnsembleSelector


def make_estimators():
    return [
        ("logistic", LogisticRegression(max_iter=1000, random_state=42)),
        ("tree", DecisionTreeClassifier(random_state=42)),
    ]


def test_predict_requires_fit():
    X, _ = load_iris(return_X_y=True)
    selector = GreedyEnsembleSelector(make_estimators())

    with pytest.raises(NotFittedError):
        selector.predict(X)

    with pytest.raises(NotFittedError):
        selector.predict_proba(X)


def test_fit_stores_selected_estimators_and_predictions():
    X, y = load_iris(return_X_y=True)
    selector = GreedyEnsembleSelector(
        make_estimators(),
        cv=5,
        max_ensemble_size=2,
        metric="accuracy",
        random_state=42,
    )

    selector.fit(X, y)

    assert 1 <= len(selector.selected_indices_) <= 2
    assert len(selector.selected_indices_) == len(selector.selected_names_)
    assert len(selector.selected_indices_) == len(selector.selection_scores_)
    assert len(selector.selected_indices_) == len(selector.estimators_)
    assert np.array_equal(selector.classes_, np.unique(y))

    probabilities = selector.predict_proba(X)
    predictions = selector.predict(X)

    assert probabilities.shape == (len(X), len(np.unique(y)))
    assert predictions.shape == (len(X),)
    assert np.all(probabilities >= 0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.all(np.isin(predictions, selector.classes_))


def test_accuracy_selection_scores_do_not_decrease():
    X, y = load_iris(return_X_y=True)
    selector = GreedyEnsembleSelector(
        make_estimators(),
        cv=5,
        max_ensemble_size=None,
        metric="accuracy",
        random_state=42,
    )

    selector.fit(X, y)

    assert all(
        later > earlier
        for earlier, later in zip(
            selector.selection_scores_,
            selector.selection_scores_[1:],
        )
    )


def test_log_loss_selection_scores_strictly_decrease():
    X, y = load_iris(return_X_y=True)
    selector = GreedyEnsembleSelector(
        make_estimators(),
        cv=5,
        max_ensemble_size=2,
        metric="log loss",
        random_state=42,
    )

    selector.fit(X, y)

    assert all(
        later < earlier
        for earlier, later in zip(
            selector.selection_scores_,
            selector.selection_scores_[1:],
        )
    )


def test_invalid_metric_and_ensemble_size_raise():
    X, y = load_iris(return_X_y=True)

    with pytest.raises(ValueError):
        GreedyEnsembleSelector(
            make_estimators(),
            metric="f1",
        ).fit(X, y)

    with pytest.raises(ValueError):
        GreedyEnsembleSelector(
            make_estimators(),
            max_ensemble_size=0,
        ).fit(X, y)