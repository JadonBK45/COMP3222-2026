"""Lab 3, Task 3: a simple majority-vote ensemble (student starter)."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeClassifier


class BasicEnsembleClassifier(ClassifierMixin, BaseEstimator):
    """Fit cloned classifiers on random subsets and combine their votes.

    If base_estimator is None, use DecisionTreeClassifier(max_depth=5).
    With replace=False, each member samples without replacement; True gives
    bootstrap samples. A tied vote should choose the smallest class label.
    """

    def __init__(self, base_estimator=None, n_estimators=10,
                 sample_fraction=0.6, replace=False, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.sample_fraction = sample_fraction
        self.replace = replace
        self.random_state = random_state

    def fit(self, X, y):
        # TODO: Use np.random.default_rng(self.random_state).
        rng = np.random.default_rng(self.random_state)
        # TODO: Choose the default tree if base_estimator is None.
        if self.base_estimator is None:
            self.base_estimator = DecisionTreeClassifier(max_depth=5)
        # TODO: For each member, clone the base estimator, draw row indices,
        # fit it on the sampled rows and store it in self.estimators_.
        self.estimators_ = []
        for _ in range(self.n_estimators):
            m = max(1, int(round(self.sample_fraction * len(X))))
            selected = rng.choice(len(X), size=m, replace=self.replace)
            member = clone(self.base_estimator)
            member.fit(X[selected], y[selected])
            self.estimators_.append(member)
        return self

    def predict(self, X):
        # TODO: Collect each member's predictions, then vote row by row.
        # Use np.bincount to count votes and np.argmax to find the winner.
        predictions = np.array([member.predict(X) for member in self.estimators_])
        return np.array([np.argmax(np.bincount(row)) for row in predictions.T])

    # Optional extension: add predict_proba by averaging member probabilities.


if __name__ == "__main__":
    from lab3_task1 import load_credit_approval_data
    from lab3_task1 import FrequencyEncoderTransformer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X, y = load_credit_approval_data(
        "../../data/lab3/credit_approval.csv"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Learn frequency mappings from training data only.
    encoder = FrequencyEncoderTransformer()

    X_train_encoded = encoder.fit_transform(X_train)
    X_test_encoded = encoder.transform(X_test)

    # Create and train the ensemble.
    ensemble = BasicEnsembleClassifier(
        n_estimators=10,
        sample_fraction=0.6,
        replace=False,
        random_state=42
    )

    ensemble.fit(X_train_encoded, y_train)

    # Generate predictions for the test data.
    y_pred = ensemble.predict(X_test_encoded)

    # Display actual and predicted values.
    print("Real values:   ", y_test)
    print("Predictions:   ", y_pred)

    # Calculate percentage accuracy.
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2%}")