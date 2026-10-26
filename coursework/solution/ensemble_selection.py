from sklearn.base import ClassifierMixin, BaseEstimator, clone
from .pool import generate_oof_probas
import numpy as np
from sklearn.metrics import log_loss
from sklearn.exceptions import NotFittedError

class GreedyEnsembleSelector(ClassifierMixin, BaseEstimator):
    def __init__(self, estimators, cv=5, max_ensemble_size=None, metric="accuracy", random_state=None):
        self.estimators = estimators
        self.cv = cv
        self.max_ensemble_size = max_ensemble_size
        self.metric = metric
        self.random_state = random_state

    def fit(self, X, y):
        # Metric validation
        if self.metric not in ("accuracy", "log loss"):
            raise ValueError(f"Unsupported metric: {self.metric}")
        # max_ensemle size validation
        if (self.max_ensemble_size is not None and (
                isinstance(self.max_ensemble_size, (bool, np.bool_))
                or not isinstance(self.max_ensemble_size, (int, np.integer))
                or self.max_ensemble_size <= 0
            )
        ):
            raise ValueError(
                "max_ensemble_size must be None or a positive integer."
            )
        # Maintain reference to original estimator pool, in case fit is called multiple times
        pool_estimators = self.estimators

        classes = np.unique(y)

        # Generate out-of-fold probabilities for each estimator in the pool
        oof_probas, accuracy_scores = generate_oof_probas(pool_estimators, X, y, cv=self.cv, random_state=self.random_state)

        if self.metric == "accuracy":
            first_index = int(np.argmax(accuracy_scores))
            current_score = float(accuracy_scores[first_index])
        else:
            individual_log_losses = np.array([log_loss(y, probabilities, labels=classes) for probabilities in oof_probas])
            first_index = int(np.argmin(individual_log_losses))
            current_score = float(individual_log_losses[first_index])

        selected_indices = [first_index]
        selection_scores = [current_score]
        remaining_indices = [index for index in range(len(pool_estimators)) if index != first_index]

        # Continue adding estimators while selection is sitll possible
        while remaining_indices and (self.max_ensemble_size is None or len(selected_indices) < self.max_ensemble_size):
            best_candidate_index = None
            if self.metric == "accuracy":
                best_candidate_score = -np.inf
            else:
                best_candidate_score = np.inf

            for candidate_index in remaining_indices:
                candidate_indices = selected_indices + [candidate_index]
                averaged_probas = np.mean(oof_probas[candidate_indices], axis=0)
                if self.metric == "accuracy":
                    predictions = classes[np.argmax(averaged_probas, axis=1)]
                    candidate_score = float(np.mean(predictions == y))
                    is_better = candidate_score > best_candidate_score
                else:
                    candidate_score = float(log_loss(y, averaged_probas, labels=classes))
                    is_better = candidate_score < best_candidate_score

                if is_better:
                    best_candidate_index = candidate_index
                    best_candidate_score = candidate_score

            improves = (
                best_candidate_score > current_score
                if self.metric == "accuracy"
                else best_candidate_score < current_score
            )
            if not improves:
                break

            selected_indices.append(best_candidate_index)
            remaining_indices.remove(best_candidate_index)
            current_score = best_candidate_score
            selection_scores.append(current_score)

        # Store fitted attributes
        self.classes_ = classes
        self.selected_indices_ = selected_indices
        self.selected_names_ = [pool_estimators[index][0] for index in selected_indices]
        self.selection_scores_ = selection_scores
        self.estimators_ = []

        # Clone selected estimators and fit them on the entire dataset
        for index in selected_indices:
            estimator = clone(pool_estimators[index][1])
            estimator.fit(X, y)
            self.estimators_.append(estimator)

        return self

    def predict_proba(self, X):
        # Check if the model has been fitted
        if not hasattr(self, "selected_indices_"):
            raise NotFittedError("This GreedyEnsembleSelector instance is not fitted yet.")

        probabilities = []
        for estimator in self.estimators_:
            estimator_probas = estimator.predict_proba(X)
            aligned = np.zeros((len(X), len(self.classes_)))
            for column, class_label in enumerate(estimator.classes_):
                class_index = np.where(self.classes_ == class_label)[0][0]
                aligned[:, class_index] = estimator_probas[:, column]
            probabilities.append(aligned)
        return np.mean(probabilities, axis=0)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]