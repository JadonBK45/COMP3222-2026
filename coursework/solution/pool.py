from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
import numpy as np
def generate_oof_probas(estimators,X,y,cv=5,random_state=None):

    # Input validation
    names = [name for name, _ in estimators]
    class_order = np.unique(y)
    _, class_counts = np.unique(y, return_counts=True)
    
    if not isinstance(estimators, list):
        raise TypeError("estimators must be a list.")
    if len(estimators) == 0:
        raise ValueError("estimators must not be empty.")
    if not all(isinstance(item, tuple) and len(item) == 2 for item in estimators):
        raise ValueError("Each estimator must be a (name, estimator) tuple.")
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
    if len(class_order) < 2:
        raise ValueError("y must contain at least two classes.")
    if cv > len(y):
        raise ValueError("cv cannot exceed the number of cases.")
    if np.any(class_counts < cv):
        raise ValueError("Each class must have at least cv examples.")
    if isinstance(random_state, (bool, np.bool_)) or not (random_state is None or isinstance(random_state, (int, np.integer))):
        raise TypeError("random_state must be an integer or None.")
    for name, estimator in estimators:
        if not callable(getattr(estimator, "predict_proba", None)):
            raise ValueError(f"Estimator '{name}' must implement predict_proba.")

    #Initialise the probabilities array with zeros
    class_order = np.unique(y)
    probabilities = np.zeros((len(estimators), X.shape[0], len(np.unique(y))))
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

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    X, y = load_iris(return_X_y=True)
    estimators = [("Random Forest", RandomForestClassifier()), ("Logistic Regression", LogisticRegression(max_iter=1000))]
    oof_probas, oof_scores = generate_oof_probas(estimators, X, y, cv=5, random_state=42)
    print(oof_probas.shape)  # Should be (2, 150, 3) for the iris dataset with 2 classifiers and 3 classes
    print(oof_scores.shape)  # Should be (2,) for the iris dataset with 2 classifiers
    print(oof_scores)  # Print the accuracy scores for each estimator