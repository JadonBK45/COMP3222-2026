"""Example code for Lab Sheet 1.

This file is the worked example referred to in the lab sheet. It shows

* Task 1: a pure Python Euclidean distance, and the timing/plotting pattern you
  need for the timing experiment.
* Task 2: how to load the two data files in ``data/lab1`` with pandas.

The exercises themselves (the Numba version, DTW, the preprocessing pipeline and
the single feature classifier) are left for you to write.
"""

import math
import time
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import IterativeImputer
from numba import njit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

# Data lives in <repo root>/data/lab1, this file is in <repo root>/code/lab1.
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "lab1"


class OneFeatureClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, feature_index=0):
        self.feature_index = feature_index

    def fit(self, X, y):
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("OneFeatureClassifier requires exactly two classes")

        feature_values = X[:, self.feature_index]
        class_0_mean = np.mean(feature_values[y == classes[0]])
        class_1_mean = np.mean(feature_values[y == classes[1]])

        self.classes_ = classes
        self.threshold_ = np.median(feature_values)
        if class_0_mean < class_1_mean:
            self.less_than_class_ = classes[0]
            self.greater_than_class_ = classes[1]
        else:
            self.less_than_class_ = classes[1]
            self.greater_than_class_ = classes[0]
        return self

    def predict(self, X):
        check_is_fitted(self, ["threshold_", "less_than_class_", "greater_than_class_"])
        feature_values = X[:, self.feature_index]
        return np.where(
            feature_values < self.threshold_,
            self.less_than_class_,
            self.greater_than_class_,
        )


def euclidean_distance_python(x: np.ndarray, y: np.ndarray) -> float:
    sum = 0
    for i in range (len(x)):
        sum += (x[i]-y[i])**2
    return sum**0.5

@njit
def dup_euclidean_distance_python(x: np.ndarray, y: np.ndarray) -> float:
    sum = 0
    for i in range (len(x)):
        sum += (x[i]-y[i])**2
    return sum**0.5


def time_distance(distance_function, n: int, random_state: int = 0) -> float:
    """Time a single call of ``distance_function`` on two random series of length n.

    Parameters
    ----------
    distance_function : callable
        Takes two 1D np.ndarray and returns a float.
    n : int
        Length of the random series to generate.
    random_state : int, default=0
        Seed for the random number generator, so runs are repeatable.

    Returns
    -------
    float
        Time taken in seconds.
    """
    rng = np.random.default_rng(random_state)
    x = rng.random(n)
    y = rng.random(n)
    start = time.perf_counter()
    distance_function(x, y)
    return time.perf_counter() - start


def part1_example(show: bool = True):
    """Time the Python distance for increasing n and plot the result.

    This is the timing and plotting pattern for Task 1.3. It only times the
    pure Python version. Once you have written your ``@njit`` version, time it
    in the same loop and add a second line to the plot, so you can compare the
    two and estimate the speed-up.

    Parameters
    ----------
    show : bool, default=True
        If True, display the plot. The figure is saved either way.

    Returns
    -------
    tuple of list
        The series lengths used and the times taken.
    """
    # Basic usage.
    x = np.arange(1.0, 11.0)
    y = np.arange(2.0, 12.0)
    print("Python:", euclidean_distance_python(x, y))

    # Time for increasing sizes of input.
    sizes = []
    t_py_list = []
    t_njit_list= []
    for n in range(100_000, 1_000_001, 100_000):
        sizes.append(n)
        t_py_list.append(time_distance(euclidean_distance_python, n))
        t_njit_list.append(time_distance(dup_euclidean_distance_python, n))
        print(f"n = {n:>9,}  njit = {t_njit_list[-1]:.6f}s, python = {t_py_list[-1]:.6f}s")

    # --- Plot: time vs n (log-log) ---
    plt.figure(figsize=(7.5, 5.0))
    plt.plot(sizes, t_py_list, marker="o", label="ED Python")
    plt.plot(sizes, t_njit_list, marker="o", label="ED Numba")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("n (log scale)")
    plt.ylabel("Time (seconds, log scale)")
    plt.title("Euclidean distance: time vs n")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("jit_speed_time_vs_n.png", dpi=150)
    if show:
        plt.show()
    return sizes, t_py_list, t_njit_list


def one_hot_example():
    """Load one_hot.csv, the mixed categorical/continuous data used in Task 2A.

    Returns
    -------
    tuple
        ``(X, rank, colour, y)``, the continuous features, the ordinal feature,
        the nominal feature and the class labels.
    """
    path = DATA_DIR / "one_hot.csv"
    df = pd.read_csv(path, header=None, names=["x1", "x2", "x3", "rank", "colour", "y"])
    X = df[["x1", "x2", "x3"]].to_numpy(dtype=float)  # shape (n, 3)
    rank = df["rank"].to_numpy(dtype=int)  # shape (n,)
    colour = df["colour"].to_numpy(dtype=str)  # shape (n,)
    colour_2d = colour.reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False)
    colour_encoded = encoder.fit_transform(colour_2d)
    y = df["y"].to_numpy(dtype=int)  # shape (n,)
    X_reformatted = np.column_stack((X,rank, colour_encoded))
    print(f"Shape:",X_reformatted.shape, "Type:",type(X_reformatted), "Contains NaN:",np.isnan(X_reformatted).any(), "Contains infinity:",np.isinf(X_reformatted).any(), "Labels:",np.unique(y))
    
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_reformatted, y)

    print("Classifier fitted successfully")
    print("Training accuracy:", classifier.score(X_reformatted, y))
    
    return X_reformatted, y


def missing_example():
    """Load missing.csv, the data with missing values used in Task 2B.

    Missing values are either blank or the string ``NaN``, both of which pandas
    reads as ``np.nan``.

    Returns
    -------
    tuple of np.ndarray
        ``(X, y)``, the features (containing NaNs) and the class labels.
    """
    path = DATA_DIR / "missing.csv"
    df = pd.read_csv(path, header=None, names=["x1", "x2", "x3", "y"])
    X = df[["x1", "x2", "x3"]].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=int)
    mean_imp = IterativeImputer(random_state=0)
    X_impute = mean_imp.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_impute)
    print(f"X shape {X_impute.shape}, {np.isnan(X_impute).sum()} missing values")
    print(f"X scaled shape {X_scaled.shape}, {np.isnan(X_scaled).sum()} missing values")
    return X_impute, y

def pipeline_example():
    X, y = missing_example()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", KNeighborsClassifier()),
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(f"Test predictions: {y_pred}")
    print(f"Test accuracy: {pipe.score(X_test, y_test):.3f}")
    return pipe


def custom_classifier_example(feature_index=0):
    X, y = one_hot_example()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    classifier = OneFeatureClassifier(feature_index=feature_index)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(f"Selected feature: {feature_index}")
    print(f"Threshold: {classifier.threshold_:.3f}")
    print(f"Test predictions: {y_pred}")
    print(f"Test accuracy: {np.mean(y_pred == y_test):.3f}")
    return classifier


if __name__ == "__main__":
    custom_classifier_example(feature_index=0)
    # missing_example()
    # part1_example()
    # pipeline_example()
