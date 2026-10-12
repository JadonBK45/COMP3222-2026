import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def frequency_encode(x: np.ndarray) -> np.ndarray:
    """Replace each category with its frequency in the array.

    Missing values are treated as one category.
    """
    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError("x must be a 1D array")

    if x.size == 0:
        return np.array([], dtype=float)

    missing = np.array([pd.isna(value) for value in x], dtype=bool)
    counts = {}
    missing_key = object()

    for value, is_missing in zip(x, missing):
        key = missing_key if is_missing else value
        counts[key] = counts.get(key, 0) + 1

    encoded = []
    for value, is_missing in zip(x, missing):
        key = missing_key if is_missing else value
        encoded.append(counts[key] / x.size)

    return np.asarray(encoded, dtype=float)


def impact_encode(x: np.ndarray, y: np.ndarray) -> list[float]:
    """Perform impact encoding on a 1D numpy array.

    Each string value is replaced by the frequency of class 1 in y.
    Parameters
    ----------
    x : np.ndarray
        A 1D numpy array of categorical values (strings).
    y : np.ndarray
        A 1D numpy array of numerical target values.

    Returns
    -------
    list[float]
        A list containing the frequency of each value within its target class,
        in the original order of ``x`` and ``y``.
    """

    # Split cases based on Class 1 or Class 0
    x = x.astype(str)
    value_by_target = {}
    counts = set()

    for value, target in zip(x,y):
        value_by_target.setdefault(target, []).append(value)

    for target, values in value_by_target.items():
        frequencies = frequency_encode(np.asarray(values, dtype=object))
        counts.update(
            (value, target, float(frequency))
            for value, frequency in zip(values, frequencies)
        )

    frequency_lookup = {
        (value, target): frequency
        for value, target, frequency in counts
    }

    X = [
        frequency_lookup[(value, target)]
        for value, target in zip(x, y)
    ]

    return X

def check_functions():
    # Outlook / Play Golf data from the image (14 rows)
    outlook = np.array([
        "sunny", "sunny", "overcast", "rain", "rain", "rain",
        "overcast", "sunny", "sunny", "rain", "sunny", "overcast",
        "overcast", "rain"
    ], dtype=object)

    play_golf = np.array([
        0, 0, 0, 0, 0, 1,
        1, 1, 1, 1, 1, 1,
        1, 1
    ], dtype=int)

    # Run encoders
    freq_encoded = frequency_encode(outlook)
    impact_encoded = impact_encode(outlook, play_golf)

    # Pretty-print results
    print("Original  |  y  |  Freq-enc  |  Impact-enc (mean y per category)")
    print("-" * 64)
    for o, yv, f, ie in zip(outlook, play_golf, freq_encoded, impact_encoded):
        print(f"{o:9s} | {yv:2d} | {f:9.3f} | {ie:9.3f}")

    # Show learned mappings for clarity
    # Frequency map
    u, c = np.unique(outlook, return_counts=True)
    freq_map = {ui: ci / len(outlook) for ui, ci in zip(u, c)}
    # Impact map
    sums_counts = {}
    for cat, yv in zip(outlook, play_golf):
        s, cc = sums_counts.get(cat, (0.0, 0))
        sums_counts[cat] = (s + yv, cc + 1)
    impact_map = {cat: s / cc for cat, (s, cc) in sums_counts.items()}

    print("\nFrequency map:", freq_map)
    print("Impact map (mean(y) per category):", impact_map)


def load_credit_approval_data(path):
    """
    Load a CSV into a NumPy array with mixed types (floats + strings).
    Assumes the target column is named `target`.

    Parameters
    ----------
    path : str | PathLike
        CSV file path.

    Returns
    -------
    np.ndarray                      # if target is None
    or (np.ndarray, np.ndarray)     # if target provided
    """
    df = pd.read_csv(path)
    y = df["target"].to_numpy()
    X = df.drop(columns=["target"]).to_numpy(dtype=object)
    return X, y

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# def frequency_encode(x: np.ndarray) -> np.ndarray:
#     x = np.asarray(x)
#     uniques, counts = np.unique(x, return_counts=True)
#     freqs = counts.astype(float) / x.size
#     lookup = {u: f for u, f in zip(uniques, freqs)}
#     out = np.empty(x.size, dtype=float)
#     for i in range(x.size):
#         out[i] = lookup[x[i]]
#     return out


class FrequencyEncoderTransformer(BaseEstimator, TransformerMixin):
    """Frequency-encode categorical columns and preserve numeric columns."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=object)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        self.numeric_columns_ = []
        self.frequency_maps_ = []

        for column in range(X.shape[1]):
            values = X[:, column]

            if all(
                pd.isna(value) or isinstance(value, (int, float, np.number))
                for value in values
            ):
                self.numeric_columns_.append(True)
                self.frequency_maps_.append(None)
            else:
                self.numeric_columns_.append(False)

                non_missing_values = [
                    value for value in values if not pd.isna(value)
                ]

                frequencies = frequency_encode(
                    np.asarray(non_missing_values, dtype=object)
                )

                mapping = {
                    value: float(frequency)
                    for value, frequency
                    in zip(non_missing_values, frequencies)
                }

                self.frequency_maps_.append(mapping)

        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=object)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of columns than the fitted data")

        transformed = np.empty(X.shape, dtype=float)

        for column in range(X.shape[1]):
            values = X[:, column]

            if self.numeric_columns_[column]:
                transformed[:, column] = values.astype(float)
            else:
                mapping = self.frequency_maps_[column]

                transformed[:, column] = [
                    0.0 if pd.isna(value) else mapping.get(value, 0.0)
                    for value in values
                ]

        return transformed

if __name__ == "__main__":
    X, y = load_credit_approval_data("../../data/lab3/credit_approval.csv")
    # check_functions()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    freq = FrequencyEncoderTransformer()
    X_train_encoded = freq.fit_transform(X_train)
    X_test_encoded = freq.transform(X_test)
    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(X_train_encoded, y_train)
    y_pred = tree.predict(X_test_encoded)

    print("Real values: ", y_test)
    print("Predictions: ", y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2%}")
