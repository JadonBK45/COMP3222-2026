import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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


def impact_encode(x: np.ndarray, y:np.ndarray) -> np.ndarray:
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
    np.ndarray
        A 1D numpy array where each category is replaced by the mean of the
        target variable for that category.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    # Split cases based on Class 1 or Class 0
    sums_counts = {}
    for cat, yv in zip(x, y):
        s, cc = sums_counts.get(cat, (0.0, 0))
        sums_counts[cat] = (s + yv, cc + 1)
    print("sums_counts:", sums_counts)
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
    """
    Frequency-encode any non-numeric (string/bytes/mixed) columns.
    Numeric columns pass through unchanged. No NaN handling.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


if __name__ == "__main__":
    X, y = load_credit_approval_data("../../data/lab3/credit_approval.csv")
    # print(X.shape, y.shape)
    # print("Unique target values:", set(y))
    # print(X[1])
    # print(X.shape[1])
    # print(len(X[:, 0]))
    print(impact_encode(X[:, 3], y))
    # freq = FrequencyEncoderTransformer()
    # X2 = freq.fit_transform(X)
    # print(X2[0])

