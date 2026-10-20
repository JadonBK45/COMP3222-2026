"""Lab 2, Task 2 starter: fit and visualise a decision tree on PlayGolf.

Run from the repository root with: python code/lab2/lab2_task2.py
Complete the steps below using Lab Sheet 2. The data-loading step is provided.
"""

from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "lab2" / "chickenPox.csv"


def load_playgolf():
    """Return the four attributes as a DataFrame and the labels as an array."""
    data = pd.read_csv(DATA_PATH)
    X = data.drop(columns="Chicken Pox")
    y = data["Chicken Pox"].to_numpy()
    return X, y


def make_model(criterion):
    preprocessor = ColumnTransformer(
        transformers=[
            ("Body Temp", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Body Temp"]),
        ],
        remainder="passthrough",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(criterion=criterion, random_state=42)),
        ]
    )


if __name__ == "__main__":
    X, y = load_playgolf()

    # A. Try fitting DecisionTreeClassifier on X as it is. What error do you get?
    # Then one-hot encode Outlook, keeping Temp, Humidity, and Windy numeric.
    # ColumnTransformer and OneHotEncoder can do this in a Pipeline.
    # Check that the transformed X has shape (14, 6).

    gini_model = make_model("gini")
    entropy_model = make_model("entropy")
    gini_model.fit(X, y)
    entropy_model.fit(X, y)

    X_transformed = gini_model.named_steps["preprocessor"].transform(X)
    feature_names = gini_model.named_steps["preprocessor"].get_feature_names_out()

    print("Transformed shape:", X_transformed.shape)
    print("Feature names:", feature_names)

    # B. Fit a DecisionTreeClassifier to the numeric features and y.
    # Use sklearn.tree.plot_tree with feature_names and class_names.
    # If you used a Pipeline, plot its fitted tree step, not the Pipeline itself.
    # How does this tree compare with the one from the lecture?

    class_names = ["Pox = False", "Pox = True"]
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    tree.plot_tree(
        gini_model.named_steps["classifier"],
        feature_names=feature_names,
        class_names=class_names,
        ax=axes[0],
    )
    axes[0].set_title("Gini")
    tree.plot_tree(
        entropy_model.named_steps["classifier"],
        feature_names=feature_names,
        class_names=class_names,
        ax=axes[1],
    )
    axes[1].set_title("Entropy")
    plt.tight_layout()
    plt.show()


    # Try ExtraTreeClassifier as well. Compare the two trees and their
    # training predictions. What would you need to compare them fairly on
    # new data?
"""Lab 2, Task 2 starter: fit and visualise a decision tree on PlayGolf.

Run from the repository root with: python code/lab2/lab2_task2.py
Complete the steps below using Lab Sheet 2. The data-loading step is provided.
"""

from pathlib import Path

import pandas as pd


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "lab2" / "playgolf.csv"


def load_playgolf():
    """Return the four attributes as a DataFrame and the labels as an array."""
    data = pd.read_csv(DATA_PATH)
    X = data.drop(columns="PlayGolf")
    y = data["PlayGolf"].to_numpy()
    return X, y


if __name__ == "__main__":
    X, y = load_playgolf()
    print("Attributes:", list(X.columns))
    print("Raw X shape:", X.shape, "y shape:", y.shape)
    print("Outlook values:", X["Outlook"].unique())

    # A. Try fitting DecisionTreeClassifier on X as it is. What error do you get?
    # Then one-hot encode Outlook, keeping Temp, Humidity, and Windy numeric.
    # ColumnTransformer and OneHotEncoder can do this in a Pipeline.
    # Check that the transformed X has shape (14, 6).

    # B. Fit a DecisionTreeClassifier to the numeric features and y.
    # Use sklearn.tree.plot_tree with feature_names and class_names.
    # If you used a Pipeline, plot its fitted tree step, not the Pipeline itself.
    # How does this tree compare with the one from the lecture?

    # Try ExtraTreeClassifier as well. Compare the two trees and their
    # training predictions. What would you need to compare them fairly on
    # new data?
