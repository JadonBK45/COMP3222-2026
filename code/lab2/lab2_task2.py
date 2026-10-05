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


if __name__ == "__main__":
    X, y = load_playgolf()

    # A. Try fitting DecisionTreeClassifier on X as it is. What error do you get?
    # Then one-hot encode Outlook, keeping Temp, Humidity, and Windy numeric.
    # ColumnTransformer and OneHotEncoder can do this in a Pipeline.
    # Check that the transformed X has shape (14, 6).

    preprocessor = ColumnTransformer(
        transformers=[
            ("Body Temp", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Body Temp"]),
        ],
        remainder="passthrough",
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(criterion="gini", random_state=42)),
        ]
    )

    model.fit(X, y)

    X_transformed = model.named_steps["preprocessor"].transform(X)
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()

    print("Transformed shape:", X_transformed.shape)
    print("Feature names:", feature_names)

    # B. Fit a DecisionTreeClassifier to the numeric features and y.
    # Use sklearn.tree.plot_tree with feature_names and class_names.
    # If you used a Pipeline, plot its fitted tree step, not the Pipeline itself.
    # How does this tree compare with the one from the lecture?

    fitted_tree = model.named_steps["classifier"]
    tree.plot_tree(
        fitted_tree,
        feature_names=feature_names,
        class_names=["Pox = False", "Pox = True"],
    )
    plt.tight_layout()
    plt.show()


    # Try ExtraTreeClassifier as well. Compare the two trees and their
    # training predictions. What would you need to compare them fairly on
    # new data?