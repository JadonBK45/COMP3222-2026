"""Lab 3, Task 2: compare regression models (student starter)."""
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


import pandas as pd
from sklearn.model_selection import train_test_split


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "lab3" / "winequality-red.csv"


def load_wine_quality_data(path=DATA_PATH):
    """Return wine measurements X and quality scores y as NumPy arrays."""
    data = pd.read_csv(path, sep=";")
    X = data.drop(columns="quality").to_numpy()
    y = data["quality"].to_numpy()
    return X, y


if __name__ == "__main__":
    X, y = load_wine_quality_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=42
    )
    print(f"Wine quality: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"Training: {len(y_train)} rows; test: {len(y_test)} rows")
    print()
    # TODO: Fit LinearRegression on X_train, y_train.
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    # TODO: Print mean_squared_error for training and test predictions.
    y_train_pred = lr.predict(X_train)
    y_test_pred = lr.predict(X_test)
    TrainMSE = mean_squared_error(y_train, y_train_pred)
    TestMSE = mean_squared_error(y_test, y_test_pred)
    print(f"Training MSE: {TrainMSE:.4f}")
    print(f"Test MSE: {TestMSE:.4f}")
    print()
    # TODO: Repeat for DecisionTreeRegressor(max_depth=3, random_state=42).
    dt = DecisionTreeRegressor(max_depth=3, random_state=42)
    dt.fit(X_train, y_train)
    y_train_pred_dt = dt.predict(X_train)
    y_test_pred_dt = dt.predict(X_test)
    TrainMSE_dt = mean_squared_error(y_train, y_train_pred_dt)
    TestMSE_dt = mean_squared_error(y_test, y_test_pred_dt)
    print(f"Training MSE (Decision Tree): {TrainMSE_dt:.4f}")
    print(f"Test MSE (Decision Tree): {TestMSE_dt:.4f}")
    print()
    # TODO: Try several tree depths, then RandomForestRegressor(random_state=42).
    dt_depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for depth in dt_depths:
        dt = DecisionTreeRegressor(max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)
        y_test_pred_dt = dt.predict(X_test)
        TestMSE_dt = mean_squared_error(y_test, y_test_pred_dt)
        print(f"Test MSE (Decision Tree, depth={depth}): {TestMSE_dt:.4f}")
    print()
    # TODO: Fit RandomForestRegressor with 100 trees.
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_test_pred_rf = rf.predict(X_test)
    TestMSE_rf = mean_squared_error(y_test, y_test_pred_rf)
    print(f"Training MSE (Random Forest): {mean_squared_error(y_train, rf.predict(X_train)):.4f}")
    print(f"Test MSE (Random Forest): {TestMSE_rf:.4f}")
