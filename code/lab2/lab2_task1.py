import pandas as pd
import numpy as np

def impurity(counts: np.ndarray) -> float:
    """Return Gini impurity for one node given class counts."""
    total = len(counts)
    values = np.unique(counts)
    sum = 0
    for v in values:
        # print(f"Value: {v}, Count: {np.sum(counts == v)}, Total: {total}")
        sum += (np.sum(counts == v)/total) ** 2
    return 1.0 - sum

def gini_gain(attr: np.ndarray, y: np.ndarray) -> float:
    """Compute Gini gain for splitting labels y by categorical attribute attr."""
    # Find impurity of root node
    root_impurity = impurity(y)
    #Split y by the value of attr and find impurity of each split
    values = np.unique(attr)
    # print(f"Values: {values} Counts: {[np.sum(attr == v) for v in values]}")
    weighted_impurity = 0
    for v in values:
        split_y = y[attr == v]
        # print(f"Split: {v} Counts: {len(split_y)} Impurity: {impurity(split_y)}")
        weighted_impurity += (len(split_y)/len(y)) * impurity(split_y)
    return root_impurity - weighted_impurity


def entropy(y: np.ndarray) -> float:
    """Return the entropy of the class labels in one node."""
    if len(y) == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return float(-np.sum(probabilities * np.log2(probabilities)))


def information_gain(attr: np.ndarray, y: np.ndarray) -> float:
    root_entropy = entropy(y)
    weighted_entropy = 0.0

    for value in np.unique(attr):
        split_y = y[attr == value]
        weighted_entropy += (len(split_y) / len(y)) * entropy(split_y)

    return root_entropy - weighted_entropy


def checkGolf():
    data = pd.read_csv('data/lab2/playgolf.csv')
    print(data)
    y = data.iloc[:, 4].to_numpy()
    X = data.iloc[:, 0:4].to_numpy()
    outlook = X[:, 0]
    #Calculate first split values to decide split decision
    print(f"Gini Gain for Outlook: {gini_gain(outlook, y)}")
    temperature = X[:, 1]
    print(f"Gini Gain for Temperature: {gini_gain(temperature, y)}")
    humidity = X[:, 2]
    print(f"Gini Gain for Humidity: {gini_gain(humidity, y)}")
    windy = X[:, 3]
    print(f"Gini Gain for Windy: {gini_gain(windy, y)}")

def checkPoxGini():
    data = pd.read_csv('data/lab2/chickenPox.csv')
    print(data)
    y = data.iloc[:, 3].to_numpy()
    X = data.iloc[:, 0:3].to_numpy()
    body_temp = X[:, 0]
    #Calculate first split values to decide split decision
    print(f"Gini Gain for Body Temperature: {gini_gain(body_temp, y)}")
    stomach_ache = X[:, 1]
    print(f"Gini Gain for Stomach Ache: {gini_gain(stomach_ache, y)}")
    spots = X[:, 2]
    print(f"Gini Gain for Spots: {gini_gain(spots, y)}")

def checkPoxEntropy():
    data = pd.read_csv('data/lab2/chickenPox.csv')
    print(data)
    y = data.iloc[:, 3].to_numpy()
    X = data.iloc[:, 0:3].to_numpy()
    body_temp = X[:, 0]
    #Calculate first split values to decide split decision
    print(f"Information Gain for Body Temperature: {information_gain(body_temp, y)}")
    stomach_ache = X[:, 1]
    print(f"Information Gain for Stomach Ache: {information_gain(stomach_ache, y)}")
    spots = X[:, 2]
    print(f"Information Gain for Spots: {information_gain(spots, y)}")


if __name__ == "__main__":    # Example usage
    # print(f"Gini Impurity: {impurity}")
    data = pd.read_csv('data/lab2/chickenPox.csv')
    print(data)
    y = data.iloc[:, 3].to_numpy()
    X = data.iloc[:, 0:3].to_numpy()
    body_temp = X[:, 0]
    stomach_ache = X[:, 1]
    spots = X[:, 2]
    print(f"Entropy of the labels: {entropy(y)}")
    print(f"Information Gain for Spots: {information_gain(spots, y)}")
    print(f"Information Gain for Stomach Ache: {information_gain(stomach_ache, y)}")
    print(f"Information Gain for Body Temperature: {information_gain(body_temp, y)}")


