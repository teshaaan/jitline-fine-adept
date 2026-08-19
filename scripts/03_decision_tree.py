from pathlib import Path

import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "tiny_pr_data.csv"
FEATURE_COLUMNS = [
    "lines_added",
    "files_changed",
    "author_previous_prs",
]
RANDOM_STATE = 42
TEST_SIZE = 0.3


def main():
    # Load the dataset
    data = pd.read_csv(DATA_PATH)

    # Features: information the model can use
    X = data[FEATURE_COLUMNS]

    # Label: what we want to predict
    y = data["defective"]

    print("Features:")
    print(X)

    print("\nLabels:")
    print(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTraining features:")
    print(X_train)

    print("\nTesting features:")
    print(X_test)

    model = DecisionTreeClassifier(
        max_depth=3,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nActual answers:")
    print(y_test.to_list())

    print("Predictions:")
    print(predictions)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    print("\nModel performance:")
    print("Accuracy:", round(accuracy, 2))
    print("Precision:", round(precision, 2))
    print("Recall:", round(recall, 2))
    print("F1 Score:", round(f1, 2))

    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nProbabilities:")

    for probability in probabilities:
        print(round(float(probability), 2))


if __name__ == "__main__":
    main()
