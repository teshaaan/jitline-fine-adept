from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "tiny_pr_data.csv"
FEATURE_COLUMNS = [
    "lines_added",
    "files_changed",
    "author_previous_prs",
]
RANDOM_STATE = 42
TEST_SIZE = 0.3


def main():
    # Load dataset
    data = pd.read_csv(DATA_PATH)

    # Features
    X = data[FEATURE_COLUMNS]

    # Label
    y = data["defective"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Tiny training data needs a permissive split threshold.
    model = XGBClassifier(
        n_estimators=50,
        max_depth=2,
        learning_rate=0.1,
        min_child_weight=0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )

    # Train model
    model.fit(X_train, y_train)

    # Predict classes
    predictions = model.predict(X_test)

    print("Actual:")
    print(y_test.to_list())

    print("\nPredicted:")
    print(predictions)

    # Evaluate
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    print("\nPerformance:")
    print("Accuracy:", round(accuracy, 2))
    print("Precision:", round(precision, 2))
    print("Recall:", round(recall, 2))
    print("F1 Score:", round(f1, 2))

    # Risk probabilities
    risk_probabilities = model.predict_proba(X_test)[:, 1]

    print("\nRisk probabilities:")

    for probability in risk_probabilities:
        print(round(float(probability), 2))

    # Feature importance
    print("\nFeature importance:")

    for feature, importance in zip(
        X.columns,
        model.feature_importances_,
    ):
        print(feature, ":", round(float(importance), 3))


if __name__ == "__main__":
    main()
