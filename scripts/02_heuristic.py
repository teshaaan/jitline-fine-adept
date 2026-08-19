from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "tiny_pr_data.csv"
RISK_THRESHOLD = 50


def calculate_risk(row):
    score = 0

    # Rule 1:
    # Very large PRs get 40 risk points.
    if row["lines_added"] > 500:
        score += 40

    # Rule 2:
    # PRs touching many files get 30 risk points.
    if row["files_changed"] > 15:
        score += 30

    # Rule 3:
    # Authors with little previous PR experience get 30 points.
    if row["author_previous_prs"] < 5:
        score += 30

    return score


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def main():
    # Load our historical PR dataset.
    data = pd.read_csv(DATA_PATH)

    # Calculate a risk score for every PR.
    data["risk_score"] = data.apply(calculate_risk, axis=1)

    # We decide that score >= 50 means risky.
    data["prediction"] = (data["risk_score"] >= RISK_THRESHOLD).astype(int)

    print(data)

    actual = data["defective"]
    predicted = data["prediction"]

    true_positive = ((actual == 1) & (predicted == 1)).sum()
    true_negative = ((actual == 0) & (predicted == 0)).sum()
    false_positive = ((actual == 0) & (predicted == 1)).sum()
    false_negative = ((actual == 1) & (predicted == 0)).sum()

    print("\nResults:")
    print("True Positives:", true_positive)
    print("True Negatives:", true_negative)
    print("False Positives:", false_positive)
    print("False Negatives:", false_negative)

    accuracy = safe_divide(true_positive + true_negative, len(data))
    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    print("\nMetrics:")
    print("Accuracy:", round(accuracy, 2))
    print("Precision:", round(precision, 2))
    print("Recall:", round(recall, 2))
    print("F1 Score:", round(f1, 2))


if __name__ == "__main__":
    main()
