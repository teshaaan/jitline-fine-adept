from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "tiny_pr_data.csv"


def main():
    data = pd.read_csv(DATA_PATH)

    print("Dataset:")
    print(data)

    print("\nNumber of rows and columns:")
    print(data.shape)

    print("\nColumn names:")
    print(data.columns.to_list())

    print("\nHow many defective and non-defective PRs?")
    print(data["defective"].value_counts())


if __name__ == "__main__":
    main()
