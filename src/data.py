"""Download the California housing dataset and save it as a CSV."""
from pathlib import Path

from sklearn.datasets import fetch_california_housing

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "california_housing.csv"


def download(path=CSV_PATH):
    df = fetch_california_housing(as_frame=True).frame  # features + MedHouseVal target
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    df = download()
    print(f"Saved {len(df)} rows x {df.shape[1]} columns to {CSV_PATH}")
