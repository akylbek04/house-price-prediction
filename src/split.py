"""Load the dataset and split it into train and test sets."""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import CSV_PATH

TARGET = "MedHouseVal"


def load_split(test_size=0.2, seed=42):
    df = pd.read_csv(CSV_PATH)
    X, y = df.drop(columns=TARGET), df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=seed)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_split()
    print("train:", X_train.shape, "test:", X_test.shape)
