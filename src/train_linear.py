"""Baseline: scaled Linear Regression."""
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.metrics import evaluate
from src.split import load_split

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_split()
    model = make_pipeline(StandardScaler(), LinearRegression()).fit(X_train, y_train)
    print("train:", evaluate(model, X_train, y_train))
    print("test: ", evaluate(model, X_test, y_test))
