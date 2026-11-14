"""Random Forest regressor."""
from sklearn.ensemble import RandomForestRegressor

from src.metrics import evaluate
from src.split import load_split

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_split()
    model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42).fit(X_train, y_train)
    print("train:", evaluate(model, X_train, y_train))
    print("test: ", evaluate(model, X_test, y_test))
