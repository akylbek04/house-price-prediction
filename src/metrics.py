"""Shared evaluation helper."""
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(model, X, y):
    pred = model.predict(X)
    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": mean_squared_error(y, pred) ** 0.5,
        "R2": r2_score(y, pred),
    }
