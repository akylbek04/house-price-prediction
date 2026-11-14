# House Price Prediction

Predicting median house values in California districts with scikit-learn, and comparing a Linear Regression baseline with a Random Forest.

**Dataset:** California housing (20,640 districts, 8 numeric features, target `MedHouseVal` in $100k), loaded via `sklearn.datasets.fetch_california_housing`.

## Results (20% hold-out test set)

| Model | MAE | RMSE | R2 | 5-fold CV RMSE |
|---|---|---|---|---|
| Linear Regression (scaled) | 0.533 | 0.746 | 0.576 | 0.721 ± 0.011 |
| Random Forest (200 trees) | 0.327 | 0.504 | 0.806 | 0.510 ± 0.006 |

Random Forest cuts RMSE by about 32%. Median income is the strongest predictor (feature importance 0.53).

## Project structure

```
data/        dataset CSV
notebooks/   01_eda.ipynb, 02_model_comparison.ipynb
src/         data download, split, metrics, training scripts
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.data           # download dataset
python -m src.train_linear   # baseline
python -m src.train_forest   # Random Forest
jupyter notebook notebooks/
```

## Possible next steps
- Tune Random Forest hyperparameters (depth, min samples) to reduce overfitting
- Try Gradient Boosting
- Add engineered features (rooms per person, distance to coast)
