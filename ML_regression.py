'''
Author S. Yang
This code demonstrates how to construct pipelines to perform grid search with 
5-split KFold to compare performances (mean_squared_error) of regressors: 
Linear Regression, Decision tree, Random Forest, Gradient boosting.
Imputer and StandardScalar are used in the pipeline for preprocessing.
'''

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models and their hyperparameters
models = [
    ('Linear Regression', LinearRegression(), {}),
    ('Decision Tree', DecisionTreeRegressor(), {'max_depth': [2, 5, 10]}),
    ('Random Forest', RandomForestRegressor(), {'n_estimators': [10, 20, 50]}),
    ('Gradient Boosting', GradientBoostingRegressor(), {'n_estimators': [50, 100, 200]})
]

# Create a pipeline with scaling and a classifier
def create_pipeline(model):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

# Create a KFold object for cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform GridSearchCV for each model
for name, model, param_grid in models:
    pipeline = create_pipeline(model)
    grid_search = GridSearchCV(pipeline, 
        param_grid={'model__' + k: v for k, v in param_grid.items()},
        cv=kf,
        scoring='neg_mean_squared_error'
    )
    grid_search.fit(X_train, y_train)
    # Print the best parameters and score
    print('='*10,name)
    print(f'Best parameters: {grid_search.best_params_}')
    print(f'Best score: {-grid_search.best_score_}')
    # Evaluate the best model using accuracy score
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'MSE of testing dataset = {mse:.3f}')
