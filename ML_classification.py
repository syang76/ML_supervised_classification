'''
Author S. Yang
This code demonstrates how to construct pipelines to perform grid search
and compare and performances (accuracy scores) of classifiers: 
Logistic Regression, KNN, SVM, and Random Forest. 
Imputer and StandardScalar are used in the pipeline for preprocessing.
'''
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Load dataset
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Define the models and their parameters
models = {
    'Logistic Regression': {'model': LogisticRegression(), 'params': {'model__C': [0.1, 1, 10]} },
    'KNN': {'model': KNeighborsClassifier(), 'params': {'model__n_neighbors': [3, 5, 7]} },
    'SVM': {'model': SVC(), 'params': {'model__C': [0.1, 1, 10], 'model__kernel':['linear', 'rbf']} },
    'Random Forest': {'model': RandomForestClassifier(), 'params': {'model__n_estimators': [10, 50, 100], 'model__max_depth': [None, 5, 10]} }
}
# Create a pipeline with scaling and a classifier
def create_pipeline(model):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

results = []
for name, model in models.items():
    print('='*10, name, 'model params = ',model['params'])
    pipeline = create_pipeline(model['model'])
    # Perform GridSearchCV using KFold
    grid_search = GridSearchCV(pipeline, model['params'], cv=KFold(n_splits=5), scoring='accuracy')
    grid_search.fit(X_train, y_train)
    results.append({'model': name, 'best_score': grid_search.best_score_, 'best_params': grid_search.best_params_})
    # Evaluate the best model using accuracy score
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy of testing dataset = {accuracy:.3f}')
scores = []
labels = []
for result in results:
    print('*'*10,'summary: {}={:.3f}'.format(result['model'],result['best_score']))