import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
# from mlxtend.plotting import plot_decision_regions

df = pd.read_csv('cukrzyca.csv')

X = df.iloc[:, :-1]
y = df.outcome
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
params = {
    'max_depth': range(2, 6, 2),
    'n_estimators': range(10, 101, 20),
    'max_features': range(2, 10, 2),
    #'min_samples_split': [2, 3, 4],
    'random_state': [0],
    'criterion': ['gini', 'entropy', 'log_loss']
}

grid = GridSearchCV(model, params, scoring='accuracy', cv=10, verbose=2)
grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)
print(grid.best_estimator_)
print(pd.DataFrame(grid.best_estimator_.feature_importances_,X.columns).sort_values(0, ascending=False))
y_pred = grid.best_estimator_.predict(X_test)
print(pd.DataFrame(confusion_matrix(y_test, y_pred)))