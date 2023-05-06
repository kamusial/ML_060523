import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('heart.csv', comment='#')
print(df.head(10).to_string())
print(df.target.value_counts())
X = df.iloc[:,:-1]
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def drzewo(max_depth):
    print('\nMax depth wynosi ',max_depth)
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))
    print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))

drzewo(2)
drzewo(3)
drzewo(5)
drzewo(10)

print('\nLogistyczna regresja')
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))

import joblib
joblib.dump(model, 'Log_reg_v2.3.model')
model_kamila = joblib.load('Log_reg_v2.3.model')

print(dir(model_kamila))
print(model_kamila._estimator_type, model_kamila.get_params)


model = DecisionTreeClassifier()
params = {
    'max_depth': range(2, 15),
    'max_features': range(2, 14),
    'min_samples_split': [2, 3, 4, 5, 6],
    'random_state': [0],
    'criterion': ['gini', 'entropy', 'log_loss']
}

# grid = GridSearchCV(model, params, scoring='accuracy', cv=10, verbose=1)
# grid.fit(X_train, y_train)
#
# print(grid.best_params_)
# print(grid.best_score_)
# print(grid.best_estimator_)
# print(pd.DataFrame(grid.best_estimator_.feature_importances_,X.columns).sort_values(0, ascending=False))
# y_pred = grid.best_estimator_.predict(X_test)
# print(pd.DataFrame(confusion_matrix(y_test, y_pred)))
