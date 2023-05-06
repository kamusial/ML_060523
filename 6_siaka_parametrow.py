import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

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

model = DecisionTreeClassifier()
params = {
    'max_depth': range(2, 11, 2),
    'max_features': range(2, 14, 2),
    'min_sample_split': [2, 4, 6],
    'random_state': [0],
    'criterion': ['gini', 'entropy', 'log_loss']
}

grid = GridSearchCV(model, params, scoring='accuracy', cv=10, verbose=1)
grid.fit(X_train, y_train)
