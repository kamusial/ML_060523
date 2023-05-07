import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
# from mlxtend.plotting import plot_decision_regions

df = pd.read_csv('iris.csv')
print(df)
print(df['class'].value_counts())
species = {
    'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica':2
}
df['class_value'] = df['class'].map(species)

X1 = df[['sepallength', 'sepalwidth']]
X2 = df[['petallength','petalwidth']]
X3 = df.iloc[:, :4]
y = df.class_value
X_train, X_test, y_train, y_test = train_test_split(X3, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=10, max_depth=5)
model.fit(X3, y)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))