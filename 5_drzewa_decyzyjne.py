import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_decision_regions

df = pd.read_csv('iris.csv')
print(df)
print(df['class'].value_counts())
species = {
    'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica':2
}
df['class_value'] = df['class'].map(species)
print(df['class_value'].value_counts())
# sns.scatterplot(data=df, x='sepallength', y='sepalwidth', hue='class')
# plt.title('Sepal')
# sns.scatterplot(data=df, x='petallength', y='petalwidth', hue='class')
# plt.title('Petal')
# plt.show()

X1 = df[['sepallength', 'sepalwidth']]
X2 = df[['petallength','petalwidth']]
X3 = df.iloc[:, :4]
y = df.class_value

model = DecisionTreeClassifier(max_depth=10)
model.fit(X3, y)
# plot_decision_regions(X2.values, y.values, model)
# plt.show()

print(pd.DataFrame(model.feature_importances_, X3.columns))
