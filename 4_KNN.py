import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

df = pd.read_csv('iris.csv')
print(df)
print(df['class'].value_counts())
species = {
    'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica':2
}
df['class_value'] = df['class'].map(species)
print(df['class_value'].value_counts())
sample = np.array([5.6, 3.2, 5.2, 1.45])   #mój liść
# sns.scatterplot(data=df, x='sepallength', y='sepalwidth', hue='class')
# plt.scatter(5.6, 3.2, c='r')
# sns.scatterplot(data=df, x='petallength', y='petalwidth', hue='class')
# plt.scatter(5.2, 1.45, c='r')
# plt.show()

#recznie
df['distance'] = (df.sepallength - sample[0]) ** 2 + (df.sepalwidth - sample[1]) ** 2 + \
                 (df.petallength - sample[2]) ** 2 + (df.petalwidth - sample[3]) ** 2
print(df.sort_values('distance').head(5).to_string())

X = df.iloc[:, :4]
y = df.class_value
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = KNeighborsClassifier()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))

result = []
for k in range(1, 101):
    model = KNeighborsClassifier(k)
    model.fit(X_train, y_train)
    result.append(model.score(X_test, y_test))
plt.plot(range(1, 101), result)
plt.grid()
plt.show()

