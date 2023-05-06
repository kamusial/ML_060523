import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

df = pd.read_csv('diabetes.csv')
print(df.head(5).to_string())
print(df.describe().T.to_string())
print(df.isna().sum())
print(df.outcome.value_counts())
#wszędzie, gdzie są zera albo brak wartości, chcemy wpisać wartosć średnią (nieuwzględniającą zer)
for col in ['glucose', 'bloodpressure', 'skinthickness', 'insulin', 'bmi', 'diabetespedigreefunction', 'age']:
    df[col].replace(0, np.NaN, inplace=True)   #zamiana zer na NaN
    mean_ = df[col].mean()
    df[col].replace(np.NaN, mean_, inplace=True)
print(df.isna().sum())
print(df.describe().T.to_string())

df.to_csv('cukrzyca.csv')
X = df.iloc[:, :-1]
y = df.outcome
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))

print('\nZmiana danych')
df1 = df.query('outcome==0').sample(n=500, random_state=0)
df2 = df.query('outcome==1').sample(n=500, random_state=0)
df3 = pd.concat([df1, df2])

X = df3.iloc[:, :-1]
y = df3.outcome
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))