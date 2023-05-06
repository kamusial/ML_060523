import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('otodom.csv')
print(df.head(5).to_string())
print(df.describe().T.to_string())   # T zamiana wierszy i kolumn

# print(df.corr())   #korelacja
# sns.heatmap(df.corr(), annot=True)
# plt.show()
# sns.histplot(df.cena)
# plt.show()

_min = df.describe().loc['min', 'cena']
q1 = df.describe().loc['25%', 'cena']
q3 = df.describe().loc['75%', 'cena']

df1 = df[ (df.cena >= _min) & (df.cena <= q3) & (df.rok_budowy < df.describe().loc['max','rok_budowy']) ]
print(df1.describe().T.to_string())
# sns.histplot(df1.cena)
# plt.show()

#algorytm
print(df1.columns)
X = df1.iloc[:, 2:]
y = df1.cena
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(model.coef_)
print(model.intercept_)