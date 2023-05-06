import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('weight-height.csv')
#print(type(df))
print(df.head(5))
#czy klasy zrównoważone
print(df.Gender.value_counts())
df.Height *= 2.54
df.Weight /= 2.2
print('Zmiana jednostek')
print(df.head(5))
print('zmienne niezależne -> plec i wzrost,    zmienna zależna -> waga')
#sns.histplot(df.Weight)
# sns.histplot(df.query("Gender=='Male'").Weight)
# sns.histplot(df.query("Gender=='Female'").Weight)
# plt.show()
df = pd.get_dummies(df)   #zmiana na dane numeryczne
#print(df)
del(df["Gender_Male"])  #usuń kolumnę
df.rename(columns={'Gender_Female': 'Gender'}, inplace=True)
#print(df)

model = LinearRegression()
model.fit(df[['Height', 'Gender']], df['Weight'])
print(model.coef_, model.intercept_)

print('wzór: Height * ',model.coef_[0], '+ Gender * ',model.coef_[1], '+ ',model.intercept_)

gender = 0
height = 70
weight = height * model.coef_[0] + gender * model.coef_[1] + model.intercept_
print(weight)