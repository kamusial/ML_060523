import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('otodom.csv')
print(df.head(5).to_string())
print(df.describe().T.to_string())   # T zamiana wierszy i kolumn

print(df.corr())   #korelacja
sns.heatmap(df.corr(), annot=True)
plt.show()
sns.histplot(df.cena)
plt.show()
