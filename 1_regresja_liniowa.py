import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('weight-height.csv')
#print(type(df))
print(df.head(5))
#czy klasy zrównoważone
print(df.Gender.value_counts())