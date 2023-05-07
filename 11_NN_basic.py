import pandas as pd
import matplotlib.pyplot as plt
#komponenty do budowy sieci
from tensorflow.keras.models import Sequential   #szkielet sieci
from tensorflow.keras.layers import Dense
from tensorflow.random import set_seed

set_seed(0)
model = Sequential()
model.add(Dense(4, input_shape=[1], activation='linear'))
model.add(Dense(2, activation='linear'))
model.add(Dense(1))

#kompilacja
model.compile(optimizer='rmsprop', loss='mse')

df = pd.read_csv('f-c.csv', usecols=[1, 2])
print(df.head())

plt.scatter(df.F, df.C)
plt.show()

result = model.fit(df.F, df.C, epochs=1500)

df1 = pd.DataFrame(result.history)
print(df1.head())
df1.plot()
plt.show()

y_pred = model.predict(df.F)

plt.scatter(df.F, df.C)
plt.plot(df.F, y_pred, c='r')
plt.show()