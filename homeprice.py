import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

from sklearn import linear_model


df = pd.read_csv('homeprices.csv')
plt.xlabel('area(sqr ft)')
plt.ylabel('Price(US$)')
plt.scatter(df.area,df.price,color='red')



reg = linear_model.LinearRegression()
reg.fit(df[['area']],df.price)

prediction  = reg.predict([[3300]])
print(prediction[0])


plt.plot(df.area,reg.predict(df[['area']]),color='blue')
plt.show()

d = pd.read_csv("prices.csv")

p = reg.predict(d)

d['prices'] = p 

d.to_csv("new_prices.csv")