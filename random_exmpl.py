import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics  import mean_squared_error

#computing y=e^x for x=1 to 50
x=np.arange(1,51).reshape(-1,1)
y=np.exp(x)
print(y)

df=pd.DataFrame(data={'x':x.flatten(),'y':y.flatten()})
df.to_csv('exp6.csv')

#predicting the values for y when x is between 51 to 60
regressor=LinearRegression()
regressor.fit(x,y)
x1=np.arange(51,61).reshape(-1,1)
y_pred=regressor.predict(x1)
print("predicted vals for y when x is between 51 to 60",y_pred)

#eval error in terms of training set
y1=regressor.predict(x)
error=mean_squared_error(y,y1)
print("error",error)
