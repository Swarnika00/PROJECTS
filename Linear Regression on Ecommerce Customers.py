# Importing the Dependencies
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#Data Collection & Pre-Processing
df=pd.read_csv("/home/sawra/Documents/SWARNIKA/D.S/07 Linear Regression/Ecommerce Customers (2).csv")

df.head()
df.dtypes

q=df.describe()

#exploratory DA
sns.jointplot(x="Time on Website", y="Yearly Amount Spent", data=df)
sns.jointplot(x="Time on App", y="Yearly Amount Spent", data=df)
sns.pairplot(df)

x=df.iloc[:,3:7].values
y=df.iloc[:,-1].values

# Splitting the data into training data & test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

# Linear Regression
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(x_train,y_train)

#prediction
pred_=lm.predict(x_test)            

import sklearn.metrics as metrics
metrics.mean_absolute_error(y_test, pred_)   
metrics.mean_squared_error(y_test, pred_)
np.sqrt(metrics.mean_squared_error(y_test, pred_))


plt.scatter(y_test,pred_)
plt.xlabel("Prediction")
plt.ylabel("y test")

    

#finding the coefficient of features/large value better dependincy
lm.coef_
pd.DataFrame(lm.coef_,index=["Avg. Session Length",
"Time on App",
"Time on Website",
"Length of Membership"])

























