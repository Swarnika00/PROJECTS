#Importing the Dependencies
import pandas as pd
train=pd.read_csv("/home/sawra/Documents/python/CSV FILE/loan_data_set.csv")
test=pd.read_csv("/home/sawra/Documents/SWARNIKA/D.S/08 Logistic Regression/test_loan_data.csv")

#Find the null values
train.isnull().sum()
test.isnull().sum()
    

train.dtypes

train["Gender"].fillna(train["Gender"].mode()[0],inplace=True)
train["Married"].fillna(train["Married"].mode()[0],inplace=True)
train["Dependents"].fillna(train["Dependents"].mode()[0],inplace=True)
train["Self_Employed"].fillna(train["Self_Employed"].mode()[0],inplace=True)
train["LoanAmount"].fillna(train["LoanAmount"].median(),inplace=True)
train["Loan_Amount_Term"].fillna(train["Loan_Amount_Term"].median(),inplace=True)
train["Credit_History"].fillna(train["Credit_History"].median(),inplace=True) 

test["Gender"].fillna(train["Gender"].mode()[0],inplace=True)
test["Married"].fillna(train["Married"].mode()[0],inplace=True)
test["Dependents"].fillna(train["Dependents"].mode()[0],inplace=True)
test["Self_Employed"].fillna(train["Self_Employed"].mode()[0],inplace=True)
test["LoanAmount"].fillna(train["LoanAmount"].median(),inplace=True)
test["Loan_Amount_Term"].fillna(train["Loan_Amount_Term"].median(),inplace=True)
test["Credit_History"].fillna(train["Credit_History"].median(),inplace=True) 

train=train.drop("Loan_ID",axis=1)
test=test.drop("Loan_ID",axis=1)


x=train.drop("Loan_Status",axis=1)
y=train.Loan_Status

#ONE HOT ENCODING (convert all string value into numerical )
x=pd.get_dummies(x)
train=pd.get_dummies(train)
test=pd.get_dummies(test)


from sklearn.model_selection import train_test_split
x_train,x_val,y_train, y_val=train_test_split(x,y,test_size=0.3,random_state=(0)) 

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Train the model
model=LogisticRegression()
model.fit(x_train,y_train)

#Validation 
pred_val=model.predict(x_val)
accuracy_score(y_val, pred_val)

#Prediction
test["new_pred"]=model.predict(test)

















