
import category_encoders as ce
import pandas as pd
df=pd.read_csv("/home/sawra/Documents/SWARNIKA/D.S/06 Random Forest/car_evaluation.csv")

df.head()
df.tail()

col_names=["buying","maint","doors","persons","lug_boot","safety","class"]
df.columns=col_names

#cheaking null values
df.info()

#checking data types of features
df.dtypes

#checking unique values or frequency distribution of data
for x in col_names:
    print(df[x].value_counts())

x=df.drop(["class"],axis=1)
y=df["class"]

#Splitting data into train test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=(0))


#features engineering
encoder=ce.OrdinlEncoder(cols=["buying","maint","doors","persons","lug_boot","safety"])
x_train=encoder.fit_transform(x_train)
x_test=encoder.transform(x_test)
     














 
