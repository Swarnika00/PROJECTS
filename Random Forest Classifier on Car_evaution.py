
import pandas as pd
import seaborn as sms
import matplotlib.pyplot as plt 
df = pd.read_csv("/home/sawra/Documents/SWARNIKA/D.S/06 Random Forest/car_evaluation.csv")

df.head()
df.tail()

col_names = ["buying", "maint", "doors",
             "persons", "lug_boot", "safety", "class"]
df.columns = col_names

# checking null values
df.info()

# checking data types of features
df.dtypes

# checking unique values or frequency distribution of data
for x in col_names:
    print(df[x].value_counts())

x = df.drop(["class"], axis=1)
y = df["class"]

# Splitting data into train test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.25, random_state=(0))


# features engineering
import category_encoders as ce
encoder = ce.OrdinalEncoder(
    cols=["buying", "maint", "doors", "persons", "lug_boot", "safety"])
x_train = encoder.fit_transform(x_train)
x_test = encoder.transform(x_test)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, random_state=(40))

rfc .fit(x_train, y_train)
y_pred=rfc.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

#Feature importance
feature_score=pd.Series(rfc.feature_importances_,index=x_train.columns.sort_values)
print(feature_score)

sms.barplot(x=feature_score, y=feature_score.index)
plt.xlabel("feature.score")
plt.ylabel("features")
plt.title("importances")
plt.show    

















