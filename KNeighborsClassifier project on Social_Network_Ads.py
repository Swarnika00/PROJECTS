
import pandas as pd
df=pd.read_csv("/home/sawra/Documents/SWARNIKA/D.S/05 KNN/Social_Network_Ads.csv")

x=df.iloc[:,[2,3]].values
y=df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=(0))

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=17)

classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)













