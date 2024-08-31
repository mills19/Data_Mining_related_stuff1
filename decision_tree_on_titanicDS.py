import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report

df=pd.read_csv("Titanic.csv")
df.describe()
print(df)

df.dropna()

df=pd.get_dummies(df,columns=['Cabin'],drop_first=True)
display(df)

x=df.drop('Survived',axis=1)
y=df['Survived']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
x_train.shape
df =df.drop(['Parch'], axis=1)

criteria_l=['gini','entropy']
splitter=['best','random']
for criterion in criteria:
    for splitter in splitter:
        
        model=DecisionTreeClassifier(criterion=criteria,splitter=splitter,random_state=42)
        model.fit(x_train,y_train)
        
        #to see how well it can predict
        pred=model.predict(x_test)
        
        #to eval performance of the model
        accuracy=accuracy_score(y_test,pred)
        class_rep=classification_report(y_test,pred)
        
        print("Accuracy",accuracy)
        print("Classification report:",class_rep)


