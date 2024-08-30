import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
titanic=pd.read_csv("C:\\Users\\OneDrive\\Desktop\\Titanic.csv")
titanic.head()

drop_useless=['Pclass','SibSp','Embarked','Parch']
titanic.drop(drop_useless,axis=1)

titanic.dropna()

df= pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

missing=['Age','Parch']
df.dropna(subset=missing)

x=df.drop('Survived',axis=1)
y=df['Survived']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)

