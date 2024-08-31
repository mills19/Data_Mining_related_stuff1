import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df=pd.read_csv("Titanic.csv")

df=df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'],axis=1)
df=df.dropna()
df=pd.get_dummies(df,columns=['Sex','Embarked'],drop_first=True)
X=df.drop('Survived',axis=1)
y=df['Survived']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LogisticRegression(random_state=42)
model.fit(X_train,y_train)

pred=model.predict(X_test)
probabi=model.predict_proba(X_test)[:,1]

accuracy=accuracy_score(y_test,pred)
classification_rep=classification_report(y_test,pred)

print(accuracy)
print(classification_rep)
print(probabi)
