import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

titanic_df = pd.read_csv('Titanic.csv')

print("Last 10 records in the dataset:")
print(titanic_df.tail(10))

print("\nNames of the rows present in the dataset:")
print(titanic_df.index)

titanic_df=titanic_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin','Sex','Embarked'],axis=1)
titanic_df=titanic_df.dropna()

X = titanic_df.drop('Survived', axis=1)
y = titanic_df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

naive_bayes_classifier = GaussianNB()
naive_bayes_classifier.fit(X_train, y_train)

predictions = naive_bayes_classifier.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("\nAccuracy obtained using Gaussian Naive Bayes algorithm:", accuracy)
