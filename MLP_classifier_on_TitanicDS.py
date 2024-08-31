import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

titanic_df = pd.read_csv('Titanic.csv')

mean_age = titanic_df['Age'].mean()
print("mean_age",mean_age)

titanic_df=titanic_df.drop(['PassengerId', 'Name', 'Ticket','Sex', 'Cabin','Embarked'],axis=1)
titanic_df=titanic_df.dropna()

X = titanic_df.drop(['Survived'], axis=1)  # Features
y = titanic_df['Survived']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

classifier = MLPClassifier(random_state=42)
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"\nAccuracy obtained using Multinomial Perceptron Neural Network Classifier: {accuracy:.2f}")



