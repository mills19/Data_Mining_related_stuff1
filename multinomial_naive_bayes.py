import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report

titanic_df = pd.read_csv('Titanic.csv')
print(titanic_df.info())

titanic_df=titanic_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin','Embarked'],axis=1)
titanic_df=titanic_df.dropna()

X = titanic_df.drop('Survived', axis=1)
y = titanic_df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

selected_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Converting categorical variables to dummy variables
X_train_selected = pd.get_dummies(X_train_selected, columns=['Sex'], drop_first=True)
X_test_selected = pd.get_dummies(X_test_selected, columns=['Sex'], drop_first=True)

classifier = MultinomialNB()
classifier.fit(X_train_selected, y_train)

predictions = classifier.predict(X_test_selected)

accuracy = accuracy_score(y_test, predictions)
print("\nAccuracy obtained :", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, predictions))
