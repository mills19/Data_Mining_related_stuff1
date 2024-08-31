import  pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report
#creating a synthetic dataset
data={'text': ['Be Positive','You\'re so negative','He is positive','I do not like this'],
      'label':['positive','negative','positive','negative']}
d=pd.DataFrame(data)
print(d)

X_train,X_test,y_train,y_test=train_test_split(d['text'],d['label'],
                                    test_size=0.2, random_state=42)

#text vectorization
vectorizer=CountVectorizer()
X_train_vec=vectorizer.fit_transform(X_train)
X_test_vec=vectorizer.transform(X_test)

#training the classifier model
classifier=MultinomialNB()
classifier.fit(X_train_vec,y_train)

#making predictions on the test set
prediction=classifier.predict(X_test_vec)

#evaluating accuracy
accuracy=accuracy_score(prediction,y_test)
classification_rep=classification_report(prediction,y_test)

print("Accuracy is",accuracy)
print(classification_rep)




