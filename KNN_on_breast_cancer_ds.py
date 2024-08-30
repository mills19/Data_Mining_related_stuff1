import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
cancer_dataset = load_breast_cancer()

cancer_dataset.feature_names
cancer_dataset.data.shape
cancer_dataset.target


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


X_train,Y_train,x_test,y_test=train_test_split(cancer_dataset.data,cancer_dataset.target,random_state=0)
X_train.shape
x_test.shape
Y_train.shape
y_test.shape


training_accuracy=[]
test_accuracy=[]
neighbors_settings= range(1,11)
for n_neighbors in neighbors_settings:
    clf=KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train,Y_train)
    training_accuracy.append(clf.score(X_train,Y_train))
    test_accuracy.append(clf.score(x_test,y_test))
    


print(training_accuracy)

import mglearn
import matplotlib.pyplot as plt
mglearn.plots.plot_kmeans_algorithm()


from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
X,y=make_blobs(random_state=1)
kkmeans=KMeans(n_clusters=3)
kkmeans.fit(X)

kkmeans.predict(X)

mglearn.discrete_scatter(X[:,0],X[:,1],kkmeans.labels_,markers='o')
mglearn.discrete_scatter(kkmeans.cluster_centers_[:,0],kkmeans.cluster_centers_[:,1],
                         [0,1,2],markers='^', markeredgewidth=2)
