import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
x=[5,10,3,4,8,13,12]
y=[19,21,34,32,37,43,38]
classes=[0,1,0,0,1,1,1]
plt.scatter(x,y,c=classes)
plt.show()

data=list(zip(x,y))
print(data)

#fit the model
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(data,classes)

#predict new point
new_x=18
new_y=78
new_point=[(new_x,new_y)]
pred=knn.predict(new_point)
print(pred)

plt.scatter(x+[new_x],y+[new_y],c=classes+[pred[0]])
plt.show()

