#loading data using pandas
import pandas as pd

titanic=pd.read_csv("Titanic.csv")

# to view  first 5 rows of the dataset
titanic.head()

#chceking null values
titanic.isnull().sum()

#graphical analysis 
import seaborn as sns
import matplotlib.pyplot as plt
#countplot
sns.catplot(x="Sex",hue="Survived" ,kind="count",data=titanic)

group=titanic.groupby(['Pclass','Survived'])
group.size().unstack()

pclass_survived=group.size().unstack()
#2D representation
sns.heatmap(pclass_survived,annot=True,fmt="d")

#age(continuous ) vs survived..violinplot displays distribution 
#of data acroll all levels of a category
sns.violinplot(x="Sex",y="Age",hue="Survived",data=titanic,split=True)


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
df=pd.read_csv("Titanic.csv")
display(df)

#sorting the names 
sorted=df.sort_values(by=['Name'])
display(sorted)

#filtering rows
filterr=df.query('Survived==True')
display(filterr)

#filtering columns
filterrr=df.filter(['PassengerId','Name','Age'])
display(filterrr)

#removing duplicates
display(df.duplicated())

dup_r=df.drop_duplicates()

#for missing values..
display(df.isna())

#listwise deletion using dropna func
dr=df.dropna(how='any')

long=pd.melt(dup_r,id_vars=['PassengerId','Name'],value_vars=['SibSp','Parch'],
             var_name='Total',value_name='Values')
display(long)

#long->wide
wide=pd.pivot(long,index=['PassengerId','Name'],columns='Total',values='Values')

#transformation
df['Age'].hist()


#normalization
scaling=df.copy()
min_target=np.min_target(scaling['Age'])
max_target=np.max_target(scaling['Age'])
scaling['norm_age']=(scaling['age']-min_target)/(max_target-min_target)
display(scaling)

#transformation using normalizationd
cols=['PassengerId','Pclass','Fare']
from sklearn.preprocessing import Normalizer
norm=Normalizer()
Normal=norm.fit_transform(df[cols].iloc[:,range(0,2)].values)
#data_norm.fit_transform(data[cols].iloc[:,range(0,7)].values)
sns.displot(Normal[:,1],fill=True,color='red')
plt.xlabel('Normalized values')
plt.show()
