import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("Titanic.csv")
missing_count=df.isnull().sum()
plt.figure(figsize=(10,5))
missing_count.plot(kind='bar',color='red')
plt.title("Number of missing values for each attribute")
plt.xlabel('Attribute')
plt.ylabel('No of missing values')
plt.show()


survival_count=df['Survived'].value_counts()
plt.figure(figsize=(5,5))
plt.pie(survival_count,labels=['Survived','Not survived'],colors=['red','blue'],autopct='%1.1f%%',startangle=90)
plt.title('Percentage of survival')
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(data=df,x='Age',hue='Survived',kde=True,palette=['yellow','lightgreen'],multiple='stack',bins=30)
plt.title('Relationship between age and survival')
plt.xlabel('Age')
plt.ylabel('Survival')
plt.show()
