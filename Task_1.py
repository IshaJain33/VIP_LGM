#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn import tree


# In[2]:


df=pd.read_csv("iris.csv",header=None,names=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)','Species'])
df


# In[3]:


df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.value_counts()


# In[9]:


df.isnull().sum()


# In[10]:


plt.figure(figsize=(10,7))
sns.boxplot(x='Species',y='sepal width (cm)',data=df ,palette='YlGnBu')


# In[11]:


sns.countplot(x='Species',data=df, palette="OrRd")


# In[12]:


df.plot(kind='scatter',x='sepal length (cm)',y='sepal width (cm)')
plt.show()


# In[14]:


X = df.drop(df.columns[-1], axis=1)
y = df[df.columns[-1]]


# In[15]:


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[17]:


decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)


# In[18]:


y_pred = decision_tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)


# In[19]:


print("Accuracy:", accuracy)


# In[21]:


print("\nClassification Report \n")
print(class_report)


# In[22]:


from sklearn import tree
plt.figure(figsize=(15, 10)) 
tree.plot_tree(decision_tree, filled=True) 


# In[25]:


print(confusion_matrix(y_test, y_pred)) 


# In[26]:


cm  = confusion_matrix(y_test, y_pred) 
cm_df = pd.DataFrame(cm,index = ['setosa','versicolor','virginica'], columns = ['setosa','versicolor','virginica'])
plt.figure(figsize=(5.5,4))
sns.heatmap(cm_df,   annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[ ]:




