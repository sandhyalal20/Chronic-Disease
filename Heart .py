#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split


# In[3]:


data = pd.read_csv(r'C:\Users\ELCOT\Desktop\heart.csv')


# In[4]:


data.shape


# In[5]:


data.head()


# In[6]:


data.describe()


# In[7]:


sns.pairplot(data=data[['age','sex','cp','trestbps','target']],hue='target')


# In[8]:


sns.pairplot(data=data[['exang','oldpeak','slope','target']],hue='target')


# In[9]:


train = data.drop('target',axis = 1)
train.head()


# In[10]:


target = data.target
target.head()


# In[11]:


x_train,x_test,y_train,y_test = train_test_split(train,target,test_size = 0.3,random_state = 109)


# In[12]:


print("X_train_size ==>",x_train.shape)
print("Y_train_size ==>",y_train.shape)
print("X_test_size ==>",x_test.shape)
print("Y_test_size ==>",y_test.shape)


# In[13]:


clf = svm.SVC(kernel = 'linear')
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)


# In[14]:


print("ACCURACY :",metrics.accuracy_score(y_test,y_pred))


# In[15]:


print("PRECISION :",metrics.precision_score(y_test,y_pred))


# In[16]:


print("RECALL :",metrics.recall_score(y_test,y_pred))


# In[18]:


x_train,x_test,y_train,y_test = train_test_split(train,target,test_size=0.30,random_state=10)


# In[19]:


from sklearn.impute import SimpleImputer
fill_values = SimpleImputer(missing_values = 0,strategy='mean',verbose=0)
x_train = fill_values.fit_transform(x_train)
x_test = fill_values.fit_transform(x_test)


# In[20]:


from sklearn.ensemble import RandomForestClassifier
random_forest_mode1 = RandomForestClassifier(random_state=10)

random_forest_mode1.fit(x_train,y_train.ravel())


# In[21]:


predict_train_data = random_forest_mode1.predict(x_test)

from sklearn import metrics

print("Accuracy: {0}".format(metrics.accuracy_score(y_test,predict_train_data)))


# In[22]:


from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(x_train,y_train.ravel())

y_pred = gnb.predict(x_test)


# In[23]:


from sklearn import metrics

print("ACCURACY : {0}".format(metrics.accuracy_score(y_test,y_pred)))


# In[24]:


from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()

classifier = classifier.fit(x_train,y_train)


# In[25]:


y_pred = classifier.predict(x_test)
print("ACCURACY : {0}".format(metrics.accuracy_score(y_test,y_pred)))


# In[ ]:




