#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Random forest is a method that operates by constructing multiple decision trees during training phase 
# The decisio


# In[1]:


from sklearn.datasets import load_iris
import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestRegressor


# In[2]:


iris=load_iris()
df= pd.DataFrame(iris.data,columns=iris.feature_names)
df


# In[3]:


df['species']=pd.Categorical.from_codes(iris.target,iris.target_names)
df


# In[6]:


from sklearn.preprocessing import LabelEncoder
a=LabelEncoder()
df.iloc[:,4]=a.fit_transform(df.iloc[:,4])
df


# In[11]:


y=df['species']


# In[12]:


x=df.drop('species',axis=1)
x


# In[13]:


from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[14]:


df


# In[15]:


regressor=RandomForestRegressor(n_estimators=2,random_state=0)
regressor.fit(x_train,y_train)


# In[16]:


y_predict = regressor.predict(x_test)


# In[17]:


y_predict


# In[18]:


from sklearn.metrics import r2_score
r2_score(y_test,y_predict)


# In[ ]:




