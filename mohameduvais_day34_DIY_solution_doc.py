#!/usr/bin/env python
# coding: utf-8

# Q1. Problem Statement: Linear Regression
# Load the housing_price.csv dataset to a DataFrame and perform the following tasks:
# The dataset contains only numeric data, and the median house value column is our target
# variable, so with the help of linear regression, build a model that can predict accurate
# house prices. Perform the below tasks and build a model:
# 1. Load the housing_price dataset into DataFrame
# 2. Find the null value and drop then, If any
# 3. Split data into two DataFrames x and y based on dependent and independent
# variables
# 4. Split x and y into 80% training set and 20% testing set. Set the random state to
# 10. Call the LinearRegression model, then fit the model using train data
# 5. Print the R2 value, coefficient, and intercept
# 6. Compare actual and predicted values.
# 7. Print the final summary

# In[53]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


# In[54]:


a=pd.read_csv('housing_price.csv')
a


# In[55]:


a.info()


# In[56]:


a.isnull().sum()


# In[57]:


a.describe()


# In[58]:


x=a.drop('median_house_value',axis=1)
y=a.median_house_value
print('independent data:''\n', np.array(x))
print('\n')
print('dependent data:', np.array(y))    


# In[59]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)
print('x_train and x_test dataset shape:',x_train.shape,x_test.shape)
print('y_train and y_test dataset shape:',y_train.shape,y_test.shape)


# In[60]:


b=LinearRegression()
b.fit(x,y)
c=b.score(x,y)
print('R square value is :''\n', c)
print('\n')
print('Coefficient is:''\n',b.coef_)
print('\n')
print('interceot is:''\n',b.intercept_)


# In[61]:


print(y)


# In[75]:


e=b.predict(x_test)
d=pd.DataFrame({'Actual':y_test,'Predicted':e})
f=d.reset_index()
g=f.drop('index',axis=1)
g


# In[77]:


import statsmodels.api as m
z=m.add_constant(x)
i=m.OLS(y,z)
h=i.fit()
print(h.summary())


# In[ ]:




