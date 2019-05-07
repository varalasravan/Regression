#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


USAhousing = pd.read_csv('USA_Housing.csv')


# In[4]:


USAhousing.head()


# In[5]:


USAhousing.describe()


# In[13]:


X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']


# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
X_test


# In[18]:


from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train,y_train)


# In[19]:


print(lm.intercept_)


# In[20]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[21]:


predictions = lm.predict(X_test)


# In[22]:


plt.scatter(y_test,predictions)

