
# coding: utf-8

# In[34]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[35]:


dataset = pd.read_csv('50-Startups.csv')
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : ,  4 ].values


# In[39]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[: , 3] = labelencoder.fit_transform(X[ : , 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()


# In[40]:


X = X[: , 1:]


# In[26]:


from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# In[27]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


# In[28]:


y_pred = regressor.predict(X_test)


# In[29]:


print(y_pred)

