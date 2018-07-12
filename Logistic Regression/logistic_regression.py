#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 19:41:54 2018

@author: i351036
"""
 
################################### Step 1 | Data Pre-Processing ########################################

# Importing the Libraries
 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/Users/i351036/Downloads/100-Days-Of-ML-Code/Logistic Regression/Social_Network_Ads.csv')

X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


################################### Step 2 | Logistic Regression Model ########################################

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)



################################### Step 3 | Predection ########################################
y_pred = classifier.predict(X_test)


################################### Step 4 | Evaluating The Prediction ########################################

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



