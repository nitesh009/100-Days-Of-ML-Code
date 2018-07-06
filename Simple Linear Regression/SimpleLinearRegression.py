#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 18:15:20 2018

@author: knitesh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import *
from sklearn.linear_model import LinearRegression


boston=load_boston()
print(boston.data.shape)
print(boston.feature_names)
print(np.max(boston.target),np.min(boston.target),np.mean(boston.target))


# Splitting the data into train and test. Here we are not doing validation as we do not have really large dataset.
X_train,X_test,y_train,y_test=train_test_split(boston.data,boston.target,test_size=0.25,random_state=33)


# Now we will normalize our data as it is one of the most useful thing to do beofre training.
# If the data is not normalized then the model will not fit properly. 
scalerX = StandardScaler().fit(X_train)
scalery = StandardScaler().fit(np.reshape(y_train,(-1,1)))

X_train = scalerX.transform(X_train)
y_train = scalery.transform(np.reshape(y_train,(-1,1)))
X_test = scalerX.transform(X_test)
y_test = scalery.transform(np.reshape(y_test,(-1,1)))


print (np.max(X_train), np.min(X_train), np.mean(X_train), np.max(y_train), np.min(y_train), np.mean(y_train))


# converting reshaped Y vector into array again
y_train=y_train.flatten(order='C')
y_test=y_test.flatten(order='C')





# Here we will create a simple function which we can use for training and validation of the model.
def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
#     Fitting the model on the data
    clf.fit(X_train, y_train)
    
    print ("Accuracy on training set: {}".format(clf.score(X_train, y_train)))
    print ("Accuracy on testing set: {}".format(clf.score(X_test, y_test)))



# Initializing the model and doing training using our function.# Initi 
clf=LinearRegression()
train_and_evaluate(clf, X_train, X_test, y_train, y_test)




# Now we will do some simple plotting. # Now w 
# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
y = boston.target
predicted = cross_val_predict(clf, boston.data, y, cv=10)



fig, ax = plt.subplots()
ax.scatter(y, predicted, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
