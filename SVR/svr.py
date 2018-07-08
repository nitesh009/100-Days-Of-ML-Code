#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 09:55:33 2018

@author: knitesh
"""

import numpy as np
import pandas as pd

df = pd.read_csv("/Users/i351036/Downloads/100-Days-Of-ML-Code/SVR/computer-hardware.csv")

print(df.head())

print("Total number of enteried: {}".format(df.size))
print("Shape of the data: {}".format(df.shape))

# Now let's normalize the data. The values of all features are ranging very differently.
# So normalizing is very necessary. Also we are spliting the data into train and test for future use. 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cross_validation import train_test_split


# Let's convert our vendor's name into number. As it is categorial variable we can not fit that into model directly.
# So we will convert that into some number ranging from 0 to the number of unique vendor's name. 
encoder = LabelEncoder()
df['vendor_name'] = encoder.fit_transform(df['vendor_name']).astype('str')
# print(df['vendor'].head())

y = np.array(df["ERP"])
X = np.array(df[["vendor_name","MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP"]]) #Here we are not using vendor name as that is categorial variable.


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33)


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




from sklearn.svm import SVR
kernels = ['rbf', 'linear', 'poly', 'sigmoid'] 

def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    # Fitting the model on the data
    clf.fit(X_train, y_train)
    
    print ("Accuracy on training set: {}".format(clf.score(X_train, y_train)))
    print ("Accuracy on testing set: {}".format(clf.score(X_test, y_test)))


for i in kernels:
    clf = SVR(kernel= i)
    print("Results of SVR with kernel: {}".format(i))
    train_and_evaluate(clf, X_train, X_test, y_train, y_test)
    print("")
    
    
 y_pred = SVR(kernel= 'linear').fit(X_train, y_train).predict(X_test)
from sklearn.metrics import mean_squared_error
print("Root Mean Square Error is: {}".format(mean_squared_error(y_test,y_pred)))