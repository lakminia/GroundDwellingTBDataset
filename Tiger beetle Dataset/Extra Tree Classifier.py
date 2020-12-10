# importing necessary libraries
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn import svm
import numpy as np
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import math
from time import time
#load data 
names = ['Habitat type ','Climatic zone','co-ordinate(N)/latitude','co-ordinate€/longitude','elevation(m)','Temperature©','solar_radiation(w/m2)','relative humidity(%)','wind speed(MPH)','soil moisture(%)','soil salinity(Us/Cm)','soil PH','body length(mm)','species'
]
dataframe = read_csv('Ground-dwelling tiger beetle dataset.csv', names=names)
#print(dataframe)
array = dataframe.values

print(array)

# X -> features, y -> label
X = array[:,0:13]
y = array[:,13]
 
 
# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
 
# training a linear SVM classifier
# training a Naive Bayes classifier
#from sklearn.naive_bayes import GaussianNB
#gnb = GaussianNB().fit(X_train, y_train)
#gnb_predictions = gnb.predict(X_test)
t0 = time()
clf = ExtraTreesClassifier(n_estimators=14).fit(X_train, y_train)

ens_predictions = clf.predict(X_test)
print('time',time() - t0)
# model accuracy for X_test
print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))
accuracy = clf.score(X_test, y_test)
print(accuracy)
 
# creating a confusion matrix
cm = confusion_matrix(y_test, ens_predictions)
print(cm)


