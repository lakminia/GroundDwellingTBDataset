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
from time import time 
#load data 
names = ['Habitat type ','Climatic zone','co-ordinate(N)/latitude','co-ordinate€/longitude','elevation(m)','Temperature©','solar_radiation(w/m2)','relative humidity(%)','wind speed(MPH)','soil moisture(%)','soil salinity(Us/Cm)','soil PH','body length(mm)','species'
]
dataframe = read_csv('Ground-dwelling tiger beetle dataset.csv', names=names)
#print(dataframe)
array = dataframe.values

#print(array)

# X -> features, y -> label
X = array[:,0:13]
y = array[:,13]
 
# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
 
# training a Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB
t0 = time()
gnb = GaussianNB().fit(X_train, y_train)
gnb_predictions = gnb.predict(X_test)
print('time',time() - t0) 
# accuracy on X_test
print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))
accuracy = gnb.score(X_test, y_test)
print (accuracy)
 
# creating a confusion matrix
cm = confusion_matrix(y_test, gnb_predictions)
print(cm)
