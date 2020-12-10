# importing necessary libraries
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import numpy as np
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn import metrics
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
 
# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
 
# training a KNN classifier
from sklearn.neighbors import KNeighborsClassifier
t0 = time()
knn = KNeighborsClassifier(n_neighbors = 1).fit(X_train, y_train)
knn_predictions = knn.predict(X_test)
print('time',time() - t0)
# accuracy on X_test
print(len(X_train))
print(len(X_test))
print(len(y_train))      
accuracy = knn.score(X_test, y_test)
print (accuracy)
 
# creating a confusion matrix

cm = confusion_matrix(y_test, knn_predictions)
print(cm)

print('answer  5')
svm_pred=knn.predict([[1,	2,	7.906233,	81.564474,	2.6,	30,	280,	69,	12,	7.4,	2020,	8.3,	12.16]])
print(svm_pred)

print('answer  4')
svm_pred=knn.predict([[2,	1,	6.936944,	79.985278,	22.9,	38,	578,	53,	0,	0,	0,	0,	8]])
print(svm_pred)

