# importing necessary libraries
import numpy as np
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn import svm

from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.utils import resample
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
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)
#X_train, X_test, y_train, y_test = resample(X, y,random_state=0)
 
# training a linear SVM classifier
#from sklearn.svm import SVC
t0 = time()
svm_model_rbf = svm.SVC(kernel = 'rbf',C = 205).fit(X_train, y_train)
svm_predictions = svm_model_rbf.predict(X_test)
print('time',time() - t0) 
# model accuracy for X_test
print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))
accuracy = svm_model_rbf.score(X_test, y_test)
'''
for color in X_test:
    print(color)
    
for color in y_test:
    print(color)
'''
print(accuracy)
 
# creating a confusion matrix
cm = confusion_matrix(y_test, svm_predictions)
print(cm)



'''
svm_pred=svm_model_rbf.predict([[2,	2,	8.357778,	80.443611,	80.69,	34,	159,	47,	21,	0.35,	0,	6.9,	12.6]])
print(svm_pred)

svm_pred=svm_model_rbf.predict([[2,	1,	6.936944,	79.985278,	22.9,	38,	578,	53,	0,	0,	0,	0,	8]])
print('answer  4')
print(svm_pred)


svm_pred=svm_model_rbf.predict([[1,	2,	7.906233,	81.564474,	2.6,	30,	280,	69,	12,	7.4,	2020,	8.3,	12.16]])
print('answer  5')
print(svm_pred)

print('answer 6')
svm_pred=svm_model_rbf.predict([[4,	1,	6.784722,	80.134167,	26.2,	32,	126,	64,	0,	17.96,	0,	6,	9.45]])
print(svm_pred)

print('answer 12')
svm_pred=svm_model_rbf.predict([[1,	2,	8.866413,	81.020591,	5.2,	39,	550,	45,	4,	4.5,	380,	7.8,	11.5]])
print(svm_pred)
'''
