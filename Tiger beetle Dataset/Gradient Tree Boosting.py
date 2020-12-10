# importing necessary libraries
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import math
from sklearn import metrics
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
tree_range = range(10,30)
scores = []

t0 = time()
#clf=AdaBoostClassifier(n_estimators=100, random_state=0).fit(X_train, y_train)
clf=GradientBoostingClassifier(n_estimators=8, learning_rate=0.1,max_depth=6, random_state=0).fit(X_train, y_train)
ens_predictions = clf.predict(X_test)
print('time',time() - t0)
scores.append(metrics.accuracy_score(y_test, ens_predictions))
'''
plt.plot(tree_range, scores,'.-')
plt.xlabel('number of trees')
plt.ylabel('Testing Accuracy')
plt.show()     
'''
 
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















print('true result  4')
svm_pred=clf.predict([[2,	1,	6.936944,	79.985278,	22.9,	38,	578,	53,	0,	0,	0,	0,	8]])
print('predicted result',svm_pred)

print('true result  5')
svm_pred=clf.predict([[1,	2,	7.906233,	81.564474,	2.6,	30,	280,	69,	12,	7.4,	2020,	8.3,    12.167]])
print('predicted result',svm_pred)


svm_pred=clf.predict([[4,	2,	9.517221,	80.54071,	8.3,	35,	800,	54,	5,	12.1,	530,	8.3,	10.9]])
#print(svm_pred)

print('true result 6')
svm_pred=clf.predict([[4,	1,	6.784722,	80.134167,	26.2,	32,	126,	64,	0,	17.96,	0,	6,	9.45]])




print('predicted result',svm_pred)


print('true result 12')
svm_pred=clf.predict([[1,	2,	8.866413,	81.020591,	5.2,	39,	550,	45,	4,	4.5,	380,	7.8,	11.5]])
print('predicted result',svm_pred)


# Plot the confusion matrix as an image.
plt.matshow(cm)

    # Make various adjustments to the plot.
plt.colorbar()
tick_marks = np.arange(14)
plt.xticks(tick_marks, range(14))
plt.yticks(tick_marks, range(14))
plt.xlabel('Predicted')
plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
plt.show()

