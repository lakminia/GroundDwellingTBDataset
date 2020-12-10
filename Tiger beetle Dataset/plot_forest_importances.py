
print(__doc__)

import numpy as np

import matplotlib.pyplot as plt
from sklearn import svm

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier

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

# Build a classification task using 3 informative features
"""X, y = make_classification(n_samples=1000,
                           n_features=11,
                           n_informative=3,
                           n_redundant=0,
                           n_repeated=0,
                           n_classes=2,
                           random_state=0,
                           shuffle=False)
"""
# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=39, max_depth=None,min_samples_split=2, random_state=0).fit(X, y)


#forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")


for f in range(X.shape[1]):
    #print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    print("feature " +names[indices[f]],importances[indices[f]])

# Plot the feature importances of the forest
plt.figure()
Tfont = {'fontname':'Times New Roman'}

#plt.figure(figsize=(10,2),facecolor='w')
fig=plt.title("")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
#plt.xticks(range(X.shape[1]), names[indices[f]])
#plt.xticks(rotation=90)
plt.xticks(range(X.shape[1]), [str(names[indices[x]]) for x in range(X.shape[1])], rotation = 'vertical',**Tfont,fontsize=16)
plt.xticks(**Tfont)
#plt.ylabel('Square of Value',fontsize=14)

plt.legend()
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.show()
