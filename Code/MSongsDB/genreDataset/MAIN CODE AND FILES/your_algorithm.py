#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
import time
from sklearn.metrics import accuracy_score
features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary


### Decision Tree Algorithm

'''
from sklearn import tree
print "no. of Chris training emails:", sum(labels_train)
print "no. of Sara training emails:", len(labels_train)-sum(labels_train)
clf = tree.DecisionTreeClassifier(min_samples_split=23)
t=time.time()
clf.fit(features_train,labels_train)
print "fitting time",time.time()-t,"secs"
t=time.time()
pred=clf.predict(features_test)
print "predicting time",time.time()-t,"secs"
accuracy=accuracy_score(labels_test,pred)
print "accuracy=",accuracy
'''
### Gaussian NaiveBayes Algorithm
'''
from sklearn.naive_bayes import GaussianNB
print "no. of Chris training emails:", sum(labels_train)
print "no. of Sara training emails:", len(labels_train)-sum(labels_train)
clf = GaussianNB()
t=time.time()
clf.fit(features_train,labels_train)
print "fitting time",time.time()-t,"secs"
t=time.time()
pred=clf.predict(features_test)
print "predicting time",time.time()-t,"secs"
accuracy=accuracy_score(labels_test,pred)
print "accuracy=",accuracy
'''
### Support Vector Machine Algorithm
'''

from sklearn.svm import SVC
print "no. of Chris training emails:", sum(labels_train)
print "no. of Sara training emails:", len(labels_train)-sum(labels_train)
clf = SVC(kernel='rbf',C=10000)
t=time.time()
clf.fit(features_train,labels_train)
print "fitting time",time.time()-t,"secs"
t=time.time()
pred=clf.predict(features_test)
print "predicting time",time.time()-t,"secs"
accuracy=accuracy_score(labels_test,pred)
print "accuracy=",accuracy
'''
###K-Nearest Neighbours Algorithm
'''
from sklearn.neighbors import KNeighborsClassifier
print "no. of Chris training emails:", sum(labels_train)
print "no. of Sara training emails:", len(labels_train)-sum(labels_train)
clf = KNeighborsClassifier(n_neighbors=5)
t=time.time()
clf.fit(features_train,labels_train)
print "fitting time",time.time()-t,"secs"
t=time.time()
pred=clf.predict(features_test)
print "predicting time",time.time()-t,"secs"
accuracy=accuracy_score(labels_test,pred)
print "accuracy=",accuracy
'''

###AdaBoostClassifier Algorithm
'''
from sklearn.ensemble import AdaBoostClassifier
print "no. of Chris training emails:", sum(labels_train)
print "no. of Sara training emails:", len(labels_train)-sum(labels_train)
clf =  AdaBoostClassifier()
t=time.time()
clf.fit(features_train,labels_train)
print "fitting time",time.time()-t,"secs"
t=time.time()
pred=clf.predict(features_test)
print "predicting time",time.time()-t,"secs"
accuracy=accuracy_score(labels_test,pred)
print "accuracy=",accuracy
'''

### Random Forest Algorithm
'''
from sklearn.ensemble import RandomForestClassifier
print "no. of Chris training emails:", sum(labels_train)
print "no. of Sara training emails:", len(labels_train)-sum(labels_train)
clf =   RandomForestClassifier(min_samples_split=40)
t=time.time()
clf.fit(features_train,labels_train)
print "fitting time",time.time()-t,"secs"
t=time.time()
pred=clf.predict(features_test)
print "predicting time",time.time()-t,"secs"
accuracy=accuracy_score(labels_test,pred)
print "accuracy=",accuracy
'''
try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
