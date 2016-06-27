'''
In this file we are working with various classification algorithms 
over our data set

'''

import numpy as np
import math
import sys
import matplotlib
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
import time
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import RandomizedPCA
import pickle as pkl
import matplotlib.pyplot as plt


#Loading labels and features from the pickle file
f=open('features','r')
labels=pkl.load(f)
features=pkl.load(f)
f.close()
print type(labels),type(features)


#spliting data into training and testing data
print "spliting the data into training and testing set"
feature_train,feature_test,label_train,label_test = train_test_split(features, labels, test_size=0.20, random_state=42)
print "Defining the classifier"

# Gaussian Naive Byaes Classifier

clf = OneVsRestClassifier(GaussianNB())
x=[]
y=[]
print len(feature_train)
print len(label_train)
for a in range(1,50):
	pred = clf.fit(feature_train[:len(feature_train)/a],label_train[:len(feature_train)/a]).predict(feature_test)
	#print "accuracy_score=",accuracy_score(label_test,pred)
	#print "confusion_matrix:"
	#print(classification_report(label_test,pred))
	x.append(len(feature_train)/a)
	y.append(accuracy_score(label_test,pred))
plt.plot(x,y)
plt.title('Gaussian Naive Bayes classifier')
plt.xlabel('No. of Training Samples')
plt.ylabel('Accuracy Score')
plt.show()


'''DecisionTreeClassifier

x=[]
y=[]
for min_split in range(1,400,5):
	clf = tree.DecisionTreeClassifier(min_samples_split=min_split)
	clf.fit(feature_train,label_train)
	pred=clf.predict(feature_test)
	accuracy=accuracy_score(label_test,pred)	
	x.append(min_split)
	y.append(accuracy)
	print "k :",min_split
plt.plot(x,y)
plt.title('Decision Tree Classifier Optimization')
plt.xlabel('Value of min_samples_split')
plt.ylabel('Accuracy Score')
plt.show()
'''

''' DecisionTreeClassifier Alternate Method

clf = tree.DecisionTreeClassifier()

k_range=range(1,5)
param_grid=dict(min_samples_split=k_range)
grid=GridSearchCV(clf,param_grid,cv=10,scoring='accuracy')
grid.fit(features,labels)
print grid.grid_scores_

grid_mean_scores=[result.mean_validation_score for result in grid.grid_scores_]
print grid_mean_scores
plt.plot(k_range,grid_mean_scores)
plt.title('Min_Samples_Split Optimization')
plt.xlabel('value of k for KNN')
plt.ylabel('cross validated accuracy')
plt.show()
print grid.best_score_
print grid.best_params_
print grid.best_estimator_
'''

'''KNeighborsClassifier
x=[]
y=[]
for k in range(1,400,5):
	clf = KNeighborsClassifier(n_neighbors=k)
	clf.fit(feature_train,label_train)
	pred=clf.predict(feature_test)
	accuracy=accuracy_score(label_test,pred)	
	x.append(k)
	y.append(accuracy)
	print "k :",k
plt.plot(x,y)
plt.title('KNeighnors Classifier Optimization')
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.show()
'''

