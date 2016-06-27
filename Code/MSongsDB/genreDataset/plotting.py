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
#sys.stdout = open('output.txt', 'w')

def HPOptimizationGridSearch(feature_train,label_train):
	print "Fitting the classifier to the training set"
	t0 = time.time()
	param_grid = {
	         'min_samples_split': range(1,41),
	          'min_samples_leaf': range(1,20),
	          'n_estimators': range(1,500)
	          }
	clf = GridSearchCV(OneVsRestClassifier(RandomForestClassifier()), param_grid)
	clf = clf.fit(feature_train,label_train)
	print "done in %0.3fs" % (time.time() - t0)
	print "Best estimator found by grid search:"
	print clf.best_estimator_




def processItem(datapoint):
	
	info=[datapoint[0]]
	for feature in datapoint[4:]:
		info.append(float(feature))
	return info


def targetFeatureSplit( data ):
    """ 
        given a numpy array like the one returned from
        featureFormat, separate out the first feature
        and put it into its own list (this should be the 
        quantity you want to predict)

        return targets and features as separate lists

        (sklearn can generally handle both lists and numpy arrays as 
        input formats when training/predicting)
    """
    target = []
    features = []
    for item in data:
        target.append( item[0] )
        features.append( item[1:] )

    return target, features




genres={}
data_list=[]
def done():
	line_num=0  #line number in text file
	with open('msd_genre_dataset.txt','r') as f:
		for line in f:
		    if line_num<10:
		    	pass
		    else:
		    	temp=line.split(',')
		    	try:
		    		if(genres[temp[0]] and genres[temp[0]]<2000):
		    			if(temp[0]=="hip-hop"):
		    				temp[0]='pop'	
		    			genres[temp[0]]+=1
		    			data_list.append(processItem(temp))
		    	except:
		    		genres[temp[0]]=1
		    line_num+=1
	print "**************************************************************************"
	print "No of songs: ",line_num-10
	print "**************************************************************************"

	print "Dataset composition:"
	for type in genres.keys():
		#print "\""+ type+"\",",
		print type,genres[type]
	print "**************************************************************************"
	print "Loading data in numpy arrays."
	t=time.time()
	np_data=np.array(data_list)
	print "Time elapsed:",time.time()-t,"secs."
	t=time.time()
	

	
	scaler=MinMaxScaler()
	np_data[:,1:]=scaler.fit_transform(np_data[:,1:])
  
  	print "Splitting data into target and features" 
	labels,features=targetFeatureSplit(np_data)
	print "Time elapsed:",time.time()-t,"secs."

	print "Creating training set and Test set"
	t=time.time()
	feature_train,feature_test,label_train,label_test = train_test_split(features, labels, test_size=0.20, random_state=42)
	print "Time elapsed:",time.time()-t,"secs."
	print"Fitting and predicting the data"
	
	t=time.time()

	#HPOptimizationGridSearch(feature_train,label_train)
	
	clf=OneVsRestClassifier(RandomForestClassifier(min_samples_split=2,min_samples_leaf=1,n_estimators=300))
	clf.fit(feature_train,label_train)
	pred=clf.predict(feature_test)
	print "Time elapsed:",time.time()-t,"secs."
	

	print "accuracy_score=",accuracy_score(label_test,pred)

	t=time.time()
	print "confusion_matrix:"
	print(classification_report(label_test,pred))
	print "classification_report:"
	print classification_report(label_test,pred)
	
done()