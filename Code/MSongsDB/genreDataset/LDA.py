import numpy as np
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split



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
		    		genres[temp[0]]+=1
		    	except:
		    		genres[temp[0]]=0
		    	data_list.append(processItem(temp))
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
	#np_data=np.array(data_list)
	print "Time elapsed:",time.time()-t,"secs."
	t=time.time()


	print "Splitting data into target and features" 
	labels,features=targetFeatureSplit(data_list)
	print "Time elapsed:",time.time()-t,"secs."
	
	#print type(labels),type(features)
	
	features=np.array(features).astype(np.float)
	scaler=MinMaxScaler()
	features=scaler.fit_transform(features)
	
	

	print "Creating training set and Test set"
	t=time.time()
	feature_train,feature_test,label_train,label_test = train_test_split(features, labels, test_size=0.20, random_state=42)
	print "Time elapsed:",time.time()-t,"secs."
	print"Fitting and predicting the data"
	
	t=time.time()

	#HPOptimizationGridSearch(feature_train,label_train)
	clf = LinearDiscriminantAnalysis()
	clf.fit(feature_train, label_train)
	pred=clf.predict(feature_test)
	print "Time elapsed:",time.time()-t,"secs."
	

	print "accuracy_score=",accuracy_score(label_test,pred)

	t=time.time()
	print "confusion_matrix:"
	print(classification_report(label_test,pred))
	
done()