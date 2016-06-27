#Features in extracted from genre_dataset
#####################################################################
'''
Features                        Features Description 


genre,							Genre of the song
track_id,						Musicmatch track_id
artist_name,					Artist of the song
title,							Title of the song
loudness,						Loudness is the characteristic of a sound that is primarily a psychological correlate of physical strength (amplitude).
tempo,							The tempo of a piece of music is the speed of the underlying beat. Tempo is measured in BPM, or Beats Per Minute.
time_signature,					The top number represents how many beats there are in a measure, and the bottom number represents the note value which makes one beat.
key,							The key of a piece is a group of pitches, or scale upon which a music composition is created.	
mode,							Refers to a type of scale, coupled with a set of characteristic melodic behaviours. 
duration,                       song duration (in Secs)
avg_timbre1,					Timbre is then a general term for the distinguishable characteristics of a tone.
avg_timbre2,					Timbre is mainly determined by the harmonic content of a sound 
avg_timbre3,					and the dynamic characteristics.
avg_timbre4,
avg_timbre5,
avg_timbre6,
avg_timbre7,
avg_timbre8,
avg_timbre9,
avg_timbre10,
avg_timbre11,
avg_timbre12,
var_timbre1,
var_timbre2,
var_timbre3,
var_timbre4,
var_timbre5,
var_timbre6,
var_timbre7,
var_timbre8,
var_timbre9,
var_timbre10,
var_timbre11,
var_timbre12
'''
######################################################################################





######################################################################################
#                               ABOUT FILE
#
#File Contains code for Extracting Information from the "msd_genre_dataset.txt" file
#that contains all the features and labels that will be helpful in 
#predicting genres.
######################################################################################



######################################################################################
#
#                     Library Import Statements
#
#
######################################################################################

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
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
######################################################################################
#
#                              HELPER FUNCTIONS
#
######################################################################################

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


data_list=[]
genres=["jazz and blues","classic pop and rock","classical","punk","metal","pop","dance and electronica","hip-hop","soul and reggae","folk"]
def preprocess():
	#open msd_genre_dataset.txt in read mode

	line_num=0  #line number in text file
	#################################################################
	'''
	Steps
	1. skip 10 lines as no useful data is there.
	2. Now, Read line by line and store that info in appropriate data structure for further analysis 
	'''
	############################################################
	'''
	The with statement handles opening and closing the file, 
	including if an exception is raised in the inner block. 
	The for line in f treats the file object f as an iterable, 
	which automatically uses buffered IO and memory management 
	so you don't have to worry about large files.
	'''
	############################################################
	genres={}

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
	

	from sklearn.preprocessing import MinMaxScaler
	scaler=MinMaxScaler()
	np_data=scaler.fit_transform(np_data[:,1:])

	print "Splitting data into target and features" 
	labels,features=targetFeatureSplit(np_data)
	print "Time elapsed:",time.time()-t,"secs."
	print "Creating training set and Test set"
	t=time.time()
	feature_train,feature_test,label_train,label_test = train_test_split(features, labels, test_size=0.20, random_state=42)
	print "Time elapsed:",time.time()-t,"secs."
	print"Fitting and predicting the data"
	t=time.time()
	pred= OneVsRestClassifier(RandomForestClassifier(min_samples_split=30).fit(feature_train,label_train).predict(feature_test) 
	#print "confusion_matrix:"
	#print(classification_report(label_test,pred))
