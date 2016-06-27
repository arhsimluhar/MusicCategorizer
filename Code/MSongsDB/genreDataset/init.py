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

import numpy
import math
import sys
import matplotlib


######################################################################################
#
#                              HELPER FUNCTIONS
#
######################################################################################

def processItem(datapoint):
	info=datapoint[:4]
	for feature in datapoint[4:]:
		info.append(float(feature))
	return info




"""
MAIN CODE 

"""
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
with open('msd_genre_dataset.txt','r') as f:
	for line in f:	
	    line_num+=1
print "No of songs: ",line_num


def HPOptimizationGridSearch(feature_train,label_train):
	print "Fitting the classifier to the training set"
	t0 = time()
	param_grid = {
	         'min_samples_split': range[1:41],
	          'min_samples_leaf': range[1:20],
	          'n_estimators': range[1:500]
	          }
	clf = GridSearchCV(OneVsRestClassifier(RandomForestClassifier()), param_grid)
	clf = clf.fit(feature_train,label_train)
	print "done in %0.3fs" % (time() - t0)
	print "Best estimator found by grid search:"
	print clf.best_estimator_



        
