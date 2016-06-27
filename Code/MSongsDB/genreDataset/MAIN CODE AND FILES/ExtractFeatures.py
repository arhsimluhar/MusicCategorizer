'''
This file is extracting out the features from the million song genre dataset as per our requirement.
Making even ,balanced dataset to work upon.

'''

'''

No of songs:  59600
**************************************************************************
Dataset composition:
jazz and blues 4334
classic pop and rock 23895
classical 1874
punk 3200
metal 2103
pop 1617
dance and electronica 4935
hip-hop 434
soul and reggae 4016
folk 13192


'''


'''

No of songs:  59600
**************************************************************************
Dataset composition:
jazz and blues 4334
classic pop and rock 23895
classical 1874
punk 3200
metal 2103
pop 2051
dance and electronica 4935
soul and reggae 4016
folk 13192


'''


'''

No of songs:  59600
**************************************************************************
Dataset composition:
jazz and blues:  2001
classic pop and rock:  2001
classical:  1874
punk:  2001
metal:  2001
pop:  2001
dance and electronica:  2001
soul and reggae:  2001
folk:  2001



'''


##########################################################################
'''
Helper functions for pre-processing the data before creating pickle object



'''
def processdata(lines):
	features=[]
	labels=[]
	for line in lines:
		temp=line.split(',')
		labels.append(temp[0])
		l=[]
		for feature in temp[4:]:
			l.append(float(feature))
		features.append(l)
	return labels,features



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






#######################################################################





import numpy as np
import os
import pickle
from sklearn.preprocessing import MinMaxScaler


b=open("balanced_msd.txt",'w')
line_number=0
genres={}
with open('msd_genre_dataset.txt','r') as f:
	lines=f.readlines()
	for line in lines:
		line_number+=1
		if line_number>10:
			temp=line.split(',')
			if(temp[0]=='hip-hop'):
				temp[0]='pop'
			b.write(",".join(temp))
			try:
				genres[temp[0]]+=1
			except:
				genres[temp[0]]=1	
'''
print "File composition"
for g in genres.keys():
	print g+": ",genres[g]
'''
b.close()



###########################################################################
# phase 2: balancing the dataset
###########################################################################



genres={}
b=open("balanced_msd_final.txt",'w')
with open('balanced_msd.txt','r') as f:
	lines=f.readlines()
	for line in lines:
		temp=line.split(',')
		try:
			if(genres[temp[0]]<=2000):
				genres[temp[0]]+=1
				b.write(",".join(temp))

		except:
			genres[temp[0]]=1
			b.write(",".join(temp))
try:
	os.remove('balanced_msd.txt')
except:
	print "balanced_msd.txt is missing "

		



print "#######################################################################"				
print "                          Dataset composition"
print "#######################################################################"				

for g in genres.keys():
	print g+": ",genres[g]
b.close()



################################################################################
'''
Formatting the data into numpy arrays ,suitable using them on the go. 


Using pickle to save the serialised object for future analysis of dataset
by various algorithms
'''
################################################################################

with open('balanced_msd_final.txt','r') as f:
	lines=f.readlines()
	labels,features=processdata(lines)
np_features=np.array(features)
labels=np.array(labels)

###########################
#feature Scaling on dataset 
###########################
scaler=MinMaxScaler()
np_features=scaler.fit_transform(np_features)



########################
#pickling the data
########################
#print len(features)
#print len(np_features)

f=open("features","w")
pickle.dump(labels,f)
pickle.dump(np_features,f)
f.close()



print "Successfully did preprocessing without any error"






		
