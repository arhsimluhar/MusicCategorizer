from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
stem_data=open("stemmed_text","r")
data=pickle.load(stem_data)
#print data[0]
'''
b=open("finalLyricsList.txt","w")
with open('songLyrics.txt','r') as f:
	lines=f.readlines()
	for line in lines:
		labels,features=line.split('**/**')		
		if(labels=='hip-hop'):
			b.write("pop**/**"+features)
		if(labels!='classical' and labels!='hip-hop'):
			b.write(line)
b.close()
b=open("finalLyricsList.txt","r")
lines=b.readlines()
genres={} 
for line in lines:
	labels,features=line.split('**/**')		
	try:
		genres[labels]+=1
	except:
		genres[labels]=0
print "File composition"
for g in genres.keys():
	print g+": ",genres[g]
'''		

'''

genres={}
b=open("finalLyricsList2.txt","w")
with open('finalLyricsList.txt','r') as f:
	lines=f.readlines()
	for line in lines:
		labels,features=line.split('**/**')		
		try:
			genres[labels]+=1
		except:
			genres[labels]=0
		if(genres[labels]<=1000):
			b.write(line)
b.close()
b=open("finalLyricsList2.txt","r")
lines=b.readlines()
genres={} 
for line in lines:
	labels,features=line.split('**/**')		
	try:
		genres[labels]+=1
	except:
		genres[labels]=0
print "File composition"
for g in genres.keys():
	print g+": ",genres[g]

'''
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
        features.append( item[1] ) 
    return target, features


labels,features=targetFeatureSplit(data)
#print data[0]
'''


feature_train,feature_test,label_train,label_test = train_test_split(features, labels, test_size=0.20, random_state=42)
vectorizer=CountVectorizer()
train_counts = vectorizer.fit_transform(feature_train)
print train_counts.shape
print train_counts
print vectorizer.vocabulary_.get(u'la')

'''

feature_train,feature_test,label_train,label_test = train_test_split(features, labels, test_size=0.20, random_state=42)
vectorizer=TfidfVectorizer(sublinear_tf=True,max_df=0.5)
feature_train=vectorizer.fit_transform(feature_train)
feature_test=vectorizer.transform(feature_test).toarray()


from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split=23)
clf.fit(feature_train,label_train)
pred=clf.predict(feature_test)
importances = clf.feature_importances_
import numpy as np
indices = np.argsort(importances)[::-1]
print 'Feature Ranking: '
for i in range(10):
    print "{} feature no.{} ({})".format(i+1,indices[i],importances[indices[i]])
print vectorizer.get_feature_names()[14343]
accuracy=accuracy_score(label_test,pred)
print "accuracy=",accuracy
print "confusion_matrix:"
print(classification_report(label_test,pred))
